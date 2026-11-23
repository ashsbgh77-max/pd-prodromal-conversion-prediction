# %% [markdown]
# # Cross-validated radiogenomic prediction of phenoconversion in prodromal Parkinson's disease
# Reproducible pipeline for the analyses reported in the manuscript.
# Data: PPMI Curated Data Cut (controlled access, not included in this repository).
# See README.md for data access and how to run.

# %% Setup
import os, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test, logrank_test
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---- Configuration: edit these paths --------------------------------------
PPMI_XLSX   = Path("data/PPMI_Curated_Data_Cut_Public_20260810.xlsx")   # PPMI curated data cut
PPMI_SHEET  = "20260810"
CARRIER_CSV = Path("data/iu_genetic_consensus_20251025_30Sep2026.csv")  # LRRK2/GBA carrier file (secondary analysis)
OUT         = Path("outputs"); OUT.mkdir(exist_ok=True)
N_SPLITS, N_REPEATS, PENALIZER = 5, 20, 0.1

DAT  = ["MIA_PUTAMEN_L","MIA_PUTAMEN_R","MIA_CAUDATE_L","MIA_CAUDATE_R","MIA_STRIATUM_L","MIA_STRIATUM_R"]
GEN  = ["GP2_proxy_PGS","META5_proxy_PGS"]
MUT  = ["LRRK2_carrier","GBA_carrier"]
CLIN = ["age","SEX","updrs3_score","BMI","NP1DPRS","MSEADLG"]
PRIMARY   = {"M1":CLIN,"M2":GEN,"M3":DAT,"M4":GEN+CLIN,"M5":DAT+CLIN,"M6":DAT+GEN,"M7":DAT+GEN+CLIN}
SECONDARY = {"M2b":MUT,"M2c":GEN+MUT,"M4b":GEN+MUT+CLIN,"M7b":DAT+GEN+MUT+CLIN}
DAT_NO_STRIATUM = ["MIA_PUTAMEN_L","MIA_PUTAMEN_R","MIA_CAUDATE_L","MIA_CAUDATE_R"]

# %% 1. Cohort construction
def build_cohort(df):
    """Prodromal cohort (COHORT == 4) with complete baseline predictors; time to first later PD diagnosis.
    Returns the analytic cohort and a dict with the participant flow counts."""
    flow = {}
    df = df.copy()
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    baseline = df[(df["EVENT_ID"]=="BL") & (df["PRIMDIAG"]!=1) & (df["COHORT"]==4)].copy()
    flow["prodromal_baseline"] = baseline["PATNO"].nunique()
    baseline = baseline.sort_values(["PATNO","visit_date"]).drop_duplicates("PATNO", keep="first")
    predictors = GEN + DAT + CLIN
    complete = baseline.dropna(subset=predictors).copy()
    flow["complete_predictors"] = complete["PATNO"].nunique()

    fu = df[df["PATNO"].isin(complete["PATNO"])].merge(
        complete[["PATNO","visit_date"]].rename(columns={"visit_date":"baseline_date"}), on="PATNO")
    fu = fu[fu["visit_date"] > fu["baseline_date"]]
    first_conv = (fu[fu["PRIMDIAG"]==1].sort_values(["PATNO","visit_date"])
                  .groupby("PATNO", as_index=False).first()[["PATNO","visit_date"]]
                  .rename(columns={"visit_date":"conversion_date"}))
    last_fu = (fu.dropna(subset=["visit_date"]).groupby("PATNO", as_index=False)["visit_date"].max()
               .rename(columns={"visit_date":"last_followup_date"}))
    a = complete.merge(first_conv, on="PATNO", how="left").merge(last_fu, on="PATNO", how="left")
    a["event"] = a["conversion_date"].notna().astype(int)
    a["end_date"] = pd.to_datetime(np.where(a["event"]==1, a["conversion_date"], a["last_followup_date"]), errors="coerce")
    a["follow_up_days"]  = (a["end_date"] - a["visit_date"]).dt.days
    a["duration"] = a["follow_up_days"] / 365.25
    flow["no_follow_up_after_baseline"] = int(a["follow_up_days"].isna().sum())
    flow["negative_follow_up"]          = int((a["follow_up_days"] < 0).sum())
    a = a[a["follow_up_days"].notna() & (a["follow_up_days"] >= 0)].copy()
    flow["final"], flow["events"] = len(a), int(a["event"].sum())
    keep = ["PATNO","COHORT","visit_date","conversion_date","last_followup_date","end_date","event",
            "follow_up_days","duration"] + CLIN + DAT + GEN
    a = a[keep].rename(columns={"visit_date":"baseline_date"})
    if a["SEX"].dtype == object:
        a["SEX"] = a["SEX"].astype("category").cat.codes
    return a, flow

def add_carrier_status(cohort, carrier_csv):
    """Secondary analyses only: restrict to participants with LRRK2/GBA data."""
    mut = pd.read_csv(carrier_csv)
    for g in ("LRRK2","GBA"):
        mut[f"{g}_carrier"] = mut[g].apply(lambda x: np.nan if pd.isna(x) else (0 if str(x).strip() in ("0","0.0") else 1))
    m = cohort.merge(mut[["PATNO"]+MUT], on="PATNO", how="left").dropna(subset=MUT).copy()
    return m

# %% 2. Analysis functions
def cv_cindex(df, cols, n_splits=N_SPLITS, n_repeats=N_REPEATS, penalizer=PENALIZER):
    """Repeated k-fold CV. Standardisation is fitted on the training fold only (no leakage)."""
    out = []
    for r in range(n_repeats):
        for tr, te in KFold(n_splits, shuffle=True, random_state=r).split(df):
            a, b = df.iloc[tr].copy(), df.iloc[te].copy()
            sc = StandardScaler().fit(a[cols]); a[cols] = sc.transform(a[cols]); b[cols] = sc.transform(b[cols])
            m = CoxPHFitter(penalizer=penalizer).fit(a[["duration","event"]+cols], "duration", "event")
            out.append(concordance_index(b["duration"], -m.predict_partial_hazard(b[cols]), b["event"]))
    assert len(out) == n_splits*n_repeats, "a fold failed"
    return np.array(out)

def run_models(df, models, **kw):
    return {k: cv_cindex(df, v, **kw) for k, v in models.items()}

def nadeau_bengio(a, b, n_splits=N_SPLITS):
    """Nadeau-Bengio corrected resampled paired t-test (Nadeau & Bengio 2003)."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    t = d.mean() / np.sqrt((1/J + 1/(n_splits-1)) * d.var(ddof=1))
    return d.mean(), t, 2*stats.t.sf(abs(t), J-1)

def compare(res, pairs, holm=True):
    rows = [(f"{a} vs {b}",) + (lambda r: (round(r[0],4), round(r[2],4)))(nadeau_bengio(res[a], res[b])) for a, b in pairs]
    t = pd.DataFrame(rows, columns=["comparison","mean_diff","NB_p"])
    if holm: t["holm_p"] = np.round(multipletests(t["NB_p"], method="holm")[1], 4)
    return t

def summarise(res):
    return pd.DataFrame({k: {"mean_cindex": v.mean(), "sd": v.std(), "n_folds": len(v)} for k, v in res.items()}).T.round(4)

def fit_full(df, cols, penalizer=PENALIZER):
    X = df[["duration","event"]+cols].copy(); X[cols] = StandardScaler().fit_transform(X[cols])
    return CoxPHFitter(penalizer=penalizer).fit(X, "duration", "event"), X

def ph_and_vif(df, cols):
    cph, X = fit_full(df, cols)
    ph = proportional_hazard_test(cph, X, time_transform="rank").summary
    dat = [c for c in cols if c.startswith("MIA_")]
    Z = StandardScaler().fit_transform(df[dat])
    vif = pd.Series([variance_inflation_factor(Z, i) for i in range(len(dat))], index=dat, name="VIF")
    return cph, ph, vif

def perm_importance(df, cols, n_repeats=100, seed=42):
    rng = np.random.default_rng(seed)
    cph, X = fit_full(df, cols)
    base = concordance_index(X["duration"], -cph.predict_partial_hazard(X[cols]), X["event"])
    imp = {}
    for c in cols:
        d = []
        for _ in range(n_repeats):
            S = X.copy(); S[c] = rng.permutation(S[c].values)
            d.append(base - concordance_index(S["duration"], -cph.predict_partial_hazard(S[cols]), S["event"]))
        imp[c] = (np.mean(d), np.std(d))
    return base, imp

NICE = {"updrs3_score":"MDS-UPDRS Part III","MIA_PUTAMEN_L":"Putamen SBR, left","MIA_PUTAMEN_R":"Putamen SBR, right",
        "MIA_CAUDATE_L":"Caudate SBR, left","MIA_CAUDATE_R":"Caudate SBR, right","MIA_STRIATUM_L":"Striatum SBR, left",
        "MIA_STRIATUM_R":"Striatum SBR, right","GP2_proxy_PGS":"GP2 PRS","META5_proxy_PGS":"META5 PRS","age":"Age",
        "SEX":"Sex","BMI":"BMI","NP1DPRS":"Depression item (NP1DPRS)","MSEADLG":"Schwab & England ADL"}

def fig1(res, path):
    keys = list(PRIMARY); m = [res[k].mean() for k in keys]; s = [res[k].std() for k in keys]
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    bars = ax.bar(range(len(keys)), m, yerr=s, capsize=3, color="lightgray", edgecolor="black"); bars[-1].set_hatch("//")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    ax.set_ylabel("Concordance index (C-index)"); ax.set_ylim(0.5, 0.85)
    plt.tight_layout(); plt.savefig(path, dpi=600); plt.close()

def fig2(imp, path):
    items = sorted(imp.items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(figsize=(6.5,5))
    ax.barh([NICE.get(k,k) for k,_ in items], [v[0] for _,v in items], xerr=[v[1] for _,v in items],
            color="lightgray", edgecolor="black", hatch="//")
    ax.axvline(0, color="black", lw=.8); ax.set_xlabel("Mean reduction in C-index when permuted")
    plt.tight_layout(); plt.savefig(path, dpi=600); plt.close()

def fig3(df, cph, cols, path):
    X = df[cols].copy(); X[cols] = StandardScaler().fit_transform(X[cols])
    risk = cph.predict_partial_hazard(X); hi = risk >= risk.median()
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    for mask, lab, ls in [(hi,"High risk","-"), (~hi,"Low risk","--")]:
        KaplanMeierFitter().fit(df.loc[mask,"duration"], df.loc[mask,"event"], label=lab).plot_survival_function(ax=ax, ci_show=True, color="black", linestyle=ls)
    ax.set_xlabel("Years from baseline"); ax.set_ylabel("Phenoconversion-free probability")
    plt.tight_layout(); plt.savefig(path, dpi=600); plt.close()
    return logrank_test(df.loc[hi,"duration"], df.loc[~hi,"duration"], df.loc[hi,"event"], df.loc[~hi,"event"]).p_value

# %% 3. Run the analysis
def main(n_repeats=N_REPEATS, perm_repeats=100):
    raw = pd.read_excel(PPMI_XLSX, sheet_name=PPMI_SHEET)
    cohort, flow = build_cohort(raw)
    print("Participant flow:", flow)
    pd.Series(flow).to_csv(OUT/"participant_flow.csv")

    # primary analysis (all prodromal participants with complete data)
    res = run_models(cohort, PRIMARY, n_repeats=n_repeats)
    summarise(res).to_csv(OUT/"table_cindex_primary.csv")
    cmp_primary = compare(res, [("M7","M3"),("M2","M1"),("M7","M5")]); cmp_primary.to_csv(OUT/"table_pairwise_primary.csv")
    print(summarise(res)); print(cmp_primary)

    # proportional hazards, collinearity, hazard ratios, KM, permutation importance
    cph, ph, vif = ph_and_vif(cohort, DAT+GEN+CLIN)
    ph.to_csv(OUT/"schoenfeld_test.csv"); vif.to_csv(OUT/"vif_full_model.csv")
    cph.summary[["coef","exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].to_csv(OUT/"hazard_ratios_full_model.csv")
    print("log-rank p:", fig3(cohort, cph, DAT+GEN+CLIN, OUT/"Fig3_KM.png"))
    base, imp = perm_importance(cohort, DAT+GEN+CLIN, n_repeats=perm_repeats)
    pd.DataFrame(imp, index=["mean","sd"]).T.to_csv(OUT/"permutation_importance.csv")
    fig1(res, OUT/"Fig1_cindex.png"); fig2(imp, OUT/"Fig2_perm.png")

    # sensitivity analysis: excluding striatum variables
    res_ns = run_models(cohort, {"M3_ns":DAT_NO_STRIATUM, "M5_ns":DAT_NO_STRIATUM+CLIN, "M7_ns":DAT_NO_STRIATUM+GEN+CLIN}, n_repeats=n_repeats)
    summarise(res_ns).to_csv(OUT/"table_cindex_no_striatum.csv")
    compare({**res, **res_ns}, [("M7_ns","M5_ns"),("M7_ns","M7"),("M7_ns","M3_ns")]).to_csv(OUT/"table_pairwise_no_striatum.csv")

    # secondary analysis: LRRK2/GBA carrier status (participants with carrier data only)
    sub = add_carrier_status(cohort, CARRIER_CSV)
    print("Participants with carrier data:", len(sub), "| LRRK2/GBA carriers:", sub[MUT].sum().to_dict())
    res2 = run_models(sub, {**PRIMARY, **SECONDARY}, n_repeats=n_repeats)
    summarise(res2).to_csv(OUT/"table_cindex_with_carrier_status.csv")
    compare(res2, [("M7b","M5"),("M7b","M7"),("M2c","M2"),("M4b","M1")]).to_csv(OUT/"table_pairwise_carrier_status.csv")

if __name__ == "__main__":
    main()
