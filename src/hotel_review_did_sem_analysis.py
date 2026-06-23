"""Hotel review questionnaire analysis: DID and PLS-SEM-style paths.

The script is a cleaned, reusable version of the project workflow. It expects two
Wenjuanxing-style Excel exports: one for the real-review condition and one for
the fake-review condition. Column selection is intentionally configurable because
questionnaire exports often include long Chinese question labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import StandardScaler


DEFAULT_ITEM_MAP = {
    "pre_booking": ["PRE1", "PRE2", "PRE3", "PRE4"],
    "authenticity": ["AUTH1", "AUTH2", "AUTH3", "AUTH4"],
    "trust": ["TRUST1", "TRUST2", "TRUST3", "TRUST4"],
    "risk": ["RISK1", "RISK2", "RISK3", "RISK4"],
    "booking": ["BI1", "BI2", "BI3", "BI4", "BI5", "BI6", "BI7"],
}


def cronbach_alpha(df: pd.DataFrame) -> float:
    values = df.dropna().to_numpy(dtype=float)
    if values.shape[1] < 2 or values.shape[0] < 2:
        return np.nan
    item_vars = values.var(axis=0, ddof=1)
    total_var = values.sum(axis=1).var(ddof=1)
    return values.shape[1] / (values.shape[1] - 1) * (1 - item_vars.sum() / total_var)


def load_condition(path: Path, condition: str, review_type: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Condition"] = condition
    df["ReviewType"] = review_type
    return df


def make_scores(df: pd.DataFrame, item_map: dict[str, list[str]]) -> pd.DataFrame:
    scores = pd.DataFrame({"Condition": df["Condition"], "ReviewType": df["ReviewType"]})
    for construct, cols in item_map.items():
        available = [c for c in cols if c in df.columns]
        if available:
            scores[construct] = df[available].mean(axis=1)
    return scores


def run_did(scores: pd.DataFrame) -> pd.DataFrame:
    long_df = pd.concat(
        [
            scores[["Condition", "ReviewType", "pre_booking"]].rename(columns={"pre_booking": "booking_intention"}).assign(Post=0),
            scores[["Condition", "ReviewType", "booking"]].rename(columns={"booking": "booking_intention"}).assign(Post=1),
        ],
        ignore_index=True,
    )
    model = smf.ols("booking_intention ~ ReviewType * Post", data=long_df).fit(cov_type="HC3")
    coef = model.params.get("ReviewType:Post", np.nan)
    pval = model.pvalues.get("ReviewType:Post", np.nan)
    ci_low, ci_high = model.conf_int().loc["ReviewType:Post"] if "ReviewType:Post" in model.params else (np.nan, np.nan)
    return pd.DataFrame(
        [{"metric": "DID estimate", "value": coef}, {"metric": "p-value", "value": pval}, {"metric": "95% CI low", "value": ci_low}, {"metric": "95% CI high", "value": ci_high}]
    )


def run_path_model(scores: pd.DataFrame) -> pd.DataFrame:
    renamed = scores.rename(
        columns={
            "pre_booking": "PreBookingIntention",
            "authenticity": "ReviewAuthenticity",
            "trust": "Trust",
            "risk": "PerceivedRisk",
            "booking": "BookingIntention",
        }
    ).dropna()
    constructs = ["ReviewType", "PreBookingIntention", "ReviewAuthenticity", "Trust", "PerceivedRisk", "BookingIntention"]
    z = pd.DataFrame(StandardScaler().fit_transform(renamed[constructs]), columns=constructs)

    formulas = {
        "ReviewAuthenticity": "ReviewAuthenticity ~ ReviewType",
        "Trust": "Trust ~ ReviewType + ReviewAuthenticity",
        "PerceivedRisk": "PerceivedRisk ~ ReviewType + ReviewAuthenticity",
        "BookingIntention": "BookingIntention ~ ReviewType + PreBookingIntention + ReviewAuthenticity + Trust + PerceivedRisk",
    }
    rows = []
    for target, formula in formulas.items():
        model = smf.ols(formula, data=z).fit(cov_type="HC3")
        for predictor, beta in model.params.items():
            if predictor == "Intercept":
                continue
            rows.append({"Path": f"{predictor} -> {target}", "Beta": beta, "p_value": model.pvalues[predictor], "R2_target": model.rsquared})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-file", required=True)
    parser.add_argument("--fake-file", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    real = load_condition(Path(args.real_file), "Real review", 1)
    fake = load_condition(Path(args.fake_file), "Fake review", -1)
    raw = pd.concat([real, fake], ignore_index=True)

    scores = make_scores(raw, DEFAULT_ITEM_MAP)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores.to_csv(outdir / "construct_scores.csv", index=False)
    run_did(scores).to_csv(outdir / "did_summary.csv", index=False)
    run_path_model(scores).to_csv(outdir / "path_model_summary.csv", index=False)

    reliability = []
    for construct, cols in DEFAULT_ITEM_MAP.items():
        available = [c for c in cols if c in raw.columns]
        if available:
            reliability.append({"construct": construct, "items": len(available), "cronbach_alpha": cronbach_alpha(raw[available])})
    pd.DataFrame(reliability).to_csv(outdir / "reliability.csv", index=False)


if __name__ == "__main__":
    main()
