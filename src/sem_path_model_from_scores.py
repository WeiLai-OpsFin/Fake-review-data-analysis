"""Reproduce the SEM-style path model from checked-in construct scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


PATH_FORMULAS = {
    "ReviewAuthenticity": "ReviewAuthenticity ~ ReviewType",
    "Trust": "Trust ~ ReviewType + ReviewAuthenticity",
    "PerceivedRisk": "PerceivedRisk ~ ReviewType + ReviewAuthenticity",
    "BookingIntention": "BookingIntention ~ ReviewType + PreBookingIntention + ReviewAuthenticity + Trust + PerceivedRisk",
}


def fit_paths(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "ReviewType",
        "PreBookingIntention",
        "ReviewAuthenticity",
        "Trust",
        "PerceivedRisk",
        "BookingIntention",
    ]
    z = pd.DataFrame(StandardScaler().fit_transform(scores[columns]), columns=columns)

    path_rows = []
    r2_rows = []
    for target, formula in PATH_FORMULAS.items():
        model = smf.ols(formula, data=z).fit(cov_type="HC3")
        r2_rows.append({"Construct": target, "R2": model.rsquared, "Adjusted_R2": model.rsquared_adj})
        for predictor, beta in model.params.items():
            if predictor == "Intercept":
                continue
            path_rows.append(
                {
                    "Path": f"{predictor} -> {target}",
                    "Beta": beta,
                    "Robust_SE": model.bse[predictor],
                    "p_value": model.pvalues[predictor],
                }
            )
    return pd.DataFrame(path_rows), pd.DataFrame(r2_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-file", default="data/pls_sem_construct_scores.csv")
    parser.add_argument("--outdir", default="results/reproduced")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores_file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths, r2 = fit_paths(scores)
    paths.to_csv(outdir / "path_model_from_scores.csv", index=False)
    r2.to_csv(outdir / "r2_from_scores.csv", index=False)

    print(paths.to_string(index=False))
    print()
    print(r2.to_string(index=False))


if __name__ == "__main__":
    main()
