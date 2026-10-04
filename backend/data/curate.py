"""Curate DRAMP into the two labelled datasets Q-Peptide trains on.

Outputs (under data/processed/):
    activity.csv   -- sequence, y_activity (pMIC), mic_um, n_mic_values, cluster
    hemolysis.csv  -- sequence, y_hemolysis (p-dose), dose_um, cluster
    hemolysis_cls.csv -- sequence, label (0/1) from explicit statements only
    curation_report.json -- every count, so dropped rows are accounted for

Run:  python -m backend.data.curate
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.labels import (
    ParseStats,
    is_nodata,
    p_scale,
    parse_hemolysis,
    parse_mic,
)
from backend.utils.peptide import clean_sequence, molecular_weight

RAW_DEFAULT = Path("data/raw/general_amps.xlsx")
PROCESSED_DIR = Path("data/processed")

# Length window. Lower bound 5 excludes fragments too short for meaningful
# physicochemical descriptors; upper bound 60 keeps the set in the peptide regime where
# AMP mechanisms (membrane disruption by amphipathic helices) apply, and matches the
# length range used when the dataset was characterised.
MIN_LEN = 5
MAX_LEN = 60


@dataclass
class CurationResult:
    activity: pd.DataFrame
    hemolysis: pd.DataFrame
    hemolysis_cls: pd.DataFrame
    report: dict


def _collapse_duplicates(
    df: pd.DataFrame, value_col: str, how: str = "median"
) -> pd.DataFrame:
    """Collapse repeated sequences to one row, aggregating the label.

    Duplicate sequences are the most direct form of leakage: the same peptide appearing
    in train and test guarantees optimistic metrics. They are collapsed here, before any
    splitting, and the number of contributing measurements is kept so disagreement
    between sources stays visible.
    """
    agg = (
        df.groupby("sequence", as_index=False)
        .agg(
            **{
                value_col: (value_col, how),
                f"{value_col}_std": (value_col, "std"),
                "n_measurements": (value_col, "size"),
            }
        )
    )
    agg[f"{value_col}_std"] = agg[f"{value_col}_std"].fillna(0.0)
    return agg


def _refs_by_sequence(rows: pd.DataFrame) -> dict[str, str]:
    """Map sequence -> sorted, semicolon-joined PubMed IDs backing its measurements."""
    if "Pubmed_ID" not in rows.columns:
        return {}
    out: dict[str, str] = {}
    for seq, grp in rows.groupby("sequence")["Pubmed_ID"]:
        ids: set[str] = set()
        for v in grp.dropna().astype(str):
            for token in re.split(r"[^0-9]+", v):
                if token:
                    ids.add(token)
        out[seq] = ";".join(sorted(ids))
    return out


def curate(raw_path: Path | str = RAW_DEFAULT) -> CurationResult:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Download it first with "
            "`python -m backend.data.download`."
        )

    df = pd.read_excel(raw_path)
    stats = ParseStats(rows_total=len(df))

    # --- sequence validation -------------------------------------------------
    df["sequence"] = df["Sequence"].map(clean_sequence)
    stats.invalid_sequence = int(df["sequence"].isna().sum())
    df = df[df["sequence"].notna()].copy()

    before = len(df)
    lengths = df["sequence"].str.len()
    df = df[(lengths >= MIN_LEN) & (lengths <= MAX_LEN)].copy()
    stats.length_filtered = before - len(df)

    df["mw"] = df["sequence"].map(molecular_weight)

    # --- activity: pMIC ------------------------------------------------------
    df["mic_um"] = [
        parse_mic(t, mw) for t, mw in zip(df["Target_Organism"], df["mw"])
    ]
    act_rows = df[df["mic_um"].notna()].copy()
    stats.mic_parsed = len(act_rows)
    act_rows["y_activity"] = act_rows["mic_um"].map(p_scale)

    activity = _collapse_duplicates(
        act_rows[["sequence", "y_activity"]], "y_activity"
    )
    # keep the geometric-mean MIC alongside for reporting
    mic_by_seq = act_rows.groupby("sequence")["mic_um"].median()
    activity["mic_um"] = activity["sequence"].map(mic_by_seq)
    # Study provenance. Two sequences measured in the SAME publication share lab and
    # protocol; two measured in different publications may differ several-fold for
    # reasons that have nothing to do with sequence. Pair-level analyses need to be able
    # to separate those cases, so the source references are carried through here.
    activity["pubmed_ids"] = activity["sequence"].map(_refs_by_sequence(act_rows))

    # --- hemolysis -----------------------------------------------------------
    parsed = [
        parse_hemolysis(t, mw)
        for t, mw in zip(df["Hemolytic_activity"], df["mw"])
    ]
    df["hemo_dose_um"] = [p.dose_um for p in parsed]
    df["hemo_qual"] = [p.qualitative for p in parsed]
    df["hemo_ambiguous"] = [p.ambiguous for p in parsed]
    df["hemo_nodata"] = [p.nodata for p in parsed]

    stats.hemo_nodata = int(df["hemo_nodata"].sum())
    stats.hemo_dose_parsed = int(df["hemo_dose_um"].notna().sum())
    stats.hemo_ambiguous = int(df["hemo_ambiguous"].sum())
    stats.hemo_qualitative = int(
        ((df["hemo_qual"] != 0) & ~df["hemo_ambiguous"]).sum()
    )
    stats.hemo_unparseable = int(
        (
            ~df["hemo_nodata"]
            & df["hemo_dose_um"].isna()
            & (df["hemo_qual"] == 0)
        ).sum()
    )

    # regression set: quantitative 50%-effect dose only
    hem_rows = df[df["hemo_dose_um"].notna()].copy()
    hem_rows["y_hemolysis"] = hem_rows["hemo_dose_um"].map(p_scale)
    hemolysis = _collapse_duplicates(
        hem_rows[["sequence", "y_hemolysis"]], "y_hemolysis"
    )
    dose_by_seq = hem_rows.groupby("sequence")["hemo_dose_um"].median()
    hemolysis["dose_um"] = hemolysis["sequence"].map(dose_by_seq)
    hemolysis["pubmed_ids"] = hemolysis["sequence"].map(_refs_by_sequence(hem_rows))

    # classification set: explicit statements only, never a thresholded number.
    cls_rows = df[(df["hemo_qual"] != 0) & ~df["hemo_ambiguous"]].copy()
    cls_rows["label"] = (cls_rows["hemo_qual"] > 0).astype(int)
    # a sequence with contradictory statements across rows is dropped
    grp = cls_rows.groupby("sequence")["label"].agg(["mean", "size"])
    consistent = grp[(grp["mean"] == 0.0) | (grp["mean"] == 1.0)]
    hemolysis_cls = pd.DataFrame(
        {
            "sequence": consistent.index,
            "label": consistent["mean"].astype(int).values,
            "n_measurements": consistent["size"].values,
        }
    ).reset_index(drop=True)
    dropped_conflict = int(len(grp) - len(consistent))

    report = {
        "source_file": str(raw_path),
        "source": "DRAMP (general_amps), CC BY 4.0, http://dramp.cpu-bioinfor.org/",
        "length_window": [MIN_LEN, MAX_LEN],
        "parse_stats": stats.as_dict(),
        "activity": {
            "target": "y_activity = 6 - log10(MIC[uM])  (higher = more active)",
            "aggregation_across_organisms": "minimum MIC (most susceptible organism)",
            "duplicate_collapse": "median over repeated sequences",
            "n_unique_sequences": int(len(activity)),
            "y_mean": float(activity["y_activity"].mean()),
            "y_std": float(activity["y_activity"].std()),
            "y_min": float(activity["y_activity"].min()),
            "y_max": float(activity["y_activity"].max()),
        },
        "hemolysis_regression": {
            "target": "y_hemolysis = 6 - log10(D50[uM])  (higher = MORE hemolytic)",
            "dose_keys": "HD50 / HC50 / HL50 / LD50 / LC50 / EC50 / IC50 / MHC",
            "n_unique_sequences": int(len(hemolysis)),
            "y_mean": float(hemolysis["y_hemolysis"].mean()),
            "y_std": float(hemolysis["y_hemolysis"].std()),
            "y_min": float(hemolysis["y_hemolysis"].min()),
            "y_max": float(hemolysis["y_hemolysis"].max()),
            "caveat": (
                "pools assay types and species (human/rabbit/sheep/guinea pig); "
                "see literature_review.md limitation 4"
            ),
        },
        "hemolysis_classification": {
            "labels_from": "explicit experimenter statements only, never thresholded doses",
            "n_unique_sequences": int(len(hemolysis_cls)),
            "n_positive": int(hemolysis_cls["label"].sum()) if len(hemolysis_cls) else 0,
            "n_negative": int((hemolysis_cls["label"] == 0).sum()) if len(hemolysis_cls) else 0,
            "dropped_conflicting_sequences": dropped_conflict,
        },
        "overlap_activity_and_hemolysis": int(
            len(set(activity["sequence"]) & set(hemolysis["sequence"]))
        ),
        "known_limitations": [
            "DRAMP general_amps is all-positive: no AMP negatives, so activity is "
            "regression-only (no AMP/non-AMP classifier).",
            "Labels parsed from free text; unparseable rows dropped and counted above.",
            "Minimum-MIC aggregation biases toward susceptible test organisms.",
            "Unit conversion assumes linear unmodified peptides with MW from sequence.",
        ],
    }

    return CurationResult(activity, hemolysis, hemolysis_cls, report)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res = curate()
    res.activity.to_csv(PROCESSED_DIR / "activity.csv", index=False)
    res.hemolysis.to_csv(PROCESSED_DIR / "hemolysis.csv", index=False)
    res.hemolysis_cls.to_csv(PROCESSED_DIR / "hemolysis_cls.csv", index=False)
    (PROCESSED_DIR / "curation_report.json").write_text(
        json.dumps(res.report, indent=2)
    )

    r = res.report
    print("DRAMP curation complete")
    print(f"  activity (pMIC regression)   : {r['activity']['n_unique_sequences']:5d} sequences")
    print(f"  hemolysis (p-dose regression): {r['hemolysis_regression']['n_unique_sequences']:5d} sequences")
    print(f"  hemolysis (classification)   : {r['hemolysis_classification']['n_unique_sequences']:5d} sequences "
          f"({r['hemolysis_classification']['n_positive']}+ / "
          f"{r['hemolysis_classification']['n_negative']}-)")
    print(f"  overlap both targets         : {r['overlap_activity_and_hemolysis']:5d} sequences")
    print(f"  report -> {PROCESSED_DIR / 'curation_report.json'}")


if __name__ == "__main__":
    main()
