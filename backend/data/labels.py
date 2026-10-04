"""Extraction of quantitative activity and hemolysis labels from DRAMP free text.

DRAMP stores concentrations inside prose, with heterogeneous units and qualifiers:

    "Gram-positive bacteria: S. simulans 22 (MIC=0.304 ug/ml), ..."
    "[Ref.22467870] HD50 = 11.6 uM against human red blood cells (Type A)."

Everything here is deliberately conservative: a value is extracted only when the unit is
recognised and the magnitude is physically plausible. Unparseable rows are dropped and
counted, never guessed at. See research/literature_review.md section 2 for the measured
yields and the resulting modelling decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.utils.peptide import molecular_weight

# ---------------------------------------------------------------------------
# Unit handling
# ---------------------------------------------------------------------------
# Molar units -> factor to micromolar.
_MOLAR_TO_UM: dict[str, float] = {
    "m": 1e6, "mm": 1e3, "um": 1.0, "nm": 1e-3, "pm": 1e-6,
}
# Mass-per-volume units -> factor to ug/mL (== mg/L).
_MASS_TO_UG_PER_ML: dict[str, float] = {
    "g/l": 1e3, "mg/ml": 1e3, "mg/l": 1.0, "ug/ml": 1.0, "ug/l": 1e-3,
    "ng/ml": 1e-3, "ng/ul": 1.0, "ug/dl": 1e-2, "g/ml": 1e6, "mg/dl": 1e1,
}

# Plausibility window in uM. Anything outside is treated as a parse error: sub-picomolar
# or molar-scale MICs in this corpus are unit-parsing mistakes, not measurements.
_MIN_UM = 1e-6
_MAX_UM = 1e6

_UNIT_ALTERNATION = (
    r"m\s*M|µ\s*M|μ\s*M|u\s*M|n\s*M|p\s*M|"
    r"mg\s*/\s*m[lL]|mg\s*/\s*[lL]|mg\s*/\s*d[lL]|"
    r"µ\s*g\s*/\s*m[lL]|μ\s*g\s*/\s*m[lL]|u\s*g\s*/\s*m[lL]|"
    r"µ\s*g\s*/\s*[lL]|μ\s*g\s*/\s*[lL]|u\s*g\s*/\s*[lL]|"
    r"µ\s*g\s*/\s*d[lL]|μ\s*g\s*/\s*d[lL]|"
    r"n\s*g\s*/\s*m[lL]|g\s*/\s*[lL]|g\s*/\s*m[lL]"
)
_NUMBER = r"\d+(?:\.\d+)?(?:\s*[xX×]\s*10\s*[-−]?\s*\d+)?"

# A concentration: number, optional range tail, then a unit. Ranges like "0.025-6.4 uM"
# yield the LOW end, which for a MIC is the optimistic/most-potent reading -- consistent
# with the minimum-MIC aggregation documented in the literature review.
_CONC_RE = re.compile(
    rf"({_NUMBER})\s*(?:[-–—]|to)?\s*(?:{_NUMBER})?\s*({_UNIT_ALTERNATION})",
    re.IGNORECASE,
)

# Keys naming a 50%-effect hemolytic/lytic dose.
HEMO_DOSE_KEY = (
    r"(?:HD\s*[₅5]\s*[₀0]|HC\s*[₅5]\s*[₀0]|HL\s*[₅5]\s*[₀0]|"
    r"LD\s*[₅5]\s*[₀0]|LC\s*[₅5]\s*[₀0]|EC\s*[₅5]\s*[₀0]|IC\s*[₅5]\s*[₀0]|"
    r"MHC|50\s*%\s*h(?:a)?emoly[a-z]*)"
)
MIC_KEY = r"MIC"

# Sentinels meaning "this field holds no measurement".
_NODATA_RE = re.compile(
    r"no hemolysis information|no hemolytic information|no cytotoxicity information|"
    r"\bnot found\b|not (?:been )?(?:determined|reported|tested|evaluated|available)|"
    r"\bunknown\b|\bN/?A\b|^\s*[-–—]*\s*$",
    re.IGNORECASE,
)

# Explicit statement that the peptide is NOT hemolytic.
_NEG_RE = re.compile(
    r"(?:\bno\b|\bnon-?\b|\bnot\b|without|lack(?:s|ing)?|free of|undetectable|"
    r"negligible|minimal|little|\bweak\b|\blow\b|"
    r"did not (?:cause|induce|show|exhibit|produce)|"
    r"no (?:adverse|signs? of|detectable|significant|appreciable|obvious))"
    r"[^.;]{0,60}?"
    r"(?:h(?:a)?emoly|lysis of (?:red|human|rabbit|sheep)|"
    r"toxic(?:ity)? (?:against|to|towards) (?:human |rabbit |sheep )?(?:red|erythro)|"
    r"effect on (?:red|human) (?:blood )?(?:cell|erythro))",
    re.IGNORECASE,
)
_NEG2_RE = re.compile(
    r"h(?:a)?emoly[^.;]{0,50}?"
    r"(?:was not|were not|not detected|not observed|\babsent\b)",
    re.IGNORECASE,
)
# Explicit statement that the peptide IS hemolytic.
_POS_RE = re.compile(
    r"(?:induced|caused|showed|exhibited|displayed|strong|high(?:ly)?|potent|"
    r"significant|marked|considerable|substantial)[^.;]{0,50}?h(?:a)?emoly|"
    r"h(?:a)?emolytic (?:activity|effect)s? (?:of|was observed|at|against)|"
    r"is h(?:a)?emolytic",
    re.IGNORECASE,
)


def _normalise_unit(raw: str) -> str:
    """Collapse whitespace and unicode mu variants to a canonical lowercase key."""
    u = re.sub(r"\s+", "", raw).lower()
    return u.replace("µ", "u").replace("μ", "u")


def _parse_number(raw: str) -> float | None:
    """Parse a number that may carry a scientific-notation tail like '2 x 10-3'."""
    txt = re.sub(r"\s+", "", raw)
    m = re.match(r"^(\d+(?:\.\d+)?)[xX×]10([-−]?\d+)$", txt)
    if m:
        mantissa = float(m.group(1))
        exponent = int(m.group(2).replace("−", "-"))
        return mantissa * (10.0 ** exponent)
    try:
        return float(txt)
    except ValueError:
        return None


def to_micromolar(value: float, unit_raw: str, mw: float) -> float | None:
    """Convert a concentration to micromolar, or None if the unit is unrecognised.

    Mass-per-volume units require the molecular weight:

        c[uM] = c[ug/mL] / MW[g/mol] * 1000
    """
    u = _normalise_unit(unit_raw)
    if u in _MOLAR_TO_UM:
        return value * _MOLAR_TO_UM[u]
    if u in _MASS_TO_UG_PER_ML:
        ug_per_ml = value * _MASS_TO_UG_PER_ML[u]
        if mw <= 0:
            return None
        return ug_per_ml / mw * 1000.0
    return None


def extract_concentrations(
    text: str | None,
    mw: float,
    keyword: str | None = None,
    window: int = 70,
) -> list[float]:
    """All plausible concentrations in ``text``, in micromolar.

    If ``keyword`` is given (a regex such as ``MIC``), only the ``window`` characters
    following each keyword occurrence are searched. This matters: ``Target_Organism``
    text contains strain numbers and ATCC codes that would otherwise be read as
    concentrations.
    """
    if not isinstance(text, str) or not text:
        return []

    segments: list[str]
    if keyword:
        segments = [
            text[m.start(): m.start() + window]
            for m in re.finditer(keyword, text, re.IGNORECASE)
        ]
    else:
        segments = [text]

    out: list[float] = []
    for seg in segments:
        for m in _CONC_RE.finditer(seg):
            val = _parse_number(m.group(1))
            if val is None:
                continue
            um = to_micromolar(val, m.group(2), mw)
            if um is not None and _MIN_UM < um < _MAX_UM:
                out.append(um)
    return out


def is_nodata(text: str | None) -> bool:
    """True when the field explicitly says no measurement exists."""
    if not isinstance(text, str) or not text.strip():
        return True
    return bool(_NODATA_RE.search(text))


@dataclass(frozen=True)
class HemolysisLabel:
    """Parsed hemolysis evidence for one row.

    ``dose_um`` is a 50%-effect concentration when one was stated. ``qualitative`` is
    -1 (explicitly non-hemolytic), +1 (explicitly hemolytic) or 0 (no clear statement).
    ``ambiguous`` marks text matching both negative and positive patterns, which is
    excluded from the classification set rather than arbitrarily resolved.
    """
    dose_um: float | None
    qualitative: int
    ambiguous: bool
    nodata: bool


def parse_hemolysis(text: str | None, mw: float) -> HemolysisLabel:
    """Extract hemolysis evidence from a DRAMP ``Hemolytic_activity`` cell."""
    if is_nodata(text):
        return HemolysisLabel(None, 0, False, True)
    assert isinstance(text, str)

    doses = extract_concentrations(text, mw, keyword=HEMO_DOSE_KEY)
    dose = min(doses) if doses else None

    neg = bool(_NEG_RE.search(text) or _NEG2_RE.search(text))
    pos = bool(_POS_RE.search(text))
    ambiguous = neg and pos
    qualitative = 0 if ambiguous else (-1 if neg else (1 if pos else 0))

    return HemolysisLabel(dose, qualitative, ambiguous, False)


def parse_mic(text: str | None, mw: float) -> float | None:
    """Minimum MIC in micromolar across all organisms mentioned in the cell.

    The minimum-MIC convention (most-susceptible organism) is documented as a limitation
    in research/literature_review.md section 2.3.
    """
    vals = extract_concentrations(text, mw, keyword=MIC_KEY)
    return min(vals) if vals else None


def p_scale(concentration_um: float) -> float:
    """Convert a micromolar concentration to the p-scale: ``6 - log10(c[uM])``.

    Equivalently ``-log10(c[M])``, the conventional pMIC potency scale. Higher means a
    lower concentration suffices, i.e. more potent.
    """
    import math
    if concentration_um <= 0:
        raise ValueError("concentration must be positive")
    return 6.0 - math.log10(concentration_um)


def inverse_p_scale(p_value: float) -> float:
    """Micromolar concentration from a p-scale value."""
    return 10.0 ** (6.0 - p_value)


@dataclass
class ParseStats:
    """Counters for the curation report, so dropped rows are always accounted for."""
    rows_total: int = 0
    invalid_sequence: int = 0
    length_filtered: int = 0
    mic_parsed: int = 0
    hemo_nodata: int = 0
    hemo_dose_parsed: int = 0
    hemo_qualitative: int = 0
    hemo_ambiguous: int = 0
    hemo_unparseable: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_total": self.rows_total,
            "invalid_sequence": self.invalid_sequence,
            "length_filtered": self.length_filtered,
            "mic_parsed": self.mic_parsed,
            "hemo_nodata": self.hemo_nodata,
            "hemo_dose_parsed": self.hemo_dose_parsed,
            "hemo_qualitative": self.hemo_qualitative,
            "hemo_ambiguous": self.hemo_ambiguous,
            "hemo_unparseable": self.hemo_unparseable,
            "notes": list(self.notes),
        }


def molecular_weight_safe(seq: str) -> float:
    """MW wrapper so callers can map over possibly-empty sequences."""
    return molecular_weight(seq) if seq else 0.0
