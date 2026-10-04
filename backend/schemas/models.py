"""Pydantic request/response schemas for the Q-Peptide API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from backend.utils.peptide import AA_SET

MAX_PARENT_LENGTH = 60
MIN_PARENT_LENGTH = 5


def _validate_sequence(v: str) -> str:
    s = "".join(v.split()).upper()
    if not s:
        raise ValueError("sequence is empty")
    bad = sorted({c for c in s if c not in AA_SET})
    if bad:
        raise ValueError(
            f"sequence contains non-canonical residues {bad}; "
            "only the 20 standard amino acids are supported"
        )
    if not MIN_PARENT_LENGTH <= len(s) <= MAX_PARENT_LENGTH:
        raise ValueError(
            f"sequence length {len(s)} outside the supported range "
            f"[{MIN_PARENT_LENGTH}, {MAX_PARENT_LENGTH}]"
        )
    return s


class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="Parent peptide, one-letter amino-acid codes")

    @field_validator("sequence")
    @classmethod
    def check(cls, v: str) -> str:
        return _validate_sequence(v)


class ObjectiveWeights(BaseModel):
    alpha: float = Field(1.0, ge=0.0, description="Weight on normalised activity")
    beta: float = Field(1.0, ge=0.0, description="Weight on normalised hemolysis penalty")


class MutationGenerateRequest(BaseModel):
    sequence: str
    target_n: int = Field(14, ge=2, le=24,
                          description="Target number of binary variables")
    max_per_position: int = Field(2, ge=1, le=6)
    alpha: float = Field(1.0, ge=0.0)
    beta: float = Field(1.0, ge=0.0)
    screen: Literal["abs_delta", "improving"] = "abs_delta"

    @field_validator("sequence")
    @classmethod
    def check(cls, v: str) -> str:
        return _validate_sequence(v)


class OptimizeRequest(BaseModel):
    """A full optimization run.

    ``budget_mode`` is REQUIRED to be explicit: "at_most_k" and "exactly_k" are
    different constraints and the API never guesses which was meant.
    """
    sequence: str
    budget_k: int = Field(3, ge=1, le=10, description="Mutation budget K")
    budget_mode: Literal["at_most_k", "exactly_k"] = Field(
        "at_most_k",
        description=(
            "'at_most_k' means sum_i x_i <= K (slack encoding); "
            "'exactly_k' means sum_i x_i == K (equality penalty)"
        ),
    )
    alpha: float = Field(1.0, ge=0.0)
    beta: float = Field(1.0, ge=0.0)
    target_n: int = Field(14, ge=2, le=22)
    max_per_position: int = Field(2, ge=1, le=6)
    penalty_factor: float = Field(2.0, gt=1.0)
    penalty_mode: Literal["objective_span", "global_bound"] = "objective_span"
    qaoa_depths: list[int] = Field(default_factory=lambda: [1, 2, 3])
    qaoa_optimizers: list[Literal["COBYLA", "Nelder-Mead", "Powell"]] = Field(
        default_factory=lambda: ["COBYLA"]
    )
    qaoa_maxiter: int = Field(300, ge=10, le=5000)
    qaoa_restarts: int = Field(3, ge=1, le=10)
    shots: int = Field(8192, ge=128, le=200_000)
    seed: int = 20261004
    include_noise: bool = True
    include_hardware: bool = False
    include_surrogate_validation: bool = True

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, v: str) -> str:
        return _validate_sequence(v)

    @field_validator("qaoa_depths")
    @classmethod
    def check_depths(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("at least one QAOA depth is required")
        for p in v:
            if not 1 <= p <= 6:
                raise ValueError(f"QAOA depth {p} outside supported range [1, 6]")
        return sorted(set(v))

    @field_validator("qaoa_optimizers")
    @classmethod
    def check_optimizers(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one optimizer is required")
        return list(dict.fromkeys(v))


class QuboBuildRequest(BaseModel):
    sequence: str
    budget_k: int = Field(3, ge=1, le=10)
    budget_mode: Literal["at_most_k", "exactly_k"] = "at_most_k"
    alpha: float = Field(1.0, ge=0.0)
    beta: float = Field(1.0, ge=0.0)
    target_n: int = Field(14, ge=2, le=22)
    max_per_position: int = Field(2, ge=1, le=6)
    penalty_factor: float = Field(2.0, gt=1.0)
    penalty_mode: Literal["objective_span", "global_bound"] = "objective_span"
    include_matrix: bool = True

    @field_validator("sequence")
    @classmethod
    def check(cls, v: str) -> str:
        return _validate_sequence(v)


class QaoaRunRequest(QuboBuildRequest):
    p: int = Field(2, ge=1, le=6)
    optimizer: Literal["COBYLA", "Nelder-Mead", "Powell"] = "COBYLA"
    maxiter: int = Field(300, ge=10, le=5000)
    restarts: int = Field(3, ge=1, le=10)
    shots: int = Field(8192, ge=128, le=200_000)
    seed: int = 20261004
    noise: bool = False


class BenchmarkRequest(QuboBuildRequest):
    seed: int = 20261004


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    activity_model: dict[str, Any] | None = None
    hemolysis_model: dict[str, Any] | None = None
    quantum_backend_available: bool
    ibm_credentials_present: bool
    versions: dict[str, str]
    disclaimer: str


class ApiError(BaseModel):
    detail: str
    hint: str | None = None
