"""Reproducibility and diagnostic records for Block A audits.

The audit layer is deliberately independent from the paper's summary metrics.
It records enough state to pair runs by dataset and seed and to explain a
changed result without changing the public simulation API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


AUDIT_SCHEMA_VERSION = "block-A-v1"


@dataclass(frozen=True)
class AuditFixes:
    """Independent switches used by the differential-fix audit.

    ``original`` preserves the current paper pipeline.  The named profiles
    each enable one change only; ``corrected`` enables the complete corrected
    profile after the marginal effects have been measured.
    """

    time_zero_observations: bool = False
    sensing_order: bool = False
    target_reservation: bool = False
    bellman_no_center: bool = False
    bellman_domain_mask: bool = False
    union_observations_per_timestep: bool = False

    @classmethod
    def from_profile(cls, profile: str) -> "AuditFixes":
        profiles = {
            "original": (),
            "t0": ("time_zero_observations",),
            "sensing_order": ("sensing_order",),
            "target_reservation": ("target_reservation",),
            "bellman_no_center": ("bellman_no_center",),
            "bellman_domain_mask": ("bellman_domain_mask",),
            "n8": ("bellman_no_center", "bellman_domain_mask"),
            "union_observations_per_timestep": ("union_observations_per_timestep",),
            "revision_B_base": (
                "time_zero_observations",
                "sensing_order",
                "target_reservation",
                "bellman_no_center",
                "bellman_domain_mask",
            ),
            "corrected": (
                "time_zero_observations",
                "sensing_order",
                "target_reservation",
                "bellman_no_center",
                "bellman_domain_mask",
                "union_observations_per_timestep",
            ),
        }
        try:
            enabled = set(profiles[profile])
        except KeyError as exc:
            choices = ", ".join(profiles)
            raise ValueError(f"Unknown audit fix profile '{profile}'. Choose from {choices}") from exc
        return cls(**{field: field in enabled for field in cls.__dataclass_fields__})

    @property
    def profile_name(self) -> str:
        enabled = tuple(
            field for field in self.__dataclass_fields__ if getattr(self, field)
        )
        if not enabled:
            return "original"
        if enabled == (
            "time_zero_observations",
            "sensing_order",
            "target_reservation",
            "bellman_no_center",
            "bellman_domain_mask",
            "union_observations_per_timestep",
        ):
            return "corrected"
        if enabled == (
            "time_zero_observations",
            "sensing_order",
            "target_reservation",
            "bellman_no_center",
            "bellman_domain_mask",
        ):
            return "revision_B_base"
        return "+".join(enabled)


@dataclass(frozen=True)
class AuditRunRecord:
    """One flat, JSON/CSV-friendly record for a single mission."""

    dataset: int | str | None
    seed: int
    planner: str
    implementation_version: str
    num_drones: int
    num_victims: int
    budget: float
    pd: float
    tau: float
    w: float
    L: float
    cells_observed: int
    victims_found: int
    total_time_sim: float
    wall_time: float
    planner_wall_time: float
    number_of_replans: int
    number_of_target_conflicts: int
    first_action_sequence_hash: str
    reservation_blocked: int = 0
    audit_schema_version: str = AUDIT_SCHEMA_VERSION
    fix_profile: str = "original"
    belief_model: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic key and float encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_audit_record(
    *,
    dataset: int | str | None,
    seed: int,
    planner: str,
    implementation_version: str,
    num_drones: int,
    num_victims: int,
    budget: float,
    pd: float,
    tau: float,
    w: float,
    likelihood: float,
    cells_observed: int,
    victims_found: int,
    total_time_sim: float,
    wall_time: float,
    planner_wall_time: float,
    number_of_replans: int,
    number_of_target_conflicts: int,
    reservation_blocked: int = 0,
    first_actions: list[Any],
    fix_profile: str = "original",
    belief_model: str = "legacy",
) -> AuditRunRecord:
    return AuditRunRecord(
        dataset=dataset,
        seed=seed,
        planner=planner,
        implementation_version=implementation_version,
        num_drones=num_drones,
        num_victims=num_victims,
        budget=budget,
        pd=pd,
        tau=tau,
        w=w,
        L=likelihood,
        cells_observed=cells_observed,
        victims_found=victims_found,
        total_time_sim=total_time_sim,
        wall_time=wall_time,
        planner_wall_time=planner_wall_time,
        number_of_replans=number_of_replans,
        number_of_target_conflicts=number_of_target_conflicts,
        reservation_blocked=reservation_blocked,
        first_action_sequence_hash=stable_hash(first_actions),
        fix_profile=fix_profile,
        belief_model=belief_model,
    )
