"""Shared configuration for FedDef experiments.

Keeping every experiment parameter in one immutable object makes runs
reproducible and prevents the ad-hoc values that were hard-coded in the
original prototype.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

AttackType = Literal["random", "sign_flip", "scaling", "label_flip", "collusion"]

ALLOWED_ATTACK_TYPES: tuple[str, ...] = (
    "random",
    "sign_flip",
    "scaling",
    "label_flip",
    "collusion",
)

ALLOWED_SETTINGS: tuple[str, ...] = ("clean", "attack", "defense")


@dataclass(frozen=True)
class Config:
    """Configuration shared by the baseline, attack, and defense runs.

    The defaults match the project proposal (20 clients, 20% malicious,
    50 rounds, learning rate 0.2).  Values are validated on creation so a
    typo fails immediately instead of producing a meaningless experiment.
    """

    seed: int = 42

    # Data split.  test_size + val_size must stay below 1.0.
    test_size: float = 0.2
    val_size: float = 0.1
    target_col: str = "TenYearCHD"

    # Federated setup.
    n_clients: int = 20
    mal_fraction: float = 0.2
    epochs: int = 50
    lr: float = 0.2

    # Defense.
    theta: float = 0.25
    z0: float = 1.0

    # Attack.
    attack_type: AttackType = "random"
    attack_strength: float = 0.5

    # Paths.
    data_path: Path = field(default_factory=lambda: Path("data/framingham.csv"))
    results_dir: Path = field(default_factory=lambda: Path("results"))

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if not 0.0 < self.val_size < 1.0:
            raise ValueError("val_size must be in (0, 1)")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be less than 1.0")

        if not isinstance(self.n_clients, int) or self.n_clients < 2:
            raise ValueError("n_clients must be an integer >= 2")
        if not 0.0 <= self.mal_fraction < 1.0:
            raise ValueError("mal_fraction must be in [0, 1)")
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be an integer >= 1")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")

        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        if self.z0 < 0.0:
            raise ValueError("z0 must be non-negative")

        if self.attack_type not in ALLOWED_ATTACK_TYPES:
            raise ValueError(
                f"attack_type must be one of {ALLOWED_ATTACK_TYPES}, "
                f"got {self.attack_type!r}"
            )
        if self.attack_strength < 0.0:
            raise ValueError("attack_strength must be non-negative")

        if not isinstance(self.target_col, str) or not self.target_col.strip():
            raise ValueError("target_col must be a non-empty string")

        # Normalise paths so callers can pass strings as well as Paths.
        if not isinstance(self.data_path, Path):
            object.__setattr__(self, "data_path", Path(self.data_path))
        if not isinstance(self.results_dir, Path):
            object.__setattr__(self, "results_dir", Path(self.results_dir))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "seed": self.seed,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "target_col": self.target_col,
            "n_clients": self.n_clients,
            "mal_fraction": self.mal_fraction,
            "epochs": self.epochs,
            "lr": self.lr,
            "theta": self.theta,
            "z0": self.z0,
            "attack_type": self.attack_type,
            "attack_strength": self.attack_strength,
            "data_path": str(self.data_path),
            "results_dir": str(self.results_dir),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Config":
        """Build a Config from a dictionary, ignoring unknown keys."""
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def ensure_results_dir(self) -> Path:
        """Create and return the results directory."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir

    def save(self, path: str | Path) -> Path:
        """Write the configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Read a configuration from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError(f"Config file must contain a JSON object: {path}")
        return cls.from_dict(values)


DEFAULT_CONFIG = Config()
