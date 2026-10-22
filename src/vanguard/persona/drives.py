"""Drive profiles and motivational baselines for P.E.R.S.O.N.A."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriveProfile:
    """Psychological and functional drives governing a persona's autonomous focus."""

    persona_id: str
    weights: dict[str, float] = field(default_factory=dict)

    def get_drive(self, name: str, default: float = 0.5) -> float:
        """Get the weight of a specific drive."""
        return self.weights.get(name, default)

    def update_drive(self, name: str, value: float) -> None:
        """Update a drive's intensity (bounded 0.0 - 1.0)."""
        self.weights[name] = max(0.0, min(1.0, value))
