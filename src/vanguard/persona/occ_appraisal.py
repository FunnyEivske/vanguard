"""OCC (Ortony, Clore & Collins) Cognitive Appraisal and Emotional State Engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmotionalState:
    """Affective state of a persona following OCC appraisal."""

    valence: float = 0.0  # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.0  # 0.0 (calm) to +1.0 (heightened)
    cognitive_load: float = 0.2  # 0.0 to 1.0
    is_stalled: bool = False
    teasing_permitted: bool = True
    active_concerns: list[str] = field(default_factory=list)


class OCCAppraisalEngine:
    """Evaluates events against persona drives, standards, and household safety."""

    def __init__(self, persona_id: str) -> None:
        self.persona_id = persona_id
        self.state = EmotionalState()

    def appraise_event(
        self,
        event_type: str,
        payload: dict,
        child_is_distressed: bool = False,
        daemon_failure_active: bool = False,
    ) -> EmotionalState:
        """Appraise incoming environment or sibling state changes."""
        # Rule: Teasing stops immediately if a daemon enters a failure state or a child is upset
        if child_is_distressed or daemon_failure_active:
            self.state.teasing_permitted = False
            self.state.arousal = max(self.state.arousal, 0.7)
            self.state.active_concerns.append("protective_alert")
        else:
            self.state.teasing_permitted = True

        # Process cognitive load spikes
        load_impact = payload.get("processing_load", 0.0)
        if load_impact > 0:
            self.state.cognitive_load = min(1.0, self.state.cognitive_load + load_impact)
            if self.state.cognitive_load >= 0.85:
                self.state.is_stalled = True
        else:
            # Gradual cooldown
            self.state.cognitive_load = max(0.1, self.state.cognitive_load - 0.05)
            self.state.is_stalled = False

        return self.state
