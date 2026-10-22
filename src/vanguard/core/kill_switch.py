"""Emergency Kill Switch for V.A.N.G.U.A.R.D.

Provides immediate system-wide cessation of voice dispatch, actuators, and background tasks.
"""

from __future__ import annotations

import logging

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event, EventCategory

logger = logging.getLogger(__name__)


class KillSwitch:
    """System-level safety interlock."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.is_engaged: bool = False
        self.engagement_reason: str | None = None
        self.engaged_by: str | None = None

    async def engage(self, triggered_by: str, reason: str = "Emergency kill switch engaged") -> None:
        """Engage the kill switch immediately."""
        self.is_engaged = True
        self.engagement_reason = reason
        self.engaged_by = triggered_by

        logger.critical(f"EMERGENCY KILL SWITCH ENGAGED by {triggered_by}: {reason}")

        await self.event_bus.publish(
            Event(
                category=EventCategory.ALERT,
                event_type="kill_switch_engaged",
                source="kill_switch",
                payload={"triggered_by": triggered_by, "reason": reason},
            )
        )

    async def disengage(self, reset_by: str) -> bool:
        """Disengage kill switch, only allowed by family heads."""
        if reset_by.lower() not in ("eivind", "wife"):
            logger.warning(f"Unauthorized reset attempt on kill switch by {reset_by}")
            return False

        self.is_engaged = False
        self.engagement_reason = None
        self.engaged_by = None

        logger.info(f"Emergency kill switch disengaged by {reset_by}")

        await self.event_bus.publish(
            Event(
                category=EventCategory.ALERT,
                event_type="kill_switch_disengaged",
                source="kill_switch",
                payload={"reset_by": reset_by},
            )
        )
        return True
