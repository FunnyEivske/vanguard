"""Permission Router for V.A.N.G.U.A.R.D.

Enforces role-based permissions, EDITH sandboxing, hardware actuation boundaries,
and child safety constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCommand:
    """Requested tool or hardware action."""

    caller_id: str  # Persona or Human ID
    tool_name: str
    target_device: str | None = None
    is_high_risk: bool = False
    is_destructive: bool = False
    is_door_actuation: bool = False
    requires_human_consent: bool = False
    params: dict[str, Any] | None = None


@dataclass
class PermissionResult:
    """Result of permission verification."""

    allowed: bool
    sandboxed: bool
    reason: str


class PermissionRouter:
    """Enforces fine-grained permission rules across personas and family roles."""

    def __init__(self, personas_config: dict[str, Any], people_config: dict[str, Any]) -> None:
        self.personas_config = personas_config
        self.people_config = people_config

    def evaluate_command(
        self,
        command: ToolCommand,
        initiating_human_id: str | None = None,
        human_consent_granted: bool = False,
    ) -> PermissionResult:
        """Evaluate if a tool command is authorized to execute in production or sandbox."""
        caller = command.caller_id.lower()

        # Human direct commands
        if initiating_human_id and caller in ("eivind", "wife"):
            return PermissionResult(allowed=True, sandboxed=False, reason="Authorized by family head.")

        # Children restrictions
        if initiating_human_id in ("child_elder", "child_younger"):
            if command.is_destructive or command.is_door_actuation or command.is_high_risk:
                if not human_consent_granted:
                    return PermissionResult(
                        allowed=False,
                        sandboxed=False,
                        reason="Child request requires guardian consent for destructive or high-risk actions.",
                    )

        # Visitor restrictions
        if initiating_human_id == "visitor" or caller == "visitor":
            if command.is_door_actuation or command.is_high_risk or command.is_destructive:
                return PermissionResult(
                    allowed=False,
                    sandboxed=False,
                    reason="Visitors do not hold authorization for physical actuation or high-risk tools.",
                )

        # EDITH Persona Rules:
        # 1. Zero door actuation authority
        # 2. High-risk tools run in sandbox simulation only
        # 3. Destructive commands require human consent
        if caller == "edith":
            if command.is_door_actuation:
                return PermissionResult(
                    allowed=False,
                    sandboxed=False,
                    reason="E.D.I.T.H. has zero door actuation authority.",
                )
            if command.is_destructive and not human_consent_granted:
                return PermissionResult(
                    allowed=False,
                    sandboxed=False,
                    reason="E.D.I.T.H. destructive command requires explicit human consent.",
                )
            if command.is_high_risk:
                return PermissionResult(
                    allowed=True,
                    sandboxed=True,
                    reason="E.D.I.T.H. high-risk tools execute strictly within sandbox simulation.",
                )

        # DUM-E & U Persona Rules: Zero door actuation authority
        if caller in ("dume", "u"):
            if command.is_door_actuation:
                return PermissionResult(
                    allowed=False,
                    sandboxed=False,
                    reason=f"{caller.upper()} is a daemon with zero door actuation authority.",
                )

        # AI-to-AI direction rule: No AI directs another without human consent
        if caller in self.personas_config and command.caller_id != initiating_human_id:
            if command.is_destructive and not human_consent_granted:
                return PermissionResult(
                    allowed=False,
                    sandboxed=False,
                    reason="AI cannot initiate destructive actions without human leadership.",
                )

        return PermissionResult(allowed=True, sandboxed=False, reason="Command authorized.")
