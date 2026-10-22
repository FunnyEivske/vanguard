"""Tests verifying floor management, human barge-in, room exclusivity, and sandboxing permissions."""

import pytest

from vanguard.core.events import SpeechRequest
from vanguard.core.floor_manager import FloorManager
from vanguard.core.permission_router import PermissionRouter, ToolCommand


@pytest.mark.asyncio
async def test_human_barge_in_priority(floor_manager: FloorManager) -> None:
    """Ensure a speaking human immediately preempts any active AI speaker in the room."""
    # Friday takes the floor
    req_friday = SpeechRequest(speaker_id="friday", room_id="living_room", text="Telemetry report...")
    granted_friday = await floor_manager.request_floor(req_friday)
    assert granted_friday
    assert floor_manager.get_active_speaker("living_room") == "friday"

    # Human barges in
    req_human = SpeechRequest(speaker_id="eivind", room_id="living_room", text="Stop, Friday.", priority=10)
    granted_human = await floor_manager.request_floor(req_human)
    assert granted_human
    # Friday has been evicted, Eivind holds the floor
    assert floor_manager.get_active_speaker("living_room") == "eivind"

    # While human speaks, another AI cannot interrupt
    req_jarvis = SpeechRequest(speaker_id="jarvis", room_id="living_room", text="Pardon me...")
    granted_jarvis = await floor_manager.request_floor(req_jarvis)
    assert not granted_jarvis
    assert floor_manager.get_active_speaker("living_room") == "eivind"


@pytest.mark.asyncio
async def test_room_presence_exclusivity(floor_manager: FloorManager) -> None:
    """Ensure an AI persona cannot be active in two rooms simultaneously."""
    # Jarvis speaks in office
    req_office = SpeechRequest(speaker_id="jarvis", room_id="office", text="Office systems ready.")
    await floor_manager.request_floor(req_office)
    assert floor_manager.get_persona_room("jarvis") == "office"
    assert floor_manager.get_active_speaker("office") == "jarvis"

    # Jarvis moves/speaks in living room -> office is cleanly vacated
    req_living = SpeechRequest(speaker_id="jarvis", room_id="living_room", text="Now in living room.")
    await floor_manager.request_floor(req_living)

    assert floor_manager.get_persona_room("jarvis") == "living_room"
    assert floor_manager.get_active_speaker("living_room") == "jarvis"
    assert floor_manager.get_active_speaker("office") is None


def test_lead_mode_override(floor_manager: FloorManager) -> None:
    """Ensure temporary crisis Lead Mode can be cancelled at any time by family heads."""
    floor_manager.enable_lead_mode(holder="jarvis")
    assert floor_manager.lead_mode_active

    # Unauthorized child cannot override lead mode
    assert not floor_manager.override_lead_mode(overridden_by="child_elder")
    assert floor_manager.lead_mode_active

    # Family head overrides lead mode
    assert floor_manager.override_lead_mode(overridden_by="eivind")
    assert not floor_manager.lead_mode_active


def test_edith_sandboxing_and_door_authority(permission_router: PermissionRouter) -> None:
    """Ensure E.D.I.T.H. is barred from door actuation and high-risk tools are forced into sandbox."""
    # Rule 1: Zero door actuation authority
    cmd_door = ToolCommand(caller_id="edith", tool_name="actuate_front_door", is_door_actuation=True)
    res_door = permission_router.evaluate_command(cmd_door)
    assert not res_door.allowed
    assert "zero door actuation authority" in res_door.reason

    # Rule 2: High-risk tools run in sandbox simulation only
    cmd_high_risk = ToolCommand(caller_id="edith", tool_name="drone_swarm_sim", is_high_risk=True)
    res_high_risk = permission_router.evaluate_command(cmd_high_risk)
    assert res_high_risk.allowed
    assert res_high_risk.sandboxed
    assert "sandbox simulation" in res_high_risk.reason

    # Rule 3: Destructive commands require human consent
    cmd_destr = ToolCommand(caller_id="edith", tool_name="wipe_scratch_cache", is_destructive=True)
    res_destr_no_consent = permission_router.evaluate_command(cmd_destr, human_consent_granted=False)
    assert not res_destr_no_consent.allowed

    res_destr_with_consent = permission_router.evaluate_command(cmd_destr, human_consent_granted=True)
    assert res_destr_with_consent.allowed


def test_daemon_zero_door_actuation(permission_router: PermissionRouter) -> None:
    """Ensure D.U.M.-E and U have zero door actuation authority."""
    for daemon_id in ("dume", "u"):
        cmd = ToolCommand(caller_id=daemon_id, tool_name="unlock_workshop_door", is_door_actuation=True)
        res = permission_router.evaluate_command(cmd)
        assert not res.allowed
        assert "zero door actuation authority" in res.reason


def test_child_safety_restrictions(permission_router: PermissionRouter) -> None:
    """Ensure children cannot trigger high-risk or destructive actions without guardian consent."""
    cmd = ToolCommand(caller_id="living_room_device", tool_name="purchase_item", is_high_risk=True)
    res_unauth = permission_router.evaluate_command(cmd, initiating_human_id="child_younger", human_consent_granted=False)
    assert not res_unauth.allowed
    assert "guardian consent" in res_unauth.reason
