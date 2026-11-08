"""Multi-Room Smart House Controller for V.A.N.G.U.A.R.D.

Manages distributed hidden microphones, hidden speakers, room cameras, smart doors/locks,
lighting, climate (HVAC), and environmental sensors across every room of the house.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from vanguard.core.event_bus import EventBus
from vanguard.core.events import Event, EventCategory
from vanguard.core.permission_router import PermissionResult, PermissionRouter, ToolCommand


@dataclass
class RoomDoorState:
    """State of a physical door or lock in a room."""

    door_id: str
    room_id: str
    is_open: bool = False
    is_locked: bool = True
    last_actuated_by: str | None = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class RoomNode:
    """Hardware and environmental state of a single room in the house."""

    room_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    mic_endpoint: str = "default"
    speaker_endpoint: str = "default"
    camera_endpoint: str = "/dev/video0"
    temperature_c: float = 21.0
    lights_on: bool = True
    brightness_pct: int = 80
    doors: dict[str, RoomDoorState] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)


class HouseController:
    """Provides house-wide camera visibility, room roaming, and permission-checked actuation."""

    DEFAULT_ROOMS: list[dict[str, Any]] = [
        {
            "id": "living_room",
            "name": "Stue",
            "aliases": ["living room", "living_room", "stue", "stua", "lounge"],
            "doors": ["living_room_patio_door"],
        },
        {
            "id": "kitchen",
            "name": "Kjøkken",
            "aliases": ["kitchen", "kjøkken", "kjøkkenet", "kjokken"],
            "doors": ["kitchen_back_door"],
        },
        {
            "id": "office",
            "name": "Kontor",
            "aliases": ["office", "kontor", "kontoret", "study"],
            "doors": ["office_door"],
        },
        {
            "id": "workshop",
            "name": "Verksted",
            "aliases": ["workshop", "verksted", "verkstedet", "lab", "garage"],
            "doors": ["workshop_heavy_door"],
        },
        {
            "id": "entrance",
            "name": "Inngang",
            "aliases": ["entrance", "inngang", "gang", "hallway", "front door", "ytterdør"],
            "doors": ["front_entrance_door"],
        },
        {
            "id": "children_room",
            "name": "Barnerom",
            "aliases": ["children_room", "children room", "kids room", "barnerom", "barnerommet"],
            "doors": ["children_room_door"],
        },
    ]

    # Patterns for "go to <room>", "send to <room>", "in the <room>"
    ROAM_DIRECTIVE_PATTERNS = [
        re.compile(
            r"\b(?:go\s+to|move\s+to|head\s+to|send\s+to|speak\s+in|announce\s+in|tell\s+.*?\s+in|gå\s+til|flytt\s+til|snakk\s+i)\s+(?:the\s+)?([a-zA-ZæøåÆØÅ_]+(?:\s+room)?)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        event_bus: EventBus,
        permission_router: PermissionRouter,
        rooms_config: list[dict[str, Any]] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.permission_router = permission_router
        self.rooms: dict[str, RoomNode] = {}
        self._init_rooms(rooms_config or self.DEFAULT_ROOMS)

    def _init_rooms(self, rooms_config: list[dict[str, Any]]) -> None:
        merged_by_id: dict[str, dict[str, Any]] = {r["id"]: dict(r) for r in self.DEFAULT_ROOMS}
        for rc in rooms_config:
            rid = rc.get("id")
            if rid:
                if rid in merged_by_id:
                    merged_by_id[rid].update(rc)
                else:
                    merged_by_id[rid] = dict(rc)

        for idx, (rid, rdata) in enumerate(merged_by_id.items()):
            aliases = list(rdata.get("aliases", [rid, rdata.get("name", rid).lower()]))
            if rid not in aliases:
                aliases.append(rid)
            if rdata.get("name", "").lower() not in aliases:
                aliases.append(rdata.get("name", "").lower())

            door_names = rdata.get("doors", [f"{rid}_door"])
            doors_map = {
                d_id: RoomDoorState(door_id=d_id, room_id=rid, is_open=False, is_locked=True)
                for d_id in door_names
            }
            self.rooms[rid] = RoomNode(
                room_id=rid,
                name=rdata.get("name", rid.capitalize()),
                aliases=aliases,
                mic_endpoint=rdata.get("mic_endpoint", f"hw:room_{rid}_mic"),
                speaker_endpoint=rdata.get("speaker_endpoint", f"hw:room_{rid}_spk"),
                camera_endpoint=rdata.get("camera_endpoint", f"/dev/video{idx}"),
                doors=doors_map,
                sensors={"motion": False, "water_leak": False, "smoke": False, "air_quality": "nominal"},
            )

    def resolve_room_id(self, room_query: str) -> str | None:
        """Resolve a room name or Norwegian/English alias (e.g. 'kjøkkenet', 'stua', 'kitchen') to its canonical room_id."""
        norm = room_query.lower().strip()
        if norm in self.rooms:
            return norm
        for rid, node in self.rooms.items():
            for alias in node.aliases:
                if alias in norm or norm in alias:
                    return rid
        return None

    def detect_roam_target_room(self, text: str) -> str | None:
        """Detect if an utterance directs a persona to roam/move/speak in another room."""
        for pat in self.ROAM_DIRECTIVE_PATTERNS:
            match = pat.search(text)
            if match:
                candidate = match.group(1).strip()
                resolved = self.resolve_room_id(candidate)
                if resolved:
                    return resolved
        return None

    def detect_house_actuation_intent(
        self,
        text: str,
        persona_id: str,
        default_room_id: str,
    ) -> tuple[ToolCommand | None, str, dict[str, Any]]:
        """Detect if the human is asking the persona to actuate doors, locks, lights, or climate."""
        lower = text.lower()
        target_room = default_room_id
        for rid, node in self.rooms.items():
            for alias in node.aliases:
                if re.search(rf"\b{re.escape(alias)}\b", lower):
                    target_room = rid
                    break

        # 1. Door / Lock actuation
        if any(w in lower for w in ("unlock", "lock", "open door", "close door", "lås opp", "lås døren", "åpne døren", "lukk døren")):
            action = "unlock" if ("unlock" in lower or "lås opp" in lower or "open" in lower or "åpne" in lower) else "lock"
            room_node = self.rooms.get(target_room)
            door_id = next(iter(room_node.doors.keys()), f"{target_room}_door") if room_node else f"{target_room}_door"
            cmd = ToolCommand(
                caller_id=persona_id,
                tool_name=f"{action}_door",
                target_device=door_id,
                is_door_actuation=True,
                params={"room_id": target_room, "door_id": door_id, "action": action},
            )
            return cmd, target_room, {"type": "door", "action": action, "door_id": door_id, "room_id": target_room}

        # 2. Lights actuation
        if any(w in lower for w in ("lights on", "lights off", "turn on light", "turn off light", "lys på", "lys av", "slå på lyset", "slå av lyset")):
            turn_on = any(w in lower for w in ("lights on", "turn on", "lys på", "slå på"))
            cmd = ToolCommand(
                caller_id=persona_id,
                tool_name="set_room_lights",
                target_device=f"{target_room}_lights",
                params={"room_id": target_room, "lights_on": turn_on},
            )
            return cmd, target_room, {"type": "lights", "lights_on": turn_on, "room_id": target_room}

        # 3. Temperature / HVAC actuation
        temp_match = re.search(r"(\d{2}(?:\.\d)?)\s*(?:degrees|grader|°c|c\b)", lower)
        if temp_match and any(w in lower for w in ("temp", "heat", "varme", "set", "sett", "degrees", "grader")):
            temp_val = float(temp_match.group(1))
            cmd = ToolCommand(
                caller_id=persona_id,
                tool_name="set_room_temperature",
                target_device=f"{target_room}_hvac",
                params={"room_id": target_room, "temperature_c": temp_val},
            )
            return cmd, target_room, {"type": "climate", "temperature_c": temp_val, "room_id": target_room}

        return None, target_room, {}

    async def execute_actuation(
        self,
        command: ToolCommand,
        initiating_human_id: str,
        human_consent_granted: bool = False,
    ) -> tuple[PermissionResult, dict[str, Any]]:
        """Evaluate permissions through PermissionRouter and execute the physical house actuation if allowed."""
        perm = self.permission_router.evaluate_command(
            command=command,
            initiating_human_id=initiating_human_id,
            human_consent_granted=human_consent_granted,
        )
        params = command.params or {}
        room_id = params.get("room_id", "living_room")
        room = self.rooms.get(room_id)

        if not perm.allowed:
            await self.event_bus.publish(
                Event(
                    category=EventCategory.ALERT,
                    event_type="house_actuation_denied",
                    source="house_controller",
                    room_id=room_id,
                    payload={
                        "caller_id": command.caller_id,
                        "initiating_human_id": initiating_human_id,
                        "tool_name": command.tool_name,
                        "reason": perm.reason,
                    },
                )
            )
            return perm, {"executed": False, "reason": perm.reason}

        state_change: dict[str, Any] = {"executed": True, "sandboxed": perm.sandboxed, "room_id": room_id}
        if room and not perm.sandboxed:
            if command.is_door_actuation:
                door_id = params.get("door_id") or next(iter(room.doors.keys()))
                action = params.get("action", "lock")
                door = room.doors.setdefault(door_id, RoomDoorState(door_id=door_id, room_id=room_id))
                if action == "unlock":
                    door.is_locked = False
                    door.is_open = True
                else:
                    door.is_open = False
                    door.is_locked = True
                door.last_actuated_by = f"{command.caller_id} (for {initiating_human_id})"
                door.last_updated = time.time()
                state_change["door"] = {
                    "door_id": door_id,
                    "is_locked": door.is_locked,
                    "is_open": door.is_open,
                }
            elif command.tool_name == "set_room_lights":
                room.lights_on = bool(params.get("lights_on", True))
                state_change["lights_on"] = room.lights_on
            elif command.tool_name == "set_room_temperature":
                room.temperature_c = float(params.get("temperature_c", 21.0))
                state_change["temperature_c"] = room.temperature_c

        await self.event_bus.publish(
            Event(
                category=EventCategory.COMMAND,
                event_type="house_actuation_executed",
                source=command.caller_id,
                room_id=room_id,
                payload=state_change,
            )
        )
        return perm, state_change

    def get_house_snapshot(self, room_occupants_map: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
        """Return a complete real-time snapshot of all rooms, cameras, occupants, doors, lights, and sensors."""
        snapshot: dict[str, Any] = {}
        for rid, node in self.rooms.items():
            occ = (room_occupants_map or {}).get(rid, [])
            snapshot[rid] = {
                "room_id": rid,
                "name": node.name,
                "mic_endpoint": node.mic_endpoint,
                "speaker_endpoint": node.speaker_endpoint,
                "camera_endpoint": node.camera_endpoint,
                "occupants": occ,
                "lights_on": node.lights_on,
                "temperature_c": node.temperature_c,
                "doors": {
                    did: {"is_locked": d.is_locked, "is_open": d.is_open}
                    for did, d in node.doors.items()
                },
                "sensors": dict(node.sensors),
            }
        return snapshot

