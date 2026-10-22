"""Pytest fixtures for V.A.N.G.U.A.R.D. test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vanguard.core.audit_logger import AuditLogger
from vanguard.core.event_bus import EventBus
from vanguard.core.floor_manager import FloorManager
from vanguard.core.kill_switch import KillSwitch
from vanguard.core.permission_router import PermissionRouter
from vanguard.memory.manager import ThreeTierMemoryManager
from vanguard.output.sound_cues import SoundCueDispatcher
from vanguard.output.tts_dispatch import TTSDispatcher
from vanguard.output.ui_cards import UICardEmitter
from vanguard.persona.engine import PersonaEngine


@pytest.fixture
def repo_root() -> Path:
    """Return path to repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir(repo_root: Path) -> Path:
    """Return path to config directory."""
    return repo_root / "config"


@pytest.fixture
def personas_dir(config_dir: Path) -> Path:
    """Return path to personas directory."""
    return config_dir / "personas"


@pytest.fixture
def system_settings(config_dir: Path) -> dict:
    """Load system settings configuration."""
    with open(config_dir / "system_settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def personas_config(personas_dir: Path) -> dict:
    """Load all persona YAML configs."""
    configs = {}
    for p_file in personas_dir.glob("*.yaml"):
        with open(p_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and data.get("enabled", True):
                configs[data["id"]] = data
    return configs


@pytest.fixture
def people_config(config_dir: Path) -> dict:
    """Load people profiles."""
    profiles = {}
    people_dir = config_dir / "people"
    for p_file in people_dir.glob("*.yaml"):
        with open(p_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                pid = data.get("id") or data.get("family_unit") or data.get("template_id")
                profiles[pid] = data
    return profiles


@pytest.fixture
def event_bus() -> EventBus:
    """Fresh EventBus instance."""
    return EventBus(inter_agent_speech_aloud=False)


@pytest.fixture
def audit_logger(event_bus: EventBus) -> AuditLogger:
    """AuditLogger subscribed to event bus."""
    return AuditLogger(event_bus=event_bus)


@pytest.fixture
def floor_manager(event_bus: EventBus) -> FloorManager:
    """FloorManager instance."""
    return FloorManager(event_bus=event_bus)


@pytest.fixture
def kill_switch(event_bus: EventBus) -> KillSwitch:
    """Emergency kill switch instance."""
    return KillSwitch(event_bus=event_bus)


@pytest.fixture
def permission_router(personas_config: dict, people_config: dict) -> PermissionRouter:
    """PermissionRouter instance."""
    return PermissionRouter(personas_config=personas_config, people_config=people_config)


@pytest.fixture
def persona_engine(personas_dir: Path, system_settings: dict) -> PersonaEngine:
    """PersonaEngine instance."""
    return PersonaEngine(personas_dir=personas_dir, system_settings=system_settings)


@pytest.fixture
def memory_manager(event_bus: EventBus) -> ThreeTierMemoryManager:
    """ThreeTierMemoryManager instance."""
    manager = ThreeTierMemoryManager(event_bus=event_bus)
    yield manager
    manager.close()


@pytest.fixture
def tts_dispatcher(personas_config: dict) -> TTSDispatcher:
    """TTSDispatcher instance."""
    return TTSDispatcher(
        personas_config=personas_config,
        inter_agent_speech_aloud=False,
        sound_dispatcher=SoundCueDispatcher(),
        ui_card_emitter=UICardEmitter(),
    )
