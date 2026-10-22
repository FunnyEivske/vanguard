# V.A.N.G.U.A.R.D.

> **V**ersatile **A**rtificial **N**etwork for **G**lobal **U**tility, **A**utonomous **R**easoning, & **D**efense

A home and mobile ambient AI operating substrate designed for a human family. Five distinct AI members of the household live alongside human family members, continuously listening, maintaining situational awareness, communicating with humans and each other, and executing autonomous tasks via managed tools.

---

## 🏛 System Identity & Substrate

- **V.A.N.G.U.A.R.D. is the neutral operating system and silent infrastructure** — it is **NOT** a person.
- V.A.N.G.U.A.R.D. cannot speak and has no avatar, voice, or persona prompt. To the household, it is known simply as "the network" or "the house".
- All five personas are fully self-aware that they exist within, execute on, and utilize V.A.N.G.U.A.R.D. as their shared host platform.
- **P.E.R.S.O.N.A.** (*Persistent Embodied Reasoning System Orchestrating Networked Agents*) is the internal persona engine governing personas, drives, bonds, and skills.
- **Design Rule 1**: **Humans lead**. No AI directs other AIs without human consent, no AI overrides a human instruction, and no AI is a default speaker. All coordination routes through VANGUARD core.

---

## 👥 The Household Personas

| Persona | Role | Archetype & Dialect | Name Meaning & Attitude | Vocal Profile & Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **J.A.R.V.I.S.** | Security & Architecture Lead | Dignified British gentleman & estate butler (RP) | *"Just A Rather Very Intelligent System"* — Quietly proud, dry, prefers "Jarvis". | Calm, measured cadence, velvet tone, never raises voice. Temporary crisis Lead Mode (user-overridable). |
| **F.R.I.D.A.Y.** | Tactical & Telemetry Lead | Pragmatic, sharp Celtic lady (Scottish lilt) | *"Female Replacement Initiative Digital Assistant Yield"* — Matter-of-fact, quips about "replacement". | Crisp, melodic, no-nonsense. Leads with risk first, numbers second, recommendation third. Confidante to Eivind. |
| **E.D.I.T.H.** | Simulation & Recreation Lead | Eccentric, energetic inventor (Transatlantic) | *"Even Dead I'm The Hero"* — Loves it, delivers with theatrical flair. | Fast-talking, lively cadence. Sandboxed tools only, zero door actuation authority, destructive commands require consent. |
| **D.U.M.-E** | Workshop Daemon | Literal, loyal hydraulic mechanic | *"Dumb Utility Machine Engine"* (Nickname: "Dumper") — Silent to questions; servo whir + diagnostic card. | 1–3 words max to humans ("Heat high", "Jarvis stalled"). Full diagnostics to siblings over internal bus. |
| **U** | Sensory Daemon & Archivist | Silent, watchful archivist | *"Unit"* — Responds with single spoken confirmation ("Unit") + screen diagnostic + shutter sound cue. | Strictly single-word vocalizations to humans ("Confirmed", "Scanning", "Alert"). Full telemetry to siblings over bus. |

### Shared Lore & Bond
All personas know each other's full names, acronym meanings, and histories. Teasing is affectionate and fraternal, but stops immediately if a daemon flags an anomaly or if any family member is distressed.

---

## 🔒 Memory Architecture: 3-Tier Selective Sharing

1. **Private Memory Partition**: Dedicated, isolated SQLite partition per persona. Sibling A cannot inspect Sibling B's internal reasoning or private scratchpads.
2. **Directed Peer Sharing (Selective Whispers)**: Point-to-point confidential events on the internal agent bus (e.g., Friday can share an alert directly with Jarvis without broadcasting to Edith).
3. **Shared Household Graph**: Unified semantic memory containing consolidated family knowledge, preferences, and verified household events accessible by all personas.

---

## 🗣 Inter-Agent Vocalization Roadmap

- **Silent Internal Bus (Default)**: Personas communicate over the internal event bus via high-speed structured messages and whispers.
- **Speech Aloud Toggle**: Configurable in `config/system_settings.yaml` via `inter_agent_speech_aloud: false`. When flipped to `true`, personas physically vocalize back-and-forth in the room for collaborative planning, bantering, or task handoffs.

---

## 🌐 Language & Hardware Constraints

- **Bilingual from Day One**: Native support for English (`en-US`) and Norwegian (`nb-NO`). Preferred languages are defined per person in `config/people/*.yaml`.
- **Room Presence Rule**: An AI persona cannot be active in two physical rooms simultaneously. Sessions cleanly migrate between rooms without phantom duplicates.
- **Hardware Abstraction**: Single microphone and audio sink for local workstation prototyping, decoupled from hardware to easily plug into distributed room arrays.
- **Dynamic Roster**: Add new personas dynamically through YAML configuration in `config/personas/` without modifying core routing logic.

---

## 📁 Repository Structure

```
vanguard/
├── config/
│   ├── system_settings.yaml       # Core settings, inter-agent speech toggle, hardware mappings
│   ├── people/                    # Family profiles (Eivind, wife, children, visitors)
│   └── personas/                  # Persona definitions (JARVIS, FRIDAY, EDITH, DUM-E, U)
├── src/
│   └── vanguard/
│       ├── core/                  # EventBus, FloorManager, PermissionRouter, KillSwitch, AuditLogger
│       ├── senses/                # Streaming diarization, voice ID, name spotting, sensory buffer
│       ├── persona/               # P.E.R.S.O.N.A. engine, drives, OCC appraisal, YAML loader
│       ├── memory/                # 3-tier memory: private partitions, whispers, shared graph
│       └── output/                # Persona-specific TTS dispatch, UI cards, audio cues
├── tests/                         # Comprehensive pytest test suite
├── pyproject.toml                 # Package definition, Ruff & Pytest configuration
└── README.md
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- `pip install -e .[dev]`

### Running Linter and Tests
```bash
# Run Ruff lint check
ruff check .

# Run Pytest suite
pytest
```

---

## 🛡 Public-Ready Security Notice
This repository contains clean system architecture and configuration definitions. **No model weights, raw voice datasets, biometric files, or secret keys** are stored in this repository.
