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

## 🌐 Biometric Recognition, Online-First Architecture & Linux Server Constraints

- **Voice & Face Biometric Recognition**: J.A.R.V.I.S., F.R.I.D.A.Y., E.D.I.T.H., D.U.M.-E, and U continuously identify different household members and visitors by **acoustic voiceprint** (`VoiceIdentifier`) and **facial recognition** (`FaceIdentifier`), fused in real time per room via `IdentityFusionEngine` and backed by `BiometricStore` (`data/biometrics.db`).
- **Online-First with Local Backup**:
  - **Primary (Online)**: Cloud LLM APIs (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`), Online Whisper STT, and ElevenLabs / OpenAI TTS.
  - **Backup (Local on Linux Server)**: Automatic failover to local Ollama (`http://localhost:11434`), local `faster-whisper` STT, and local `piper` / `espeak-ng` TTS over Linux ALSA (`aplay`/`arecord`), PulseAudio (`paplay`/`parecord`), PipeWire (`pw-play`), and V4L2 (`/dev/video0`).
- **Bilingual from Day One**: Native support for English (`en-US`) and Norwegian (`nb-NO`). Preferred languages are defined per person in `config/people/*.yaml`.
- **Room Presence Rule**: An AI persona cannot be active in two physical rooms simultaneously. Sessions cleanly migrate between rooms without phantom duplicates.
- **Dynamic Roster**: Add new personas dynamically through YAML configuration in `config/personas/` and enroll new people via `vanguard enroll` without modifying core routing logic.

---

## 📁 Repository Structure

```
vanguard/
├── config/
│   ├── system_settings.yaml       # Core settings, online-first/local-backup routing, Linux hardware mappings
│   ├── people/                    # Family profiles (Eivind, wife, children, visitors)
│   └── personas/                  # Persona definitions (JARVIS, FRIDAY, EDITH, DUM-E, U)
├── src/
│   └── vanguard/
│       ├── cli.py                 # Operational CLI (`vanguard run`, `enroll`, `people`, `query`)
│       ├── runtime.py             # Live household runtime orchestrator
│       ├── core/                  # EventBus, FloorManager, PermissionRouter, KillSwitch, AuditLogger
│       ├── senses/                # Voice ID, Face ID, IdentityFusion, BiometricStore, Enrollment, Diarization
│       ├── persona/               # P.E.R.S.O.N.A. engine, Online-First/Local-Backup LLMClient, drives, OCC
│       ├── memory/                # Persistent SQLite 3-tier memory: private partitions, whispers, shared graph
│       └── output/                # Persona-specific TTS dispatch, UI cards, Linux ALSA/Pulse audio cues
├── tests/                         # Comprehensive pytest test suite
├── pyproject.toml                 # Package definition, CLI scripts, Ruff & Pytest configuration
└── README.md
```

---

## 🚀 Quickstart (Linux Server)

### Prerequisites
- Python 3.10+
- Linux audio/video utilities (`alsa-utils`, `v4l-utils`, `ffmpeg`)
- `pip install -e .[dev]`

### Enrolling People (Voice & Face Biometrics)
```bash
# Enroll a person from live Linux microphone & V4L2 camera
vanguard enroll eivind --name "Eivind" --role family_head --record-mic --capture-cam

# Or enroll from existing WAV and face image files
vanguard enroll wife --name "Wife" --role family_head --voice-wav sample.wav --face-image face.jpg

# List registered people and enrolled voice/face samples
vanguard people
```

### Running the Live System
```bash
# Start interactive or live mic/camera household runtime
vanguard run --room living_room --speak-aloud

# Process a single query with voice/face verification
vanguard query "Jarvis, check the perimeter sensors." --room office --speaker eivind
```

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
