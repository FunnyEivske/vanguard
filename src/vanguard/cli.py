"""Command-Line Interface for V.A.N.G.U.A.R.D. Operational Runtime.

Commands:
- `vanguard run`: Start the live household runtime loop (mic/webcam or interactive console).
- `vanguard enroll`: Enroll a person's voice and/or face biometrics.
- `vanguard people`: List registered household members and enrolled biometric counts.
- `vanguard query`: Process a single utterance with voice/face biometric resolution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from vanguard.runtime import VanguardRuntime
from vanguard.senses.enrollment import (
    capture_microphone_wav,
    capture_webcam_frame,
    transcribe_audio_wav,
)


def _format_turn_output(result: object) -> str:
    from vanguard.runtime import TurnExecutionResult

    assert isinstance(result, TurnExecutionResult)
    lines: list[str] = []
    sp = result.speaker
    lines.append(
        f"[BIOMETRICS] Speaker: {sp.display_name} ({sp.person_id}) | Role: {sp.role} | "
        f"Confidence: {sp.confidence:.2f} (Voice={sp.voice_matched}, Face={sp.face_matched})"
    )
    if len(result.room_occupants) > 1:
        occ_names = ", ".join(f"{o.display_name} ({o.role})" for o in result.room_occupants)
        lines.append(f"[PRESENCE]   Room Occupants: {occ_names}")

    if result.halted_by_kill_switch:
        lines.append("[ALERT]      Emergency Kill Switch is engaged. All persona output halted.")
        return "\n".join(lines)

    if result.target_persona_id == "vanguard":
        lines.append("[SUBSTRATE]  V.A.N.G.U.A.R.D. is silent operating infrastructure and does not speak.")
        return "\n".join(lines)

    if not result.persona_response:
        lines.append("[FLOOR]      No persona addressed.")
        return "\n".join(lines)

    resp = result.persona_response
    pid_upper = resp.persona_id.upper()
    if result.roamed_from_room and result.roamed_from_room != result.output_room_id:
        lines.append(f"[ROAMING]    {pid_upper} migrated: {result.roamed_from_room} -> {result.output_room_id} (hidden speakers)")
    else:
        lines.append(f"[ROOM]       {pid_upper} active on '{result.output_room_id}' hidden speakers")

    if result.actuation_report:
        lines.append(f"[HOUSE I/O]  {json.dumps(result.actuation_report, ensure_ascii=False)}")

    if resp.spoken_text:
        lines.append(f"[{pid_upper}]     \"{resp.spoken_text}\"")
    else:
        lines.append(f"[{pid_upper}]     [SILENT TO HUMANS]")

    if resp.sound_cue:
        lines.append(f"[SOUND CUE]  ({resp.sound_cue})")
    if resp.ui_card:
        lines.append(f"[UI CARD]    {json.dumps(resp.ui_card, ensure_ascii=False)}")

    return "\n".join(lines)


async def _cmd_query(args: argparse.Namespace) -> int:
    runtime = VanguardRuntime(enable_hardware_audio=args.speak_aloud)
    try:
        res = await runtime.process_turn(
            text=args.text,
            room_id=args.room,
            wav_path=args.voice_wav,
            claimed_name=args.speaker,
            face_image_path=args.face_image,
        )
        print(_format_turn_output(res))
        return 0
    finally:
        runtime.close()


def _cmd_enroll(args: argparse.Namespace) -> int:
    runtime = VanguardRuntime(enable_hardware_audio=False)
    try:
        voice_wav = args.voice_wav
        face_img = None
        face_img_path = args.face_image

        if args.record_mic and not voice_wav:
            print(f"Recording {args.duration}s voice sample for '{args.person_id}'...")
            captured_wav = capture_microphone_wav(duration_seconds=args.duration)
            voice_wav = str(captured_wav)

        if args.capture_cam and not face_img_path:
            print(f"Capturing webcam frame for '{args.person_id}'...")
            captured = capture_webcam_frame(camera_index=args.camera_index)
            if isinstance(captured, Path):
                face_img_path = str(captured)
            else:
                face_img = captured

        langs = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        summary = runtime.enrollment_service.enroll_person(
            person_id=args.person_id,
            display_name=args.name,
            role=args.role,
            preferred_languages=langs,
            relationship=args.relationship,
            primary_room=args.room,
            voice_wav_path=voice_wav,
            face_image=face_img,
            face_image_path=face_img_path,
        )
        print(
            f"Enrolled '{summary.display_name}' (id={summary.person_id}, role={summary.role})\n"
            f"  Voice Print ID : {summary.voice_print_id or 'unchanged'}\n"
            f"  Face Print ID  : {summary.face_print_id or 'unchanged'}\n"
            f"  Profile YAML   : {summary.profile_path or 'config/people/children.yaml'}"
        )
        return 0
    finally:
        runtime.close()


def _cmd_people(args: argparse.Namespace) -> int:
    runtime = VanguardRuntime(enable_hardware_audio=False)
    try:
        print(f"{'ID':<16} {'NAME':<18} {'ROLE':<16} {'VOICE SAMPLES':<15} {'FACE SAMPLES':<15}")
        print("-" * 80)
        for pid, profile in sorted(runtime.voice_identifier.profiles.items()):
            v_count = len(runtime.biometric_store.get_embeddings(modality="voice", person_id=pid))
            f_count = len(runtime.biometric_store.get_embeddings(modality="face", person_id=pid))
            name = profile.get("name", pid.capitalize())
            role = profile.get("role", "visitor")
            print(f"{pid:<16} {name:<18} {role:<16} {v_count:<15} {f_count:<15}")
        return 0
    finally:
        runtime.close()


async def _cmd_run(args: argparse.Namespace) -> int:
    runtime = VanguardRuntime(enable_hardware_audio=args.speak_aloud)
    active_personas = ", ".join(p["display_name"] for p in runtime.persona_engine.personas.values())
    print("V.A.N.G.U.A.R.D. Operational Substrate Online.")
    print(f"Active Personas : {active_personas}")
    print(f"Active Room     : {args.room} | Hardware Audio: {runtime.enable_hardware_audio}")
    print(
        "Commands: '/quit' to exit, '/room <id>' to switch mic room, '/speaker <id>' to set speaker,\n"
        "          '/house' to inspect all rooms/cameras/doors/AI locations, '/camera <room> <person>' to spot someone.\n"
    )

    current_room = args.room
    current_speaker = args.speaker

    try:
        while True:
            if args.live_mic:
                print(f"\n[{current_room}] Listening on microphone ({args.listen_seconds}s)...")
                wav_file = capture_microphone_wav(duration_seconds=args.listen_seconds)
                face_frame = None
                if args.live_cam:
                    try:
                        face_frame = capture_webcam_frame(camera_index=args.camera_index)
                    except Exception:
                        face_frame = None
                text = transcribe_audio_wav(wav_file)
                if not text:
                    wav_file.unlink(missing_ok=True)
                    continue
                print(f"Heard: \"{text}\"")
                res = await runtime.process_turn(
                    text=text,
                    room_id=current_room,
                    wav_path=str(wav_file),
                    face_image=face_frame if not isinstance(face_frame, Path) else None,
                    face_image_path=str(face_frame) if isinstance(face_frame, Path) else None,
                )
                wav_file.unlink(missing_ok=True)
                print(_format_turn_output(res))
                continue

            try:
                line = input(f"[{current_room}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue
            if line.lower() in ("/quit", "/exit", "exit", "quit"):
                break
            if line.startswith("/room "):
                raw_r = line.split(maxsplit=1)[1].strip()
                current_room = runtime.house_controller.resolve_room_id(raw_r) or raw_r
                print(f"Switched active room microphone to '{current_room}'.")
                continue
            if line.startswith("/speaker "):
                current_speaker = line.split(maxsplit=1)[1].strip()
                print(f"Set active speaker context to '{current_speaker}'.")
                continue
            if line == "/house":
                occ_map = {
                    rid: [
                        {"person_id": o.person_id, "display_name": o.display_name, "role": o.role}
                        for o in runtime.identity_fusion.get_room_occupants(rid)
                    ]
                    for rid in runtime.house_controller.rooms
                }
                snap = runtime.house_controller.get_house_snapshot(occ_map)
                for rid, rinfo in snap.items():
                    ai_here = [
                        pid.upper()
                        for pid in runtime.persona_engine.personas
                        if runtime.floor_manager.get_persona_room(pid) == rid
                    ]
                    occ_str = ", ".join(o["display_name"] for o in rinfo["occupants"]) or "empty"
                    doors_str = ", ".join(
                        f"{d}={'locked' if st['is_locked'] else 'unlocked'}"
                        for d, st in rinfo["doors"].items()
                    )
                    print(
                        f"  • {rinfo['name']:<10} ({rid:<13}) | People: {occ_str:<18} | "
                        f"AIs: {', '.join(ai_here) or '-':<10} | Temp: {rinfo['temperature_c']}°C | "
                        f"Lights: {'ON' if rinfo['lights_on'] else 'OFF':<3} | Doors: {doors_str}"
                    )
                continue
            if line.startswith("/camera "):
                parts = line.split()
                if len(parts) >= 3:
                    cam_room = runtime.house_controller.resolve_room_id(parts[1]) or parts[1]
                    cam_person = parts[2].lower()
                    await runtime.identity_fusion.resolve_identity(
                        room_id=cam_room,
                        claimed_name=cam_person,
                        face_print_id=f"fp_{cam_person}_01",
                    )
                    print(f"[CAMERA] Spotted '{cam_person}' on {cam_room} camera.")
                continue

            res = await runtime.process_turn(
                text=line,
                room_id=current_room,
                claimed_name=current_speaker,
                wav_path=args.voice_wav,
                face_image_path=args.face_image,
            )
            print(_format_turn_output(res))
        return 0
    finally:
        runtime.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vanguard",
        description="V.A.N.G.U.A.R.D. Ambient Household Operating Substrate",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # vanguard run
    p_run = sub.add_parser("run", help="Start live V.A.N.G.U.A.R.D. runtime loop")
    p_run.add_argument("--room", default="living_room", help="Active room ID")
    p_run.add_argument("--speaker", default=None, help="Optional claimed speaker name/ID")
    p_run.add_argument("--voice-wav", default=None, help="Optional WAV file for voice biometric ID")
    p_run.add_argument("--face-image", default=None, help="Optional image path for facial biometric ID")
    p_run.add_argument("--speak-aloud", action="store_true", help="Enable physical TTS and sound cues")
    p_run.add_argument("--live-mic", action="store_true", help="Continuously capture audio from microphone")
    p_run.add_argument("--live-cam", action="store_true", help="Capture webcam frames for Face ID")
    p_run.add_argument("--listen-seconds", type=float, default=4.0, help="Mic window duration in seconds")
    p_run.add_argument("--camera-index", type=int, default=0, help="Webcam device index")

    # vanguard query
    p_query = sub.add_parser("query", help="Execute a single interaction turn")
    p_query.add_argument("text", help="Utterance text addressed to a persona")
    p_query.add_argument("--room", default="living_room", help="Room ID")
    p_query.add_argument("--speaker", default=None, help="Optional speaker name/ID")
    p_query.add_argument("--voice-wav", default=None, help="Path to WAV file for Voice ID")
    p_query.add_argument("--face-image", default=None, help="Path to image file for Face ID")
    p_query.add_argument("--speak-aloud", action="store_true", help="Speak persona response aloud")

    # vanguard enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a person's voice and/or face biometrics")
    p_enroll.add_argument("person_id", help="Person identifier (e.g. eivind, wife, child_elder, guest_ola)")
    p_enroll.add_argument("--name", default=None, help="Display name")
    p_enroll.add_argument("--role", default="family_member", choices=["family_head", "family_member", "child", "visitor"])
    p_enroll.add_argument("--relationship", default="Household Member", help="Relationship description")
    p_enroll.add_argument("--languages", default="nb-NO,en-US", help="Comma-separated preferred languages")
    p_enroll.add_argument("--room", default="living_room", help="Primary room ID")
    p_enroll.add_argument("--voice-wav", default=None, help="Enroll voiceprint from WAV file")
    p_enroll.add_argument("--face-image", default=None, help="Enroll face embedding from image file")
    p_enroll.add_argument("--record-mic", action="store_true", help="Record voice sample from live microphone")
    p_enroll.add_argument("--duration", type=float, default=4.0, help="Microphone recording duration in seconds")
    p_enroll.add_argument("--capture-cam", action="store_true", help="Capture face sample from live webcam")
    p_enroll.add_argument("--camera-index", type=int, default=0, help="Webcam index")

    # vanguard people
    sub.add_parser("people", help="List registered people and enrolled voice/face biometrics")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "enroll":
        return _cmd_enroll(args)
    if args.command == "people":
        return _cmd_people(args)
    if args.command == "query":
        return asyncio.run(_cmd_query(args))
    if args.command == "run":
        return asyncio.run(_cmd_run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())

