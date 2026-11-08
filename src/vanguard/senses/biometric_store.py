"""Persistent Biometric Store & People Registry for V.A.N.G.U.A.R.D.

Manages enrolled voiceprints and facial embeddings in SQLite, performs cosine similarity
matching, and maintains the unified registry of household members and visitors.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def normalize_vector(vec: list[float]) -> list[float]:
    """L2-normalize a feature embedding vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two feature vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    norm_a = normalize_vector(vec_a)
    norm_b = normalize_vector(vec_b)
    return sum(a * b for a, b in zip(norm_a, norm_b, strict=False))


@dataclass
class BiometricRecord:
    """Stored biometric sample for a person."""

    record_id: int
    person_id: str
    modality: str  # "voice" or "face"
    print_id: str
    embedding: list[float]
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiometricMatch:
    """Best biometric match result."""

    matched: bool
    person_id: str
    print_id: str | None
    similarity: float
    modality: str


def load_people_profiles(people_dir: Path | None = None, raw_profiles: dict[str, dict] | None = None) -> dict[str, dict[str, Any]]:
    """Load and normalize all person profiles from YAML directory or raw dict.

    Expands multi-member YAML structures (such as `children.yaml` with `members: [...]`)
    into flat person_id -> profile entries.
    """
    normalized: dict[str, dict[str, Any]] = {}

    sources: list[dict[str, Any]] = []
    if raw_profiles:
        for key, val in raw_profiles.items():
            if isinstance(val, dict):
                entry = dict(val)
                if "id" not in entry and "members" not in entry and "template_id" not in entry:
                    entry["id"] = key
                sources.append(entry)

    if people_dir and people_dir.exists():
        for p_file in sorted(people_dir.glob("*.yaml")):
            with open(p_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    sources.append(data)

    for data in sources:
        if "members" in data and isinstance(data["members"], list):
            safety = data.get("safety_constraints", {})
            for member in data["members"]:
                if isinstance(member, dict) and "id" in member:
                    m_copy = dict(member)
                    m_copy.setdefault("safety_constraints", safety)
                    normalized[m_copy["id"]] = m_copy
        elif "id" in data:
            normalized[data["id"]] = dict(data)
        elif "template_id" in data:
            t_copy = dict(data)
            t_copy.setdefault("id", "visitor")
            normalized["visitor"] = t_copy

    return normalized


class BiometricStore:
    """SQLite-backed store for voiceprints and facial embeddings."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables for biometric embeddings and dynamically enrolled people."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS biometric_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    print_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_biometric_modality_person
                ON biometric_embeddings (modality, person_id)
                """
            )

    def enroll_embedding(
        self,
        person_id: str,
        modality: str,
        embedding: list[float],
        print_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BiometricRecord:
        """Store a normalized voice or face embedding for a person."""
        mod = modality.lower().strip()
        if mod not in ("voice", "face"):
            raise ValueError(f"Unsupported biometric modality: {modality}")
        if not embedding:
            raise ValueError("Embedding vector cannot be empty.")

        pid = person_id.lower().strip()
        norm_emb = normalize_vector([float(x) for x in embedding])
        prefix = "vp" if mod == "voice" else "fp"
        resolved_print_id = print_id or f"{prefix}_{pid}_{int(time.time())}"
        now = time.time()

        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO biometric_embeddings (person_id, modality, print_id, embedding_json, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    mod,
                    resolved_print_id,
                    json.dumps(norm_emb),
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            rec_id = cur.lastrowid or 0

        return BiometricRecord(
            record_id=rec_id,
            person_id=pid,
            modality=mod,
            print_id=resolved_print_id,
            embedding=norm_emb,
            created_at=now,
            metadata=metadata or {},
        )

    def get_embeddings(self, modality: str | None = None, person_id: str | None = None) -> list[BiometricRecord]:
        """Retrieve stored biometric records filtered by modality and/or person_id."""
        query = "SELECT id, person_id, modality, print_id, embedding_json, created_at, metadata_json FROM biometric_embeddings WHERE 1=1"
        params: list[Any] = []
        if modality:
            query += " AND modality = ?"
            params.append(modality.lower().strip())
        if person_id:
            query += " AND person_id = ?"
            params.append(person_id.lower().strip())

        with self.conn:
            cur = self.conn.cursor()
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        records: list[BiometricRecord] = []
        for r in rows:
            records.append(
                BiometricRecord(
                    record_id=r[0],
                    person_id=r[1],
                    modality=r[2],
                    print_id=r[3],
                    embedding=json.loads(r[4]),
                    created_at=r[5],
                    metadata=json.loads(r[6]) if r[6] else {},
                )
            )
        return records

    def match_embedding(
        self,
        modality: str,
        query_embedding: list[float],
        threshold: float = 0.75,
    ) -> BiometricMatch:
        """Compare a query embedding against all enrolled embeddings of the given modality."""
        candidates = self.get_embeddings(modality=modality)
        if not candidates or not query_embedding:
            return BiometricMatch(
                matched=False,
                person_id="visitor",
                print_id=None,
                similarity=0.0,
                modality=modality,
            )

        best_sim = -1.0
        best_record: BiometricRecord | None = None

        for rec in candidates:
            if len(rec.embedding) != len(query_embedding):
                continue
            sim = cosine_similarity(query_embedding, rec.embedding)
            if sim > best_sim:
                best_sim = sim
                best_record = rec

        if best_record is not None and best_sim >= threshold:
            return BiometricMatch(
                matched=True,
                person_id=best_record.person_id,
                print_id=best_record.print_id,
                similarity=round(float(best_sim), 4),
                modality=modality,
            )

        return BiometricMatch(
            matched=False,
            person_id="visitor",
            print_id=best_record.print_id if best_record else None,
            similarity=round(max(0.0, float(best_sim)), 4),
            modality=modality,
        )

    def delete_person_biometrics(self, person_id: str, modality: str | None = None) -> int:
        """Remove enrolled biometric embeddings for a person."""
        with self.conn:
            if modality:
                cur = self.conn.execute(
                    "DELETE FROM biometric_embeddings WHERE person_id = ? AND modality = ?",
                    (person_id.lower().strip(), modality.lower().strip()),
                )
            else:
                cur = self.conn.execute(
                    "DELETE FROM biometric_embeddings WHERE person_id = ?",
                    (person_id.lower().strip(),),
                )
            return cur.rowcount

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

