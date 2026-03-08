"""
Structured Extraction Pipeline
================================
Turns raw corpus messages into typed, grounded schema objects.

Design choices:
 - Rule-based + keyword pattern extraction (deterministic, reproducible, no model needed offline)
 - Every extracted claim carries an EvidencePointer with char offsets
 - Validation layer catches low-confidence / incomplete extractions
 - Versioning: extractor_ver + schema_ver stamped on every evidence pointer
 - Quality gates: confidence thresholds, minimum excerpt length
"""

import re, json, hashlib, uuid, datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from schema import (
    Entity, Claim, Relation, Artifact, EvidencePointer, MemoryGraph,
    EntityType, ClaimType, RelationType, ValidityStatus, new_id
)

EXTRACTOR_VER = "v1.0.0-deterministic"
SCHEMA_VER    = "2026-01-ontology-v1"

# ── Pattern banks ──────────────────────────────────────────────────────────────

DECISION_PATTERNS = [
    r"(decided|decision|approved|confirmed|agreed|will proceed|going forward|move forward)\s+(?:to\s+)?(.{10,80}?)(?:\.|$)",
    r"(we are proceeding|we have decided|decision is)\s+(?:to\s+)?(.{10,80}?)(?:\.|$)",
]
REVERSAL_PATTERNS = [
    r"(walk\s?back|reverses?|reversing|retract|pulling back|pausing|cancell?ing|no longer|revised\s+position)\s+(?:the\s+)?(.{10,80}?)(?:\.|$)",
    r"(this reverses?|we are pausing|hold all|prior decision|earlier statement)\s*(.{0,80}?)(?:\.|$)",
]
STATUS_PATTERNS = [
    r"(on track|behind schedule|at risk|completed|blocked|in progress|delayed|closed)\s*[,.\-–]?\s*(.{0,60}?)(?:\.|$)",
    r"status[:\s]+(.{10,80}?)(?:\.|$)",
]
RISK_PATTERNS = [
    r"(concern|risk|flag|issue|problem|warning|caution|raise\s+a\s+concern)\s+(?:about|re:|regarding|with)?\s*(.{10,80}?)(?:\.|$)",
    r"may not survive|not compliant|fall below|informally but not in writing",
]
ASSIGNMENT_PATTERNS = [
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:takes?|will|to|:)\s+(?:point|own|handle|lead|coordinate|prepare|circulate)\s+(.{10,80}?)(?:\.|$)",
    r"Action:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+to\s+(.{10,80}?)(?:\.|$)",
    r"Owner:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]

# Map claim type string → enum
CLAIM_TYPE_MAP = {
    "decision":     ClaimType.DECISION,
    "reversal":     ClaimType.REVERSAL,
    "status":       ClaimType.STATUS,
    "concern":      ClaimType.RISK_FLAG,
    "meeting_notes":ClaimType.MEETING,
}

def extract_char_span(body: str, excerpt: str) -> Tuple[int, int]:
    idx = body.find(excerpt[:40])
    if idx == -1:
        return 0, min(len(excerpt), len(body))
    return idx, idx + len(excerpt)

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\-.,/:@#\'\"!?]', '', text)
    return text

def confidence_score(excerpt: str, pattern_type: str, msg_type: str) -> float:
    """Heuristic confidence: longer excerpts and matching msg_type → higher."""
    base = 0.6
    if len(excerpt) > 50:  base += 0.1
    if len(excerpt) > 100: base += 0.1
    if msg_type == pattern_type: base += 0.15
    return min(base, 0.95)

def extract_entities_from_people(people_list: list, graph: MemoryGraph):
    """Bootstrap known persons into the graph."""
    for p in people_list:
        eid = f"ent_person_{p['id']}"
        e = Entity(
            entity_id=eid,
            entity_type=EntityType.PERSON,
            canonical_name=p["canonical"],
            aliases=p["aliases"],
            attributes={"role": p.get("role", "")},
        )
        graph.add_entity(e)

def extract_entities_from_projects(projects_list: list, graph: MemoryGraph):
    for p in projects_list:
        eid = f"ent_project_{p['id']}"
        e = Entity(
            entity_id=eid,
            entity_type=EntityType.PROJECT,
            canonical_name=p["name"],
            aliases=p["aliases"],
        )
        graph.add_entity(e)

def find_entity_in_text(text: str, graph: MemoryGraph) -> Optional[str]:
    """Return entity_id of first entity whose name/alias appears in text."""
    text_lower = text.lower()
    for eid, entity in graph.entities.items():
        if entity.canonical_name.lower() in text_lower:
            return eid
        for alias in entity.aliases:
            if alias.lower() in text_lower:
                return eid
    return None

def find_project_in_text(text: str, graph: MemoryGraph) -> Optional[str]:
    """Return project entity_id from text."""
    text_lower = text.lower()
    for eid, entity in graph.entities.items():
        if entity.entity_type != EntityType.PROJECT:
            continue
        if entity.canonical_name.lower() in text_lower:
            return eid
        for alias in entity.aliases:
            if alias.lower() in text_lower:
                return eid
    return None

def resolve_sender(from_alias: str, graph: MemoryGraph) -> Optional[str]:
    """Resolve email alias → canonical entity_id."""
    from_lower = from_alias.lower()
    for eid, entity in graph.entities.items():
        if entity.entity_type != EntityType.PERSON:
            continue
        for alias in entity.aliases:
            if alias.lower() == from_lower or alias.lower() in from_lower:
                return eid
    return None

def extract_claims_from_message(msg: dict, graph: MemoryGraph) -> List[Claim]:
    """
    Extract typed claims from a single message.
    Returns list of Claim objects (evidence pointers also created and added to graph).
    """
    claims = []
    body = msg["body"]
    msg_type = msg.get("msg_type", "")
    timestamp = msg["timestamp"]
    source_id = msg["id"]

    # Resolve author
    author_eid = resolve_sender(msg["from"], graph)

    # Create artifact record
    art = Artifact(
        artifact_id=f"art_{source_id}",
        artifact_type="email",
        source_id=source_id,
        content_hash=msg["content_hash"],
        timestamp=timestamp,
        author_id=author_eid or "unk",
        recipient_ids=[resolve_sender(r, graph) or r for r in msg.get("to", [])],
        metadata={
            "subject": msg["subject"],
            "thread_id": msg["thread_id"],
            "word_count": msg["word_count"],
        }
    )
    graph.add_artifact(art)

    def make_evidence(excerpt: str, conf: float) -> EvidencePointer:
        exc_clean = normalize_text(excerpt[:200])
        cstart, cend = extract_char_span(body, exc_clean)
        ev = EvidencePointer(
            evidence_id=new_id("ev"),
            source_id=source_id,
            source_type="email",
            excerpt=exc_clean,
            char_start=cstart,
            char_end=cend,
            timestamp=timestamp,
            ingested_at=datetime.datetime.utcnow().isoformat()+"Z",
            confidence=conf,
            extractor_ver=EXTRACTOR_VER,
            schema_ver=SCHEMA_VER,
        )
        graph.add_evidence(ev)
        return ev

    def make_claim(ctype: ClaimType, subject_eid: Optional[str], predicate: str,
                   obj_id: Optional[str], obj_val: Optional[str],
                   ev: EvidencePointer, validity: ValidityStatus = ValidityStatus.CURRENT) -> Claim:
        c = Claim(
            claim_id=new_id("clm"),
            claim_type=ctype,
            subject_id=subject_eid or "unk",
            predicate=predicate,
            object_id=obj_id,
            object_value=obj_val,
            validity=validity,
            valid_from=timestamp,
            valid_until=None,
            evidence_ids=[ev.evidence_id],
            confidence=ev.confidence,
            tags=[msg_type],
        )
        return c

    # ── DECISION extraction ────────────────────────────────────────────────
    if msg_type in ("decision", "meeting_notes"):
        for pattern in DECISION_PATTERNS:
            for m in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
                excerpt = m.group(0)
                conf = confidence_score(excerpt, "decision", msg_type)
                if conf < 0.55: continue
                ev = make_evidence(excerpt, conf)
                proj_eid = find_project_in_text(body, graph)
                c = make_claim(
                    ClaimType.DECISION, author_eid,
                    f"decided: {normalize_text(excerpt)[:80]}",
                    proj_eid, excerpt[:100], ev
                )
                graph.add_claim(c)
                claims.append(c)
                break

    # ── REVERSAL extraction ────────────────────────────────────────────────
    if msg_type == "reversal":
        for pattern in REVERSAL_PATTERNS:
            for m in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
                excerpt = m.group(0)
                conf = confidence_score(excerpt, "reversal", msg_type) + 0.1
                ev = make_evidence(excerpt, min(conf, 0.95))
                proj_eid = find_project_in_text(body, graph)
                c = make_claim(
                    ClaimType.REVERSAL, author_eid,
                    f"reversed: {normalize_text(excerpt)[:80]}",
                    proj_eid, excerpt[:100], ev, ValidityStatus.HISTORICAL
                )
                graph.add_claim(c)
                claims.append(c)
                break

    # ── STATUS extraction ──────────────────────────────────────────────────
    if msg_type == "status":
        for pattern in STATUS_PATTERNS:
            for m in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
                excerpt = m.group(0)
                conf = confidence_score(excerpt, "status", msg_type)
                if conf < 0.5: continue
                ev = make_evidence(excerpt, conf)
                proj_eid = find_project_in_text(body, graph)
                c = make_claim(
                    ClaimType.STATUS, proj_eid or author_eid,
                    "status",
                    None, normalize_text(excerpt)[:120], ev
                )
                graph.add_claim(c)
                claims.append(c)
                break

    # ── RISK_FLAG extraction ───────────────────────────────────────────────
    if msg_type == "concern":
        for pattern in RISK_PATTERNS:
            for m in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
                excerpt = m.group(0)
                conf = confidence_score(excerpt, "concern", msg_type)
                ev = make_evidence(excerpt, conf)
                proj_eid = find_project_in_text(body, graph)
                c = make_claim(
                    ClaimType.RISK_FLAG, author_eid,
                    "flagged_risk",
                    proj_eid, normalize_text(excerpt)[:150], ev
                )
                graph.add_claim(c)
                claims.append(c)
                break

    # ── ASSIGNMENT extraction ──────────────────────────────────────────────
    for pattern in ASSIGNMENT_PATTERNS:
        for m in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
            excerpt = m.group(0)
            conf = confidence_score(excerpt, "assignment", msg_type)
            if conf < 0.5: continue
            ev = make_evidence(excerpt, conf)
            # Try to find the person entity
            person_name = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
            assignee_eid = find_entity_in_text(person_name, graph)
            task_val = m.group(2) if m.lastindex and m.lastindex >= 2 else excerpt
            c = make_claim(
                ClaimType.ASSIGNMENT, assignee_eid or author_eid,
                "assigned_to_task",
                None, normalize_text(task_val)[:120], ev
            )
            graph.add_claim(c)
            claims.append(c)

    # ── MEETING extraction ─────────────────────────────────────────────────
    if msg_type == "meeting_notes":
        attendee_pattern = r"Attendees?:\s*(.+?)(?:\n|$)"
        for m in re.finditer(attendee_pattern, body, re.IGNORECASE):
            excerpt = m.group(0)
            conf = 0.85
            ev = make_evidence(excerpt, conf)
            proj_eid = find_project_in_text(body, graph)
            c = make_claim(
                ClaimType.MEETING, author_eid,
                "meeting_occurred",
                proj_eid, normalize_text(excerpt)[:150], ev
            )
            graph.add_claim(c)
            claims.append(c)

    # Always add AUTHORED relation
    if author_eid:
        rel = Relation(
            relation_id=new_id("rel"),
            relation_type=RelationType.AUTHORED,
            from_id=author_eid,
            to_id=f"art_{source_id}",
            evidence_ids=[],
            valid_from=timestamp,
        )
        graph.add_relation(rel)

    # Add MENTIONED_IN for all entities found in body
    for eid in set(
        eid for eid in [find_entity_in_text(body, graph)] if eid and eid != author_eid
    ):
        rel = Relation(
            relation_id=new_id("rel"),
            relation_type=RelationType.MENTIONED_IN,
            from_id=eid,
            to_id=f"art_{source_id}",
            evidence_ids=[],
            valid_from=timestamp,
        )
        graph.add_relation(rel)

    return claims

def validate_claim(c: Claim, graph: MemoryGraph) -> Tuple[bool, str]:
    """Quality gate: reject low-quality claims."""
    if not c.evidence_ids:
        return False, "no_evidence"
    evs = [graph.evidence[eid] for eid in c.evidence_ids if eid in graph.evidence]
    if not evs:
        return False, "missing_evidence_pointers"
    if all(ev.confidence < 0.55 for ev in evs):
        return False, f"all_evidence_below_confidence_threshold"
    if c.subject_id == "unk" and c.object_id is None and not c.object_value:
        return False, "no_subject_and_no_object"
    return True, "ok"

def run_extraction(corpus_path: str, graph: MemoryGraph) -> dict:
    corpus = json.loads(Path(corpus_path).read_text())

    # Bootstrap known entities
    extract_entities_from_people(corpus["people"], graph)
    extract_entities_from_projects(corpus["projects"], graph)

    stats = {"total": 0, "valid_claims": 0, "rejected_claims": 0, "artifacts": 0}

    for msg in corpus["messages"]:
        stats["total"] += 1
        claims = extract_claims_from_message(msg, graph)
        stats["artifacts"] += 1
        for c in claims:
            valid, reason = validate_claim(c, graph)
            if valid:
                stats["valid_claims"] += 1
            else:
                del graph.claims[c.claim_id]
                stats["rejected_claims"] += 1

    print(f"Extraction complete: {stats}")
    return stats

if __name__ == "__main__":
    g = MemoryGraph()
    corpus_path = "/home/claude/layer10/data/raw/corpus.json"
    stats = run_extraction(corpus_path, g)
    out = "/home/claude/layer10/data/processed/graph_raw.json"
    g.save(out)
    print(f"Raw graph saved → {out}")
    print(f"Graph: {g.to_dict()['stats']}")
