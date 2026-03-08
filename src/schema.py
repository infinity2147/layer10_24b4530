"""
Ontology / Schema definitions for the Layer10 Memory Graph.

Entity Types:    PERSON, PROJECT, ORGANIZATION, CONCEPT, ARTIFACT
Relation Types:  OWNS, PARTICIPATES_IN, REPORTS_TO, AUTHORED, MENTIONED_IN
Claim Types:     DECISION, REVERSAL, STATUS, RISK_FLAG, ASSIGNMENT, MEETING, FACT
Evidence:        Source pointer with offsets, timestamp, confidence
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import datetime, uuid, hashlib, json

# ── Enums ─────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    PERSON = "PERSON"
    PROJECT = "PROJECT"
    ORGANIZATION = "ORGANIZATION"
    CONCEPT = "CONCEPT"
    ARTIFACT = "ARTIFACT"

class ClaimType(str, Enum):
    DECISION   = "DECISION"    # "We decided to proceed with X"
    REVERSAL   = "REVERSAL"    # "We are reversing the earlier decision"
    STATUS     = "STATUS"      # "Project X is on track"
    RISK_FLAG  = "RISK_FLAG"   # "I want to flag a concern about X"
    ASSIGNMENT = "ASSIGNMENT"  # "Person P owns task T"
    MEETING    = "MEETING"     # Meeting occurred with attendees / outcomes
    FACT       = "FACT"        # General extracted fact

class RelationType(str, Enum):
    OWNS              = "OWNS"
    PARTICIPATES_IN   = "PARTICIPATES_IN"
    REPORTS_TO        = "REPORTS_TO"
    AUTHORED          = "AUTHORED"
    MENTIONED_IN      = "MENTIONED_IN"
    CONTRADICTS       = "CONTRADICTS"
    SUPERSEDES        = "SUPERSEDES"
    ALIASES           = "ALIASES"

class ValidityStatus(str, Enum):
    CURRENT    = "CURRENT"     # Currently believed to be true
    HISTORICAL = "HISTORICAL"  # Was true, no longer current
    DISPUTED   = "DISPUTED"    # Conflicting evidence exists
    RETRACTED  = "RETRACTED"   # Explicitly reversed/retracted
    REDACTED   = "REDACTED"    # Source deleted or redacted

# ── Evidence Pointer ───────────────────────────────────────────────────────────

@dataclass
class EvidencePointer:
    """Every claim must point to at least one evidence pointer."""
    evidence_id:   str
    source_id:     str           # message/doc id
    source_type:   str           # "email", "slack", "jira_ticket", etc.
    excerpt:       str           # Exact text snippet
    char_start:    int           # Character offset in source body
    char_end:      int
    timestamp:     str           # ISO8601 event time
    ingested_at:   str           # ISO8601 ingestion time
    confidence:    float         # 0.0 – 1.0
    extractor_ver: str           # e.g. "v1.2.0-claude-sonnet-4"
    schema_ver:    str           # e.g. "2026-01-ontology-v3"

    def to_dict(self): return asdict(self)

# ── Entity ─────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    entity_id:      str
    entity_type:    EntityType
    canonical_name: str
    aliases:        List[str]              = field(default_factory=list)
    attributes:     Dict[str, Any]         = field(default_factory=dict)
    evidence_ids:   List[str]              = field(default_factory=list)
    merged_from:    List[str]              = field(default_factory=list)  # audit trail
    created_at:     str                    = field(default_factory=lambda: datetime.datetime.utcnow().isoformat()+"Z")
    updated_at:     str                    = field(default_factory=lambda: datetime.datetime.utcnow().isoformat()+"Z")

    def to_dict(self): return {**asdict(self), "entity_type": self.entity_type.value}

# ── Claim ──────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    claim_id:       str
    claim_type:     ClaimType
    subject_id:     str                    # entity_id
    predicate:      str                    # human-readable relation
    object_id:      Optional[str]          # entity_id or None
    object_value:   Optional[str]          # literal value if no entity
    validity:       ValidityStatus
    valid_from:     Optional[str]          # ISO8601
    valid_until:    Optional[str]          # ISO8601 or None = current
    evidence_ids:   List[str]             = field(default_factory=list)
    superseded_by:  Optional[str]         = None   # claim_id
    supersedes:     Optional[str]         = None   # claim_id
    confidence:     float                 = 1.0
    created_at:     str                   = field(default_factory=lambda: datetime.datetime.utcnow().isoformat()+"Z")
    tags:           List[str]             = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["claim_type"] = self.claim_type.value
        d["validity"] = self.validity.value
        return d

# ── Relation (edge in the graph) ───────────────────────────────────────────────

@dataclass
class Relation:
    relation_id:    str
    relation_type:  RelationType
    from_id:        str            # entity_id
    to_id:          str            # entity_id
    attributes:     Dict[str, Any] = field(default_factory=dict)
    evidence_ids:   List[str]     = field(default_factory=list)
    valid_from:     Optional[str] = None
    valid_until:    Optional[str] = None

    def to_dict(self):
        d = asdict(self)
        d["relation_type"] = self.relation_type.value
        return d

# ── Artifact (source document) ─────────────────────────────────────────────────

@dataclass
class Artifact:
    artifact_id:   str
    artifact_type: str           # "email", "slack_message", "jira_ticket"
    source_id:     str           # original corpus id
    content_hash:  str           # SHA-256 of normalized body
    timestamp:     str
    author_id:     str
    recipient_ids: List[str]     = field(default_factory=list)
    metadata:      Dict[str, Any]= field(default_factory=dict)
    is_duplicate:  bool          = False
    duplicate_of:  Optional[str] = None
    is_redacted:   bool          = False

    def to_dict(self): return asdict(self)

# ── Memory Graph (in-memory store) ────────────────────────────────────────────

class MemoryGraph:
    def __init__(self):
        self.entities:  Dict[str, Entity]        = {}
        self.claims:    Dict[str, Claim]          = {}
        self.relations: Dict[str, Relation]       = {}
        self.artifacts: Dict[str, Artifact]       = {}
        self.evidence:  Dict[str, EvidencePointer]= {}

    def add_entity(self, e: Entity):    self.entities[e.entity_id]   = e
    def add_claim(self, c: Claim):      self.claims[c.claim_id]       = c
    def add_relation(self, r: Relation):self.relations[r.relation_id] = r
    def add_artifact(self, a: Artifact):self.artifacts[a.artifact_id] = a
    def add_evidence(self, ev: EvidencePointer): self.evidence[ev.evidence_id] = ev

    def to_dict(self):
        return {
            "entities":  {k: v.to_dict() for k,v in self.entities.items()},
            "claims":    {k: v.to_dict() for k,v in self.claims.items()},
            "relations": {k: v.to_dict() for k,v in self.relations.items()},
            "artifacts": {k: v.to_dict() for k,v in self.artifacts.items()},
            "evidence":  {k: v.to_dict() for k,v in self.evidence.items()},
            "stats": {
                "entities":  len(self.entities),
                "claims":    len(self.claims),
                "relations": len(self.relations),
                "artifacts": len(self.artifacts),
                "evidence":  len(self.evidence),
            }
        }

    def save(self, path: str):
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"
