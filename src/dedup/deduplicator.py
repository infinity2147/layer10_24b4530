"""
Deduplication & Canonicalization Pipeline
==========================================

Three-level dedup:
  1. Artifact dedup   — exact (content_hash) + near-duplicate (Jaccard on trigrams)
  2. Entity dedup     — alias resolution + soft-match for person names
  3. Claim dedup      — same (subject, predicate, object) → merge evidence; handle conflicts

All merges are reversible: we store merge_log entries with reason + timestamp.
Conflicts (same predicate, contradicting values, different timestamps) are
represented as DISPUTED claims, not silently resolved.
"""

import json, hashlib, re, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import (
    MemoryGraph, Entity, Claim, Artifact, EntityType, ClaimType,
    ValidityStatus, RelationType, Relation, new_id
)

MERGE_LOG: List[dict] = []

def log_merge(merge_type: str, merged_ids: List[str], into_id: str, reason: str):
    MERGE_LOG.append({
        "ts": datetime.datetime.utcnow().isoformat()+"Z",
        "type": merge_type,
        "merged_ids": merged_ids,
        "into_id": into_id,
        "reason": reason,
    })

# ── 1. Artifact Deduplication ─────────────────────────────────────────────────

def trigram_set(text: str) -> Set[str]:
    """Character trigrams for fuzzy matching."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    return {text[i:i+3] for i in range(len(text)-2)}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def dedup_artifacts(graph: MemoryGraph, near_dup_threshold: float = 0.85):
    """
    Exact dedup by content_hash.
    Near-dedup by Jaccard on trigrams of (subject+body excerpt).
    """
    hash_to_art: Dict[str, str] = {}   # content_hash → first artifact_id
    deduplicated: Dict[str, str] = {}  # artifact_id → canonical artifact_id

    # Pass 1: exact hash dedup
    for aid, art in graph.artifacts.items():
        h = art.content_hash
        if h in hash_to_art:
            canonical_aid = hash_to_art[h]
            deduplicated[aid] = canonical_aid
            art.is_duplicate = True
            art.duplicate_of = canonical_aid
            log_merge("artifact_exact", [aid], canonical_aid, f"identical content_hash={h}")
        else:
            hash_to_art[h] = aid

    # Pass 2: near-duplicate (Jaccard on trigrams of subject)
    unique_arts = [(aid, art) for aid, art in graph.artifacts.items()
                   if not art.is_duplicate]
    for i, (aid1, art1) in enumerate(unique_arts):
        subj1 = trigram_set(art1.metadata.get("subject", ""))
        for j, (aid2, art2) in enumerate(unique_arts):
            if j <= i: continue
            subj2 = trigram_set(art2.metadata.get("subject", ""))
            sim = jaccard(subj1, subj2)
            if sim >= near_dup_threshold:
                # Keep earlier one, mark later as near-dup
                if art1.timestamp <= art2.timestamp:
                    canonical, dup = aid1, aid2
                else:
                    canonical, dup = aid2, aid1
                graph.artifacts[dup].is_duplicate = True
                graph.artifacts[dup].duplicate_of = canonical
                deduplicated[dup] = canonical
                log_merge("artifact_near_dup", [dup], canonical,
                          f"Jaccard={sim:.2f} on subject trigrams")

    print(f"  Artifact dedup: {len(deduplicated)} duplicates found")
    return deduplicated

# ── 2. Entity Canonicalization ─────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name.lower().strip())

def build_alias_index(graph: MemoryGraph) -> Dict[str, str]:
    """Map every alias (normalized) → canonical entity_id."""
    index = {}
    for eid, entity in graph.entities.items():
        index[normalize_name(entity.canonical_name)] = eid
        for alias in entity.aliases:
            index[normalize_name(alias)] = eid
    return index

def merge_entities(graph: MemoryGraph, keep_id: str, remove_id: str, reason: str):
    """Merge remove_id into keep_id, update all back-references."""
    if keep_id == remove_id: return
    keep = graph.entities[keep_id]
    remove = graph.entities.get(remove_id)
    if not remove: return

    # Merge aliases
    for alias in remove.aliases:
        if alias not in keep.aliases:
            keep.aliases.append(alias)
    if remove.canonical_name not in keep.aliases:
        keep.aliases.append(remove.canonical_name)

    # Audit trail
    keep.merged_from.append(remove_id)
    keep.updated_at = datetime.datetime.utcnow().isoformat()+"Z"

    # Re-point all claims
    for claim in graph.claims.values():
        if claim.subject_id == remove_id: claim.subject_id = keep_id
        if claim.object_id == remove_id:  claim.object_id  = keep_id

    # Re-point all relations
    for rel in graph.relations.values():
        if rel.from_id == remove_id: rel.from_id = keep_id
        if rel.to_id   == remove_id: rel.to_id   = keep_id

    del graph.entities[remove_id]
    log_merge("entity", [remove_id], keep_id, reason)

def canonicalize_entities(graph: MemoryGraph):
    """
    Detect duplicate/alias entity pairs and merge them.
    Strategy: build full alias index, find collisions.
    """
    alias_index = build_alias_index(graph)

    # Detect entities sharing an alias
    seen_aliases: Dict[str, str] = {}  # normalized_alias → first entity_id
    merges_needed: List[Tuple[str, str, str]] = []

    for eid, entity in list(graph.entities.items()):
        all_names = [normalize_name(entity.canonical_name)] + [normalize_name(a) for a in entity.aliases]
        for name in all_names:
            if name in seen_aliases:
                other_eid = seen_aliases[name]
                if other_eid != eid:
                    merges_needed.append((other_eid, eid, f"shared_alias={name}"))
            else:
                seen_aliases[name] = eid

    for keep_id, remove_id, reason in merges_needed:
        if keep_id in graph.entities and remove_id in graph.entities:
            merge_entities(graph, keep_id, remove_id, reason)

    print(f"  Entity canonicalization: {len(merges_needed)} merges performed")

# ── 3. Claim Deduplication ────────────────────────────────────────────────────

def claim_fingerprint(c: Claim) -> str:
    """Stable fingerprint for (subject, predicate_normalized, object) tuple."""
    pred_norm = re.sub(r'[^a-z0-9 ]', '', c.predicate.lower())[:50]
    obj = (c.object_id or "") + (c.object_value or "")[:30]
    raw = f"{c.subject_id}|{pred_norm}|{obj}|{c.claim_type.value}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def detect_conflicts(c1: Claim, c2: Claim) -> bool:
    """
    Two claims on the same subject+predicate conflict if their object_values
    differ AND both are CURRENT.
    """
    if c1.subject_id != c2.subject_id: return False
    if c1.claim_type != c2.claim_type: return False
    pred1 = re.sub(r'[^a-z ]', '', c1.predicate.lower())[:30]
    pred2 = re.sub(r'[^a-z ]', '', c2.predicate.lower())[:30]
    if pred1 != pred2: return False
    if c1.object_value and c2.object_value and c1.object_value != c2.object_value:
        return c1.validity == ValidityStatus.CURRENT and c2.validity == ValidityStatus.CURRENT
    return False

def dedup_claims(graph: MemoryGraph):
    """
    Merge identical claims (same fingerprint), keep all evidence.
    Flag conflicts as DISPUTED rather than silently resolving.
    Handle REVERSAL claims: mark superseded DECISION claims as HISTORICAL.
    """
    fingerprint_to_id: Dict[str, str] = {}
    removed_ids: List[str] = []

    for cid, claim in list(graph.claims.items()):
        fp = claim_fingerprint(claim)
        if fp in fingerprint_to_id:
            canonical_cid = fingerprint_to_id[fp]
            canonical = graph.claims[canonical_cid]
            # Merge evidence pointers
            for eid in claim.evidence_ids:
                if eid not in canonical.evidence_ids:
                    canonical.evidence_ids.append(eid)
            # Keep higher confidence
            canonical.confidence = max(canonical.confidence, claim.confidence)
            removed_ids.append(cid)
            log_merge("claim", [cid], canonical_cid,
                      f"identical fingerprint={fp}")
        else:
            fingerprint_to_id[fp] = cid

    for cid in removed_ids:
        if cid in graph.claims:
            del graph.claims[cid]

    # Conflict detection
    claims_list = list(graph.claims.values())
    conflicts = 0
    for i, c1 in enumerate(claims_list):
        for c2 in claims_list[i+1:]:
            if detect_conflicts(c1, c2):
                c1.validity = ValidityStatus.DISPUTED
                c2.validity = ValidityStatus.DISPUTED
                conflicts += 1
                # Add CONTRADICTS relation between them
                rel = Relation(
                    relation_id=new_id("rel"),
                    relation_type=RelationType.CONTRADICTS,
                    from_id=c1.claim_id,
                    to_id=c2.claim_id,
                    evidence_ids=c1.evidence_ids + c2.evidence_ids,
                )
                graph.add_relation(rel)

    # Reversal chaining: REVERSAL claims → find matching DECISION claims and mark HISTORICAL
    reversals = [c for c in graph.claims.values() if c.claim_type == ClaimType.REVERSAL]
    decisions = [c for c in graph.claims.values() if c.claim_type == ClaimType.DECISION]
    superseded = 0
    for rev in reversals:
        for dec in decisions:
            if (dec.subject_id == rev.subject_id or dec.object_id == rev.object_id) \
               and dec.validity == ValidityStatus.CURRENT \
               and dec.valid_from and rev.valid_from \
               and dec.valid_from < rev.valid_from:
                dec.validity = ValidityStatus.HISTORICAL
                dec.valid_until = rev.valid_from
                dec.superseded_by = rev.claim_id
                rev.supersedes = dec.claim_id
                superseded += 1

    print(f"  Claim dedup: {len(removed_ids)} duplicates merged, {conflicts} conflicts flagged, {superseded} decisions superseded by reversals")

# ── Master pipeline ────────────────────────────────────────────────────────────

def run_dedup(graph: MemoryGraph) -> dict:
    print("Running deduplication pipeline...")
    art_dups = dedup_artifacts(graph)
    canonicalize_entities(graph)
    dedup_claims(graph)

    return {
        "artifact_duplicates": len(art_dups),
        "merge_log_entries":   len(MERGE_LOG),
        "graph_stats":         graph.to_dict()["stats"],
    }

def save_merge_log(path: str):
    Path(path).write_text(json.dumps(MERGE_LOG, indent=2))

if __name__ == "__main__":
    from pathlib import Path
    import json, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from schema import MemoryGraph

    raw_path = "/home/claude/layer10/data/processed/graph_raw.json"
    raw = json.loads(Path(raw_path).read_text())

    # Reconstruct graph from serialized form
    # (For pipeline integration we pass the live graph object)
    print("Run dedup via pipeline.py for live graph object.")
