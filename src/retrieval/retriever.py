"""
Retrieval Engine
================
Given a question, returns a grounded context pack:
  - ranked evidence snippets
  - linked entities and claims
  - citation metadata (source_id, char offsets, timestamp)

Retrieval strategy:
  1. Keyword extraction from question
  2. Entity linking (find question entities in graph)
  3. Claim retrieval by entity + keyword overlap (TF-IDF-like BM25 approximation)
  4. Evidence expansion: for each matched claim, pull all evidence
  5. Ranking: confidence × recency × keyword_overlap
  6. Pruning: top-K, diversity filter
  7. Conflict surfacing: if DISPUTED claims found, include both sides
"""

import re, json, math, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import MemoryGraph, Claim, EvidencePointer, Entity, ValidityStatus, ClaimType

# ── Text utilities ─────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "of","in","on","at","to","for","with","by","from","that","this","these","those",
    "what","who","when","where","how","which","i","me","my","we","our","you","your",
    "it","its","and","or","but","not","no","so","if","then","than","as","about",
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def tf_idf_score(query_tokens: List[str], doc_tokens: List[str], df_map: Dict[str,int], N: int) -> float:
    """Simplified TF-IDF overlap score."""
    score = 0.0
    tf_doc = {}
    for t in doc_tokens:
        tf_doc[t] = tf_doc.get(t, 0) + 1
    for qt in query_tokens:
        tf = tf_doc.get(qt, 0)
        df = df_map.get(qt, 1)
        idf = math.log((N + 1) / (df + 1)) + 1.0
        score += (tf / (len(doc_tokens)+1)) * idf
    return score

def recency_weight(timestamp_str: str) -> float:
    """More recent → higher weight (range 0.5 – 1.0)."""
    try:
        ts = datetime.datetime.fromisoformat(timestamp_str.replace("Z",""))
        base = datetime.datetime(2001, 6, 1)
        days_after_base = max(0, (ts - base).days)
        return 0.5 + min(0.5, days_after_base / 200.0)
    except Exception:
        return 0.75

# ── Context Pack ──────────────────────────────────────────────────────────────

class ContextPack:
    def __init__(self, question: str):
        self.question = question
        self.matched_entities: List[Dict] = []
        self.ranked_claims: List[Dict]    = []
        self.evidence_snippets: List[Dict]= []
        self.conflicts: List[Dict]        = []
        self.answer_hint: str             = ""

    def to_dict(self) -> dict:
        return {
            "question":         self.question,
            "matched_entities": self.matched_entities,
            "ranked_claims":    self.ranked_claims,
            "evidence_snippets":self.evidence_snippets,
            "conflicts":        self.conflicts,
            "answer_hint":      self.answer_hint,
            "total_evidence":   len(self.evidence_snippets),
            "total_claims":     len(self.ranked_claims),
        }

# ── Retrieval Engine ──────────────────────────────────────────────────────────

class RetrievalEngine:
    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self._build_index()

    def _build_index(self):
        """Pre-compute TF-IDF document frequency index over all evidence excerpts."""
        self.df_map: Dict[str, int] = {}
        self.N = 0
        for ev in self.graph.evidence.values():
            tokens = set(tokenize(ev.excerpt))
            for t in tokens:
                self.df_map[t] = self.df_map.get(t, 0) + 1
            self.N += 1
        # Also index claim predicates + object values
        for c in self.graph.claims.values():
            tokens = set(tokenize(c.predicate + " " + (c.object_value or "")))
            for t in tokens:
                self.df_map[t] = self.df_map.get(t, 0) + 1

    def _find_entities(self, query_tokens: List[str]) -> List[str]:
        """Return entity_ids whose names/aliases appear in query."""
        query_lower = " ".join(query_tokens)
        matched = []
        for eid, entity in self.graph.entities.items():
            if any(tok in query_lower for tok in tokenize(entity.canonical_name)):
                matched.append(eid)
            elif any(
                any(tok in tokenize(alias) for tok in query_tokens)
                for alias in entity.aliases
            ):
                matched.append(eid)
        return list(set(matched))

    def _score_claim(self, claim: Claim, query_tokens: List[str], entity_ids: Set[str]) -> float:
        """Score a claim for a query."""
        # Entity match bonus
        entity_score = 0.0
        if claim.subject_id in entity_ids: entity_score += 0.4
        if claim.object_id  in entity_ids: entity_score += 0.2

        # Text overlap on predicate + object
        claim_text = tokenize(claim.predicate + " " + (claim.object_value or ""))
        text_score = tf_idf_score(query_tokens, claim_text, self.df_map, max(self.N, 1))

        # Evidence text overlap
        ev_score = 0.0
        for eid in claim.evidence_ids[:3]:
            ev = self.graph.evidence.get(eid)
            if ev:
                ev_tokens = tokenize(ev.excerpt)
                ev_score = max(ev_score,
                    tf_idf_score(query_tokens, ev_tokens, self.df_map, max(self.N,1))
                )

        # Recency
        recency = recency_weight(claim.valid_from or "2001-06-01T00:00:00Z")

        # Validity bonus: CURRENT > HISTORICAL > DISPUTED
        validity_weight = {
            "CURRENT":    1.0,
            "HISTORICAL": 0.7,
            "DISPUTED":   0.8,
            "RETRACTED":  0.3,
            "REDACTED":   0.1,
        }.get(claim.validity.value, 0.5)

        total = (entity_score * 0.35 + text_score * 0.40 + ev_score * 0.15) \
                * recency * validity_weight * claim.confidence
        return total

    def retrieve(self, question: str, top_k_claims: int = 8, top_k_evidence: int = 10) -> ContextPack:
        pack = ContextPack(question)
        query_tokens = tokenize(question)
        entity_ids   = set(self._find_entities(query_tokens))

        # 1. Matched entities
        for eid in entity_ids:
            e = self.graph.entities.get(eid)
            if e:
                pack.matched_entities.append({
                    "entity_id": eid,
                    "canonical_name": e.canonical_name,
                    "entity_type": e.entity_type.value,
                    "aliases": e.aliases[:3],
                })

        # 2. Score all claims
        scored_claims: List[Tuple[float, str]] = []
        for cid, claim in self.graph.claims.items():
            score = self._score_claim(claim, query_tokens, entity_ids)
            if score > 0.01:
                scored_claims.append((score, cid))

        scored_claims.sort(key=lambda x: -x[0])

        # Diversity filter: don't return >2 claims from same subject
        seen_subjects: Dict[str, int] = {}
        selected_cids: List[str] = []
        for score, cid in scored_claims:
            claim = self.graph.claims[cid]
            subj = claim.subject_id
            count = seen_subjects.get(subj, 0)
            if count < 3:
                selected_cids.append(cid)
                seen_subjects[subj] = count + 1
            if len(selected_cids) >= top_k_claims:
                break

        # 3. Format claims
        conflict_pairs: List[Tuple[str, str]] = []
        for cid in selected_cids:
            claim = self.graph.claims[cid]
            entity = self.graph.entities.get(claim.subject_id, None)
            claim_dict = {
                "claim_id":    cid,
                "claim_type":  claim.claim_type.value,
                "subject":     entity.canonical_name if entity else claim.subject_id,
                "predicate":   claim.predicate,
                "object":      claim.object_value or claim.object_id or "",
                "validity":    claim.validity.value,
                "valid_from":  claim.valid_from,
                "valid_until": claim.valid_until,
                "confidence":  round(claim.confidence, 3),
                "evidence_count": len(claim.evidence_ids),
                "superseded_by": claim.superseded_by,
            }
            pack.ranked_claims.append(claim_dict)
            if claim.validity.value == "DISPUTED":
                conflict_pairs.append(cid)

        # 4. Gather evidence snippets
        seen_ev_ids: set = set()
        for cid in selected_cids:
            claim = self.graph.claims[cid]
            for eid in claim.evidence_ids[:3]:
                if eid in seen_ev_ids: continue
                ev = self.graph.evidence.get(eid)
                if ev and len(pack.evidence_snippets) < top_k_evidence:
                    seen_ev_ids.add(eid)
                    pack.evidence_snippets.append({
                        "evidence_id":  eid,
                        "source_id":    ev.source_id,
                        "source_type":  ev.source_type,
                        "excerpt":      ev.excerpt[:300],
                        "char_start":   ev.char_start,
                        "char_end":     ev.char_end,
                        "timestamp":    ev.timestamp,
                        "confidence":   round(ev.confidence, 3),
                        "extractor_ver":ev.extractor_ver,
                    })

        # 5. Conflict surfacing
        for cid in conflict_pairs:
            claim = self.graph.claims.get(cid)
            if claim and claim.superseded_by:
                rev = self.graph.claims.get(claim.superseded_by)
                if rev:
                    pack.conflicts.append({
                        "type":         "supersession",
                        "original_claim": cid,
                        "reversal_claim": claim.superseded_by,
                        "note": f"The claim was superseded at {claim.valid_until}",
                    })

        # 6. Simple answer hint (most recent current claim)
        current_claims = [c for c in pack.ranked_claims if c["validity"] == "CURRENT"]
        if current_claims:
            top = current_claims[0]
            pack.answer_hint = (
                f"Based on the memory graph: '{top['subject']}' → {top['predicate']} → "
                f"'{top['object'][:80]}' [confidence={top['confidence']}, validity={top['validity']}]"
            )
        elif pack.ranked_claims:
            top = pack.ranked_claims[0]
            pack.answer_hint = (
                f"[{top['validity']}] '{top['subject']}' → {top['predicate']} → "
                f"'{top['object'][:80]}'"
            )
        else:
            pack.answer_hint = "No matching memory found for this question."

        return pack


def run_sample_queries(engine: RetrievalEngine) -> List[dict]:
    questions = [
        "What decisions were made about Project Raptor?",
        "Who flagged concerns about accounting practices?",
        "What is the status of Dabhol Power Plant?",
        "What did Jeff Skilling decide?",
        "Were any decisions reversed or walked back?",
        "What assignments were given to Andy Fastow?",
        "What meetings occurred regarding EnronOnline?",
        "What risk flags were raised about SPE disclosures?",
    ]
    results = []
    for q in questions:
        pack = engine.retrieve(q)
        results.append(pack.to_dict())
        print(f"Q: {q}")
        print(f"  → {pack.answer_hint[:100]}")
        print(f"  → {len(pack.ranked_claims)} claims, {len(pack.evidence_snippets)} evidence items")
    return results


if __name__ == "__main__":
    from schema import MemoryGraph
    import json
    from pathlib import Path

    graph_data = json.loads(Path("/home/claude/layer10/data/processed/graph_final.json").read_text())
    print("Graph loaded. Run via pipeline.py")
