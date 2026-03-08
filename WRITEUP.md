# Technical Write-Up
## Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

---

## 1. Corpus Selection

**Dataset:** Enron Email Dataset, Kaggle mirror of the CMU/FERC public release.
**URL:** https://www.kaggle.com/datasets/wcukierski/enron-email-dataset?resource=download
**Reproduce:** `kaggle datasets download -d wcukierski/enron-email-dataset && unzip enron-email-dataset.zip`
**Format:** Single CSV (`emails.csv`) with columns `file` (mailbox path) and `message` (full RFC 2822 email including all headers and body).

**Why this corpus is the right choice for Layer10:**

The Enron dataset is not just a convenient public email dump — it is the most realistic approximation of what Layer10 ingests in production. It contains:

- **Identity chaos.** The same person appears as `phillip.allen@enron.com`, `Phillip K Allen`, `pallen70@hotmail.com`, `=ENRON/OU=NA/CN=RECIPIENTS/CN=PALLEN>` (LDAP/Exchange format), and `Allen, Phillip K. </O=ENRON/OU=NA/CN=...>`. No two messages are guaranteed to use the same identifier for the same person.
- **Massive duplication.** Of 4,850 parsed messages in our sample, **3,062 (63%) are duplicates** — forwarded chains, cross-posts, reply-quoting. A memory system that does not deduplicate will store the same fact thousands of times.
- **Real decisions and reversals.** Executives explicitly walk back earlier positions in writing. Assignments change owners. Projects are approved then cancelled. This is exactly the "correctness over time" problem Layer10 is built to solve.
- **Long time horizon.** Emails span 1998–2002. A memory system must distinguish "decided in 2000, reversed in 2001" from "decided in 2000, still current."
- **Unstructured + semi-structured mix.** Meeting notes, forwarded memos, one-line replies, and multi-page status reports — the full range of organizational communication.

**What was run:** `--limit 5000` (4,850 messages after filtering), producing a graph with 130 entities, 221 claims, 494 evidence pointers, and 5,434 relations from 20,542 merge operations.

---

## 2. Ontology and Schema Design

The schema (`src/schema.py`) is designed around a single principle: **every piece of memory must be traceable to its source, and every source must have a timestamp and a validity state.** This is what separates a memory system from a search index.

### 2.1 Entity Types

| Type | Description | Real Enron example |
|------|-------------|-------------------|
| `PERSON` | Named individual with canonical name + alias set | `Phillip K Allen` with aliases `phillip.allen@enron.com`, `pallen70@hotmail.com`, LDAP string |
| `PROJECT` | Named initiative, product, or topic cluster | `Gas Price Caps`, `Enron Response`, `Investment Structure` |
| `ORGANIZATION` | Company, division, or team | `Enron`, `Arthur Andersen`, `FERC` |
| `CONCEPT` | Abstract domain concept | `mark-to-market`, `SPE disclosure` |
| `ARTIFACT` | A source document or message | `allen-p/_sent_mail/42.` |

### 2.2 Claim Types

Claims are the atomic unit of memory. Each claim is a typed, timestamped, grounded assertion about the world.

| Type | Semantics | Signal in corpus |
|------|-----------|-----------------|
| `DECISION` | An agent decided/approved/proceeded with something | "decided", "approved", "agreed", "will proceed" |
| `REVERSAL` | An earlier decision was explicitly walked back | "walk back", "reversing", "no longer", "pausing" |
| `STATUS` | State of a project or task at a point in time | "on track", "completed", "behind schedule", "status:" |
| `RISK_FLAG` | A concern or compliance risk was raised | "concern", "risk", "flag", "issue", "warning" |
| `ASSIGNMENT` | A person was assigned ownership of a task | "Action:", "Owner:", "assigned to", "you will" |
| `MEETING` | A meeting occurred with attendees and outcomes | "Attendees:", "agenda", "meeting notes", "minutes" |
| `FACT` | Any other extractable atomic fact | Fallback for messages not matching the above |

### 2.3 Relation Types

Relations connect entities to each other and to artifacts, forming the graph structure.

| Type | Connects | Meaning |
|------|----------|---------|
| `AUTHORED` | PERSON → ARTIFACT | Person sent this message |
| `MENTIONED_IN` | ENTITY → ARTIFACT | Entity is referenced in this message |
| `PARTICIPATES_IN` | PERSON → PROJECT | Person is involved in this project |
| `CONTRADICTS` | CLAIM → CLAIM | Two claims assert conflicting values for the same predicate |
| `SUPERSEDES` | CLAIM → CLAIM | This reversal claim supersedes an earlier decision claim |
| `ALIASES` | ENTITY → ENTITY | Two entity records refer to the same real-world entity |

### 2.4 The Evidence Pointer

Every claim must carry at least one `EvidencePointer`. This is non-negotiable and enforced by the quality gate. The pointer contains everything needed to trace a claim back to its raw source:

```json
{
  "evidence_id":   "ev_3a7f2b19c0",
  "source_id":     "allen-p/_sent_mail/42.",
  "source_type":   "email",
  "excerpt":       "I want to raise a concern about the current risk exposure on the West desk.",
  "char_start":    214,
  "char_end":      289,
  "timestamp":     "2001-03-19T09:00:00Z",
  "ingested_at":   "2026-01-01T00:00:00Z",
  "confidence":    0.85,
  "extractor_ver": "v1.0.0-deterministic",
  "schema_ver":    "2026-01-ontology-v1"
}
```

`extractor_ver` and `schema_ver` together enable safe backfill: when the ontology changes, a migration job can filter all evidence by `schema_ver` and re-extract, then diff the new claims against the existing graph.

### 2.5 Validity States

Claims do not have a boolean "true/false" — they have a validity state that changes over time:

```
CURRENT    — currently believed to be true
HISTORICAL — was true, has been superseded by a later claim
DISPUTED   — conflicting evidence exists; both sides are preserved
RETRACTED  — explicitly retracted by the original author
REDACTED   — source was deleted or subject to GDPR erasure
```

The `valid_from` / `valid_until` pair models **bitemporal semantics**: `valid_from` is the event time (when the decision was made), `valid_until` is when we learned it was no longer true. A query for "what was the status of project X on July 15, 2001" filters on `valid_from <= 2001-07-15 AND (valid_until IS NULL OR valid_until > 2001-07-15)`.

---

## 3. Extraction Pipeline

**Design philosophy:** Extraction is a deterministic, reproducible, regression-testable system — not a prompt. This is a deliberate tradeoff: lower recall than an LLM, but every run on the same input produces identical output, every claim can be explained, and the system can be backfilled safely when the ontology changes.

### 3.1 Corpus Loading (`src/corpus_loader.py`)

The loader parses each row of `emails.csv` using Python's `email` stdlib module (RFC 2822 compliant):

1. **Header extraction:** `From`, `To` (multi-recipient, comma-separated, newline-folded), `Subject`, `Date` → ISO8601, `Message-ID`, `X-From` (Exchange display name), `X-Folder`
2. **Body extraction:** Handles both `text/plain` payloads and multipart messages. Decodes with `utf-8, errors=replace` to handle the corpus's mixed encodings.
3. **Alias resolution:** Groups `phillip.allen@enron.com` + `"Phillip K Allen"` (from X-From) into a single PERSON entity with both as aliases.
4. **Message classification:** Keyword scan of subject + first 500 chars of body → one of {decision, reversal, status, concern, assignment, meeting\_notes, general}.
5. **Project/topic extraction:** Recurring capitalized multi-word phrases from subjects (≥3 occurrences) become PROJECT entities. This surfaces real organizational topics like "Gas Price Caps", "Enron Response", "Investment Structure" without any manual curation.
6. **Quality filter:** Messages with fewer than 5 body words are skipped. Of 5,000 attempted, 4,850 passed (150 skipped).

### 3.2 Entity Bootstrapping

Before extraction runs on any message, all known persons and projects are registered in the `MemoryGraph` with their full alias sets. This means entity resolution during extraction is a fast dictionary lookup, not a repeated search.

### 3.3 Extraction Loop (`src/extraction/extractor.py`)

For each message, the extractor:

1. Creates an `Artifact` record with `content_hash = SHA-256(subject + body)` for downstream dedup
2. Resolves the sender address to a canonical `entity_id` via the alias index
3. Runs typed pattern banks against the message body:
   - `DECISION_PATTERNS` — 2 regex patterns covering "decided/approved/confirmed to [action]"
   - `REVERSAL_PATTERNS` — 2 patterns covering "walk back/reverse/pause/no longer [action]"
   - `STATUS_PATTERNS` — 2 patterns covering "on track/completed/behind schedule [context]"
   - `RISK_PATTERNS` — 2 patterns covering "concern/risk/flag about [topic]"
   - `ASSIGNMENT_PATTERNS` — 3 patterns covering "Action: [person] to [task]", "Owner: [person]"
4. For each pattern match: extract the matched span, compute `char_start`/`char_end` against the raw body, build an `EvidencePointer`, build a typed `Claim`
5. Apply the quality gate
6. Add passing claims, evidence, and relations to the graph

### 3.4 Confidence Scoring

```
base = 0.60
+ 0.10  if excerpt > 50 chars
+ 0.10  if excerpt > 100 chars
+ 0.15  if message type matches expected claim type (e.g. "reversal" message → REVERSAL claim)
+ 0.10  additional for REVERSAL claims (explicit reversals are high-signal)
cap at 0.95
```

### 3.5 Quality Gate

A claim is rejected if any of the following are true:
- `evidence_ids` is empty
- All evidence pointers have `confidence < 0.55`
- Subject is `"unk"` AND `object_id` is None AND `object_value` is empty

In this run: 494 claims extracted, 0 rejected. The patterns are conservative enough that matches are high-confidence by construction.

### 3.6 Versioning and Backfill

Every `EvidencePointer` carries:
- `extractor_ver = "v1.0.0-deterministic"` — identifies the pattern bank version
- `schema_ver = "2026-01-ontology-v1"` — identifies the ontology version

When the ontology changes (e.g. adding a new claim type, refining a pattern), the backfill procedure is:
1. Filter all evidence where `schema_ver < target_ver`
2. Re-run extraction with updated patterns on those source messages
3. Diff new claims against existing ones by fingerprint
4. Promote new claims, demote superseded ones, add migration note to merge log
5. Bump `schema_ver` on all re-processed evidence

This means ontology evolution never requires full re-ingestion — only affected messages are reprocessed.

---

## 4. Deduplication and Canonicalization

A memory system that does not deduplicate aggressively will degrade into noise. In the Enron corpus, 63% of messages are duplicates of earlier messages — the result of forwarding chains, CC storms, and cross-posts. Without dedup, the same claim would appear 3,062 times with identical evidence.

The deduplication pipeline (`src/dedup/deduplicator.py`) operates at three levels.

### 4.1 Level 1 — Artifact Deduplication

**Exact dedup:** SHA-256 hash of normalized `(subject + body)`. If two artifacts share a hash, the later one is marked `is_duplicate=True` with `duplicate_of` pointing to the canonical artifact. The canonical artifact retains all evidence that would have been attributed to duplicates.

**Near-dedup:** Character trigram Jaccard similarity on subject lines, threshold 0.85. This catches same-subject threads from different senders where the body has minor variations (added signature, whitespace differences). When a near-duplicate is detected, the earlier timestamp wins as canonical.

**Result:** 3,062 artifact duplicates detected and resolved in this run.

### 4.2 Level 2 — Entity Canonicalization

An alias index maps every `canonical_name` and every `alias` (normalized, lowercased) to a single `entity_id`. Entities that share an alias are merged via `merge_entities()`:

1. All aliases of the removed entity are appended to the kept entity
2. All `subject_id` and `object_id` references in claims are re-pointed to the kept entity
3. All `from_id` and `to_id` references in relations are re-pointed
4. The removed entity's ID is appended to `merged_from` on the kept entity (audit trail)
5. The removed entity is deleted from the graph
6. A merge log entry is written with `merged_ids`, `into_id`, `reason`, and timestamp

**Reversibility:** Any merge can be undone by reading the merge log in reverse: re-create the removed entity from the log, restore aliases, re-point back-references.

**Known limitation:** LDAP-formatted Exchange addresses (`=ENRON/OU=NA/CN=RECIPIENTS/CN=PALLEN>`) do not share a normalized alias with `phillip.allen@enron.com` unless the X-From header is present. Full resolution requires a corporate directory lookup. This is intentionally left as a documented gap — the current run shows 0 entity merges because most senders appear with consistent addresses within the 5,000-message sample. At full corpus scale with directory lookup, this number would be significant.

### 4.3 Level 3 — Claim Deduplication

**Fingerprint:** `SHA-256(subject_id | predicate_normalized[:50] | object_value[:30] | claim_type)`. Two claims with identical fingerprints are merged: their evidence lists are unioned, the maximum confidence is kept, and the duplicate is deleted.

**Conflict detection:** Two claims on the same `(subject_id, claim_type)` with different `object_value` strings, where both are `CURRENT`, are marked `DISPUTED`. A `CONTRADICTS` relation is added between them. This preserves both sides rather than silently resolving the conflict — the retrieval engine surfaces conflicts explicitly.

**Reversal chaining:** Every `REVERSAL` claim is matched against all `DECISION` claims on the same subject or object. If the decision's `valid_from` precedes the reversal's `valid_from`, the decision is updated:
- `validity` → `HISTORICAL`
- `valid_until` → reversal's `valid_from`
- `superseded_by` → reversal's `claim_id`

And the reversal is updated:
- `supersedes` → decision's `claim_id`

**Result in this run:** 273 claim duplicates merged, 71 conflicts flagged as DISPUTED, 45 decisions superseded by reversals, 20,542 merge log entries total.

---

## 5. Memory Graph Design

### 5.1 Storage Model

The `MemoryGraph` class holds five dictionaries keyed by ID: `entities`, `claims`, `relations`, `artifacts`, `evidence`. For this exercise, the graph is in-memory Python serialized to JSON. This is sufficient for the scale of this submission and makes the output trivially inspectable.

**Production mapping:**

The natural production target is **Postgres + pgvector**:
- `entities`, `claims`, `artifacts`, `evidence` as standard relational tables
- `relations` as an adjacency table (enables graph traversal with recursive CTEs)
- `claim_embeddings` column in `pgvector` for semantic retrieval
- JSONB `attributes` columns for schema flexibility without migrations
- `valid_from` / `valid_until` indexed for bitemporal range queries

For teams that need deep multi-hop graph traversal, **Neo4j** with APOC procedures handles bitemporal queries well and supports subgraph expansion natively. The tradeoff is operational complexity versus query expressiveness.

### 5.2 Time Modeling

Three timestamps are tracked per claim/evidence:

| Timestamp | Field | Meaning |
|-----------|-------|---------|
| Event time | `claim.valid_from` | When the decision/fact occurred (from message Date header) |
| Validity end | `claim.valid_until` | When this claim was superseded (set by reversal chaining) |
| Ingestion time | `evidence.ingested_at` | When the pipeline processed this source |

"Current" is defined as: `validity == CURRENT AND valid_until IS NULL`.

Point-in-time query: `valid_from <= D AND (valid_until IS NULL OR valid_until > D)`.

### 5.3 Incremental Updates and Idempotency

New message arrives:
1. Compute `content_hash` — if already in `artifacts`, skip entirely (idempotent)
2. Run extraction — new claims get new `claim_id`s
3. Run claim dedup fingerprint against existing graph — merge or flag conflict
4. Run reversal detection against existing decisions

Edit arrives (e.g. Slack message edited):
1. Look up artifact by `source_id`
2. Update `content_hash` and body
3. Re-run extraction on updated body
4. Diff new claims against old claims — retire removed ones as `HISTORICAL`, add new ones

Delete/redaction arrives:
1. Set `artifact.is_redacted = True`
2. For all claims with this as their only evidence source: set `validity = REDACTED`
3. For claims with multiple evidence sources: remove this evidence pointer, re-evaluate confidence
4. For GDPR-style person erasure: zero out all `excerpt` fields in evidence pointers for that person's messages, replace with `[REDACTED]`; the claim structure is preserved (we know a decision was made) but the exact text is gone

### 5.4 Permissions (Conceptual)

Every `EvidencePointer` carries a `source_id`. The permission layer maintains a `source_acl` table mapping `(source_id, user_id) → allowed`. At retrieval time:

```sql
SELECT c.* FROM claims c
WHERE EXISTS (
  SELECT 1 FROM evidence e
  JOIN source_acl acl ON acl.source_id = e.source_id
  WHERE e.evidence_id = ANY(c.evidence_ids)
    AND acl.user_id = $requesting_user
)
```

This ensures a claim is only returned if the requesting user can access at least one of its evidence sources. A claim grounded only in a confidential email chain is invisible to users without access to that chain — even if the logical claim (e.g. "project X was approved") is the same fact.

### 5.5 Observability

Key metrics to track in production:

| Metric | What it signals |
|--------|----------------|
| Claim confidence distribution (histogram, by extractor\_ver) | Drift in extraction quality |
| Duplicate rate (artifact dups / total artifacts) | Upstream connector de-sync |
| Conflict rate (DISPUTED / total claims) | Ontology drift or noisy patterns |
| Evidence pointer staleness (ingested\_at lag behind source timestamp) | Pipeline backlog |
| Reversal latency (time from decision ingested to reversal chained) | Real-time memory correctness |
| Entity merge rate | Identity resolution coverage |

---

## 6. Retrieval and Grounding

### 6.1 Architecture (`src/retrieval/retriever.py`)

The retrieval engine answers natural-language questions with **grounded context packs** — structured objects where every returned item traces back to a specific source, not a model-generated summary.

**Step 1 — Keyword extraction:**
Tokenize the question, remove stopwords, extract content tokens.

**Step 2 — Entity linking:**
Match question tokens against entity `canonical_name` and all aliases. Returns a set of matched `entity_id`s.

**Step 3 — Claim scoring:**
Every claim is scored against the query:

```
score = (
    entity_match_score × 0.35    # subject/object entity in query
  + tfidf_predicate_score × 0.40  # TF-IDF overlap on predicate + object text
  + tfidf_evidence_score × 0.15   # TF-IDF overlap on evidence excerpts
) × recency_weight(valid_from)    # more recent = higher weight (0.5–1.0)
  × validity_weight[validity]      # CURRENT=1.0, HISTORICAL=0.7, DISPUTED=0.8
  × claim.confidence
```

TF-IDF uses a pre-built document frequency index over all evidence excerpts and claim predicates, computed once at engine initialization.

**Step 4 — Diversity pruning:**
Cap at 3 claims per subject entity to prevent any single person dominating the context pack.

**Step 5 — Evidence expansion:**
For each selected claim, pull all `EvidencePointer` records. Deduplicate by `evidence_id`.

**Step 6 — Conflict surfacing:**
If DISPUTED claims or superseded decisions are in the result set, they are annotated explicitly in `conflicts[]` — the consumer sees both sides and can decide how to present them, rather than receiving a silently resolved answer.

### 6.2 Context Pack Format

```json
{
  "question": "Who flagged concerns about accounting practices?",
  "matched_entities": [
    {"entity_id": "ent_person_p0003", "canonical_name": "Phillip K Allen",
     "entity_type": "PERSON", "aliases": ["phillip.allen@enron.com", ...]}
  ],
  "ranked_claims": [
    {
      "claim_id": "clm_a3f7...",
      "claim_type": "RISK_FLAG",
      "subject": "Phillip K Allen",
      "predicate": "flagged_risk",
      "object": "issue to Sally Beck.",
      "validity": "CURRENT",
      "valid_from": "2001-02-05T00:00:00Z",
      "confidence": 0.75,
      "evidence_count": 3
    }
  ],
  "evidence_snippets": [
    {
      "evidence_id": "ev_3a7f...",
      "source_id": "allen-p/_sent_mail/42.",
      "source_type": "email",
      "excerpt": "issue to Sally Beck.",
      "char_start": 48,
      "char_end": 68,
      "timestamp": "2001-02-05T00:00:00Z",
      "confidence": 0.75,
      "extractor_ver": "v1.0.0-deterministic"
    }
  ],
  "conflicts": [],
  "answer_hint": "Based on the memory graph: 'Phillip K Allen' → flagged_risk → 'issue to Sally Beck.' [confidence=0.75, validity=CURRENT]"
}
```

Every item in `ranked_claims` has a non-empty `evidence_ids` list. The retrieval engine will not return a claim without evidence. This is enforced structurally, not by convention.

### 6.3 Ambiguity and Conflicts

When the matched entity set is empty, the engine falls back to pure TF-IDF over all evidence excerpts. When DISPUTED claims appear in results, both sides are included with their respective evidence, and a `conflicts` entry describes the nature of the contradiction. The consumer (whether a human in the UI or a downstream LLM) receives all available information and makes its own determination — the memory system does not silently resolve conflicts.

---

## 7. Visualization Layer

**File:** `viz/index.html` — fully self-contained. After the pipeline runs, graph data is baked directly into the HTML file. Open it in any browser without a server.

**Technical approach:** At the end of `pipeline.py`, the HTML template (which contains a `// __INJECTED_DATA__` placeholder) is read, the placeholder is replaced with the serialized graph JSON and merge log JSON, and the modified HTML is written back. The JavaScript reads from in-memory variables, not from a network fetch.

**Graph rendering:** Force-directed physics simulation (spring forces on edges, repulsion between nodes, center gravity) computed in Canvas2D. Person nodes on outer ring, project nodes on inner ring at initialization. Converges to stable layout in ~200 iterations.

**Features implemented:**

The **entity panel** (left) lists all 130 entities filterable by type and searchable by name or alias. Selecting an entity highlights it on the graph, highlights its edges, and loads its data in the evidence panel.

The **claims tab** (right) shows all claims for the selected entity sorted DISPUTED-first, then by confidence. Each claim card displays the type badge (colored by claim type), validity badge (colored by state), predicate, object excerpt, timestamps, confidence percentage, and evidence count. Clicking any claim drills into its evidence.

The **evidence tab** shows all evidence pointers across the entity's claims with full provenance: exact excerpt, source message ID, character offsets `[start:end]`, event timestamp, confidence score, and extractor version stamp. This lets a reviewer trace any claim to the exact bytes in the original email.

The **merges tab** shows the complete merge audit log — 20,542 entries in this run. Each entry shows the merge type (artifact\_exact, artifact\_near\_dup, claim), the merged IDs, the canonical ID, and the reason.

The **evidence drill-down** replaces the panel content with all evidence for a single selected claim. Superseded decisions show a red reversal banner linking to the superseding claim ID.

The **query bar** implements the retrieval engine in the browser: entity linking + TF-IDF scoring + ranking + conflict surfacing, running entirely client-side against the embedded graph data.

---

## 8. Adapting to Layer10's Production Environment

### 8.1 Ontology Changes for Email + Slack + Jira/Linear

The current ontology works well for email. Extending to Layer10's full target environment requires these additions:

| Addition | Rationale |
|----------|-----------|
| `THREAD` entity type | Slack threads and email threads are first-class memory objects, not just containers |
| `TICKET` entity type | Jira/Linear issues have structured state, assignees, labels, and explicit transitions |
| `CHANNEL` entity type | Slack channels define access scope — relevant for permissions |
| `STATUS_TRANSITION` claim type | `ticket.status: Open → In Progress → Done` — each transition is a CURRENT claim; the predecessor is auto-HISTORICAL |
| `MENTION` relation | `@user` mentions in Slack and email create explicit links in the graph |
| `LINKED_TO` relation | Jira issue ↔ Slack thread ↔ email chain — the cross-system join |
| `REACTION` claim type | Slack reactions (👍, ✅) are lightweight endorsements worth capturing as low-confidence claims |

For Jira/Linear specifically: structured fields (assignee, status, priority, labels) are extracted deterministically from the structured event payload — no pattern matching needed. Comment bodies are processed identically to email bodies through the same extraction pipeline. Webhook events (`issue.updated`, `issue.transitioned`) give exact edit times, eliminating the ambiguity present in email.

### 8.2 Extraction Contract Changes

**Email (enhanced from current):**
- Parse `In-Reply-To` and `References` headers for true thread reconstruction. Currently, threads are grouped by `Message-ID` which misses reply chains where the threading headers are absent.
- Detect quoted text (lines starting with `>`) and treat it as a separate evidence tier: lower confidence, attributed to the original author, not the forwarder.
- Handle multipart MIME with HTML stripping for HTML-only emails.

**Slack:**
- Use `thread_ts` for threading — Slack's threading model is explicit and reliable.
- User IDs (`U0123ABC`) are stable canonical identifiers. No alias resolution needed for senders.
- Handle message edits: `message.changed` events produce a new version with `edited.ts`. The retrieval engine should prefer the latest version; older versions are retained as HISTORICAL evidence.
- Reactions without text are FACT claims with `confidence = 0.3` — useful as corroboration signals but not primary evidence.

**Jira/Linear:**
- Status transitions are `STATUS_TRANSITION` claims: `(ticket_id, "transitioned_to", new_status, ts)`. Previous status claims auto-transition to HISTORICAL.
- Assignment changes are `ASSIGNMENT` claims with explicit supersession chaining.
- Comment bodies go through the full extraction pipeline (same patterns as email).
- Labels and components become `PARTICIPATES_IN` relations between tickets and concept entities.

### 8.3 Unstructured + Structured Fusion

The core join problem for Layer10 is: *"a decision was made in Slack, referenced in a Jira ticket, confirmed in email — how do you know these are about the same thing?"*

The strategy:
1. Extract ticket/issue mentions from message bodies (`PROJ-123`, `#issue-title`, `jira.company.com/browse/PROJ-123`)
2. Create `MENTIONED_IN` relations: `ticket_entity → message_artifact`
3. Match Slack channel names to Jira project keys (e.g. `#backend` → `BACKEND-*`)
4. At retrieval time, expand from ticket → mentioned\_in messages → their claims, surfacing the full discussion context alongside the structured ticket state

This is the difference between "the ticket says it's done" and "the ticket says it's done, but there's a risk flag in Slack from three days ago that wasn't resolved."

### 8.4 Long-Term Memory: Durable vs. Ephemeral

Not everything should become durable memory. The distinction:

**Durable memory (persisted to graph):**
- Decisions with named owners and explicit commitment language
- Status transitions on tracked artifacts (tickets, projects)
- Risk flags from named individuals with specific targets
- Assignments with deadlines or owners
- Meeting outcomes with committed next steps
- Any claim corroborated by ≥2 independent evidence sources

**Ephemeral context (used only in retrieval window, not persisted):**
- Conversational acknowledgements ("thanks", "sounds good", "will do")
- Status updates immediately superseded by a same-day update
- Duplicate content from forwarding chains (after dedup, these contribute nothing new)
- Emoji reactions in isolation

**Drift prevention:**
- `STATUS` claims carry a confidence decay: a status claim older than 30 days without corroboration is downgraded to `confidence × 0.7` per month. Status is inherently perishable.
- `DECISION` claims do not decay — a decision remains HISTORICAL indefinitely. It is not erased; it is just no longer CURRENT.
- Periodic full re-extraction (weekly) detects schema drift by comparing fingerprint collision rates between the new and old extractions. Rising collision rates signal that the ontology has drifted relative to what the corpus actually contains.

### 8.5 Grounding and Safety

**Provenance chain:** Every claim is traceable to `source_id` → raw message byte range (`char_start:char_end`) → author entity → event timestamp. There is no claim in the graph that cannot be fully sourced.

**Hallucination prevention:** The retrieval engine structurally cannot return a claim without `evidence_ids`. Downstream LLMs consuming context packs are instructed (via system prompt) to only make assertions about claims where `evidence_ids` is non-empty and to cite the `source_id` when doing so.

**Deletions and redactions:**
- Source deletion cascades to `is_redacted=True` on the artifact. Claims grounded only in that artifact move to `REDACTED` validity. Claims with multiple evidence pointers lose one pointer and are re-evaluated.
- GDPR right-to-erasure for a person: zero out all `excerpt` fields in evidence pointers for that person's messages, replace with `[REDACTED per request #ID]`. The claim structure is preserved — we know a decision was made, by whom, and when — but the exact wording is gone.
- Retention policies: evidence pointers older than the retention window can be nulled out while preserving the claim. The graph remembers that something happened without retaining the exact source text.

### 8.6 Operational Reality

**Scaling extraction:**
The extraction pipeline is embarrassingly parallel per message. At Layer10's scale:
- Fan out as Celery tasks or AWS Lambda, one task per message batch (50–100 messages)
- Dedup runs per-batch (exact hash is instant) and then again across batches (near-dedup requires cross-batch comparison, run as a nightly job)
- Conflict detection is O(N²) worst case — mitigated by indexing on `(subject_id, claim_type)` and sharding by entity

**Cost:**
The deterministic extractor has zero inference cost. Adding an LLM layer (for higher-recall extraction on important messages) would run at ~$0.002 per message with batch API at `claude-haiku` pricing — approximately $1,000 for the full Enron corpus, or a few dollars per day for a mid-size organization's email volume.

**Incremental updates:**
- Webhook-driven ingestion: Slack/Jira events trigger extraction within 30 seconds
- Email polling: IMAP IDLE or Microsoft Graph subscription for near-real-time new messages
- Full re-ingestion: weekly, to catch missed events and run cross-batch dedup

**Evaluation and regression testing:**
- Golden set: 50 manually verified `(question, expected_claim_id, expected_evidence_id)` triples
- Retrieval MRR@5 and evidence hit rate tracked per `extractor_ver`
- Dedup precision/recall on a labelled duplicate set (100 confirmed pairs)
- Shadow mode: new extractor version runs alongside production, claim overlap and conflict rates compared before promotion
- Claim confidence calibration: monthly, compare confidence scores against human-labelled ground truth

---

## 9. Tradeoffs and What I Would Do Differently at Scale

These are deliberate choices for this exercise, not oversights.

**Rule-based extraction vs. LLM extraction.** The regex extractor is deterministic, zero-cost, and regression-testable. It has high precision and lower recall — it will miss implicit claims like "we should probably hold off on this for now" (a soft reversal). At production scale, the right architecture is: LLM structured output extraction as the primary path, with the rule-based system as the validation layer that catches schema violations and malformed outputs. The LLM's JSON output is validated against the ontology schema before any claim is written to the graph.

**TF-IDF retrieval vs. dense retrieval.** TF-IDF is fast, interpretable, and needs no model. It misses semantic synonymy ("walked back" ≠ "reversed" ≠ "cancelled" to a keyword matcher). Production adds `text-embedding-3-small` embeddings on evidence excerpts stored in `pgvector`, with hybrid BM25 + cosine retrieval fused via Reciprocal Rank Fusion. The TF-IDF layer remains as the fast first-pass filter.

**Identity resolution.** The current system handles the cases where X-From display names are present. It does not handle Exchange LDAP strings as aliases, does not do fuzzy name matching (e.g. "Tim" = "Timothy" = "T. Belden"), and does not use the email body's signature block as an alias source. Production adds: LDAP/Exchange address normalization, a name fuzzy-match step using edit distance, and signature block parsing.

**In-memory graph.** Works correctly and is fully auditable for this scale. The production path is Postgres for relational queries + pgvector for semantic search + Apache AGE (or Neo4j) for multi-hop graph traversal. The schema maps directly — each Python dataclass becomes a table, each Dict becomes a row.

**Thread reconstruction.** Currently, `Message-ID` is used as the thread key. Real email threading requires parsing `In-Reply-To` and `References` headers to reconstruct the actual reply chain. The current approach works for single-message threads but misses the relational structure within a thread.

---

## 10. Summary

This submission implements a complete, working pipeline for grounded long-term organizational memory:

- **4,850 real Enron emails** parsed from the canonical public corpus
- **494 claims** extracted with full evidence pointers (source, excerpt, char offsets, timestamp, confidence, version)
- **3,062 artifact duplicates** caught and resolved
- **71 conflicts** preserved as DISPUTED rather than silently resolved
- **45 decisions** automatically superseded by reversal chains
- **20,542 merge operations** all reversible via the audit log
- **Bitemporal validity** — every claim knows when it was true and when it stopped being true
- **Zero external dependencies** — runs anywhere Python 3.10+ is installed
- **Interactive visualization** — self-contained HTML, open directly in any browser

The system is designed to be honest about what it does not yet do: full identity resolution for Exchange addresses, LLM-powered high-recall extraction, semantic retrieval, and production-grade graph storage. Each limitation is documented with a concrete production path.

---

*Pipeline: 4,850 messages → 494 claims → 221 post-dedup claims → 71 conflicts → 45 reversals → 20,542 merge log entries → fully grounded memory graph with bitemporal validity.*
