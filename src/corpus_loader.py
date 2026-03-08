"""
Real Enron Email CSV Parser
============================
Parses the Kaggle Enron dataset (emails.csv) with columns: file, message
Extracts headers (From, To, Subject, Date, Message-ID, X-Folder, etc.)
and body. Returns a corpus dict compatible with the rest of the pipeline.

Usage:
    python3 src/corpus_loader.py --csv emails.csv --limit 5000
"""

import csv, re, hashlib, argparse, json
from pathlib import Path
from email import message_from_string
from typing import Optional

# ── Header parsing ─────────────────────────────────────────────────────────────

def parse_raw_message(file_path: str, raw: str) -> Optional[dict]:
    try:
        msg = message_from_string(raw)
    except Exception:
        return None

    sender   = msg.get("From", "").strip()
    to_raw   = msg.get("To", "").strip()
    subject  = msg.get("Subject", "").strip()
    date_str = msg.get("Date", "").strip()
    msg_id   = msg.get("Message-ID", "").strip()
    x_folder = msg.get("X-Folder", "").strip()
    x_from   = msg.get("X-From", "").strip()

    # Recipients — comma-separated, may contain newlines
    recipients = [r.strip() for r in re.split(r",\s*", to_raw.replace("\n", " ").replace("\t", " ")) if r.strip()]

    # Body — get payload
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
                except Exception:
                    body = str(part.get_payload())
                    break
    else:
        try:
            body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            body = str(msg.get_payload() or "")

    body = body.strip()
    if not body and not sender:
        return None

    # Thread ID from Message-ID or subject normalization
    thread_id = msg_id or f"thread_{hashlib.md5(subject.encode()).hexdigest()[:8]}"

    # Derive folder owner from file path (e.g. "allen-p/_sent_mail/1.")
    folder_owner = file_path.split("/")[0] if "/" in file_path else ""

    # Content hash for dedup
    content_hash = hashlib.sha256((subject + body).encode()).hexdigest()[:16]

    # Parse date to ISO8601 best-effort
    iso_ts = parse_date(date_str)

    return {
        "id":           file_path,
        "thread_id":    thread_id,
        "timestamp":    iso_ts,
        "from":         sender,
        "to":           recipients,
        "subject":      subject,
        "body":         body,
        "x_from":       x_from,
        "x_folder":     x_folder,
        "folder_owner": folder_owner,
        "content_hash": content_hash,
        "word_count":   len(body.split()),
        "msg_type":     classify_message(subject, body),
        "claim_type":   classify_claim_type(subject, body),
    }

def parse_date(date_str: str) -> str:
    """Best-effort parse of email Date header to ISO8601."""
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "2001-01-01T00:00:00Z"

# ── Message classification ─────────────────────────────────────────────────────

DECISION_KW  = ["decided", "decision", "approved", "agreed", "confirmed", "will proceed",
                 "going forward", "move forward", "we have decided", "proceed with"]
REVERSAL_KW  = ["walk back", "reverse", "reversing", "retract", "pausing", "cancell",
                 "no longer", "changed my mind", "prior decision", "earlier position"]
STATUS_KW    = ["status", "update", "on track", "behind", "completed", "progress", "milestone"]
CONCERN_KW   = ["concern", "risk", "flag", "issue", "problem", "warning", "caution",
                 "not compliant", "may not", "question about", "worried"]
MEETING_KW   = ["attendees", "agenda", "meeting notes", "action items", "minutes",
                 "discussed", "conference call", "call notes"]
ASSIGNMENT_KW= ["action:", "owner:", "responsible", "assigned to", "you will", "please handle",
                 "take point", "follow up", "your action"]

def classify_message(subject: str, body: str) -> str:
    text = (subject + " " + body[:500]).lower()
    if any(k in text for k in REVERSAL_KW):   return "reversal"
    if any(k in text for k in DECISION_KW):   return "decision"
    if any(k in text for k in MEETING_KW):    return "meeting_notes"
    if any(k in text for k in CONCERN_KW):    return "concern"
    if any(k in text for k in ASSIGNMENT_KW): return "assignment"
    if any(k in text for k in STATUS_KW):     return "status"
    return "general"

def classify_claim_type(subject: str, body: str) -> str:
    m = classify_message(subject, body)
    return {
        "decision":     "DECISION",
        "reversal":     "REVERSAL",
        "meeting_notes":"MEETING",
        "concern":      "RISK_FLAG",
        "assignment":   "ASSIGNMENT",
        "status":       "STATUS",
        "general":      "FACT",
    }.get(m, "FACT")

# ── Person extraction from corpus ─────────────────────────────────────────────

def build_people_index(messages: list) -> list:
    """
    Derive canonical persons from observed email addresses.
    Groups aliases: phillip.allen@enron.com + "Phillip K Allen" → one PERSON entity.
    """
    from collections import defaultdict

    addr_to_names = defaultdict(set)  # email_addr → set of display names seen

    for msg in messages:
        sender = msg.get("from", "")
        xfrom  = msg.get("x_from", "")
        if sender and xfrom:
            addr_to_names[sender.lower()].add(xfrom.strip())
        elif sender:
            addr_to_names[sender.lower()].add(sender)

        for r in msg.get("to", []):
            r = r.strip().lower()
            if r:
                addr_to_names[r].add(r)

    people = []
    seen_ids = set()
    pid = 0
    for addr, names in addr_to_names.items():
        if not addr or addr in seen_ids:
            continue
        seen_ids.add(addr)
        # Canonical name: prefer X-From style (e.g. "Phillip K Allen") over email
        name_candidates = [n for n in names if not n.endswith(".com") and "@" not in n and len(n) > 3]
        canonical = name_candidates[0] if name_candidates else addr
        aliases = list(names | {addr})
        people.append({
            "id":        f"p{pid:04d}",
            "canonical": canonical,
            "aliases":   aliases,
            "role":      "",
        })
        pid += 1

    return people

# ── Main loader ───────────────────────────────────────────────────────────────

def load_enron_csv(csv_path: str, limit: int = 5000, min_body_words: int = 5) -> dict:
    """
    Load emails.csv and return corpus dict compatible with the pipeline.
    limit: max number of emails to process (full dataset = ~500k)
    """
    messages = []
    skipped  = 0
    csv.field_size_limit(10_000_000)  # Enron emails can be large

    print(f"Loading {csv_path} (limit={limit})...")
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            parsed = parse_raw_message(row["file"], row["message"])
            if parsed is None or parsed["word_count"] < min_body_words:
                skipped += 1
                continue
            messages.append(parsed)

    print(f"  Parsed {len(messages)} messages ({skipped} skipped, {limit} attempted)")

    # Build people index from observed senders
    people = build_people_index(messages)
    print(f"  Discovered {len(people)} unique email identities")

    # Trim to top 100 most active senders for entity graph (rest still in messages)
    from collections import Counter
    sender_counts = Counter(m["from"].lower() for m in messages)
    top_senders = {addr for addr, _ in sender_counts.most_common(100)}
    people_trimmed = [p for p in people if any(
        a.lower() in top_senders for a in p["aliases"]
    )]
    print(f"  Keeping {len(people_trimmed)} top-active persons as graph entities")

    # No structured project list from raw corpus — we'll extract from subjects
    projects = extract_projects_from_messages(messages)
    print(f"  Extracted {len(projects)} project/topic entities from subjects")

    return {
        "corpus":       "enron_email_dataset",
        "source":       "Kaggle: wcukierski/enron-email-dataset (emails.csv)",
        "reproduce":    "kaggle datasets download -d wcukierski/enron-email-dataset",
        "generated_at": "2026-01-01T00:00:00Z",
        "people":       people_trimmed,
        "projects":     projects,
        "messages":     messages,
        "stats": {
            "total_messages": len(messages),
            "unique_threads": len(set(m["thread_id"] for m in messages)),
            "people":         len(people_trimmed),
            "projects":       len(projects),
        }
    }

def extract_projects_from_messages(messages: list) -> list:
    """
    Heuristically extract recurring proper-noun phrases from subjects as 'project' entities.
    """
    from collections import Counter
    import re

    # Extract capitalized multi-word phrases from subjects
    phrase_re = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    phrase_counter = Counter()

    for msg in messages:
        subj = msg.get("subject", "")
        for m in phrase_re.finditer(subj):
            phrase_counter[m.group(0)] += 1

    # Keep phrases that appear ≥ 3 times and aren't just names
    common_stopnames = {"Re Re","Fw Fw","Please Let","Thank You","Best Regards"}
    projects = []
    pid = 0
    for phrase, count in phrase_counter.most_common(30):
        if count < 3: break
        if phrase in common_stopnames: continue
        projects.append({
            "id":      f"proj{pid:03d}",
            "name":    phrase,
            "aliases": [phrase],
        })
        pid += 1

    return projects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="emails.csv", help="Path to emails.csv")
    parser.add_argument("--limit", default=5000, type=int, help="Max emails to load")
    parser.add_argument("--out",   default="data/raw/corpus.json")
    args = parser.parse_args()

    corpus = load_enron_csv(args.csv, limit=args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(corpus, indent=2))
    print(f"\nCorpus saved → {args.out}")
    print(f"Stats: {corpus['stats']}")
