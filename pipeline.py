#!/usr/bin/env python3
"""
Master Pipeline — Real Enron Dataset
======================================
Usage:
    python3 pipeline.py                           # expects emails.csv in current dir
    python3 pipeline.py --csv /path/to/emails.csv
    python3 pipeline.py --csv emails.csv --limit 3000
"""

import json, sys, argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "extraction"))
sys.path.insert(0, str(ROOT / "src" / "dedup"))
sys.path.insert(0, str(ROOT / "src" / "retrieval"))

from src.schema import MemoryGraph
from src.corpus_loader import load_enron_csv
from src.extraction.extractor import run_extraction
from src.dedup.deduplicator import run_dedup, save_merge_log, MERGE_LOG
from src.retrieval.retriever import RetrievalEngine, run_sample_queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="emails.csv", help="Path to Enron emails.csv")
    parser.add_argument("--limit", default=5000, type=int, help="Max emails to process")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"\nERROR: '{csv_path}' not found.")
        print("Download the dataset:")
        print("  kaggle datasets download -d wcukierski/enron-email-dataset")
        print("  unzip enron-email-dataset.zip")
        print("Then: python3 pipeline.py --csv emails.csv")
        sys.exit(1)

    print("=" * 60)
    print("Layer10 Memory Pipeline — Real Enron Dataset")
    print("=" * 60)

    # 1. Load
    print(f"\n[1/5] Loading {csv_path} (limit={args.limit})...")
    corpus = load_enron_csv(str(csv_path), limit=args.limit)
    corpus_path = ROOT / "data" / "raw" / "corpus.json"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus, indent=2))
    s = corpus["stats"]
    print(f"    {s['total_messages']} messages, {s['unique_threads']} threads, "
          f"{s['people']} person entities, {s['projects']} project entities")

    # 2. Extract
    print("\n[2/5] Running extraction pipeline...")
    graph = MemoryGraph()
    run_extraction(str(corpus_path), graph)
    print(f"    Entities:  {len(graph.entities)}")
    print(f"    Claims:    {len(graph.claims)}")
    print(f"    Evidence:  {len(graph.evidence)}")
    print(f"    Artifacts: {len(graph.artifacts)}")

    raw_path = ROOT / "data" / "processed" / "graph_raw.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(raw_path))

    # 3. Dedup
    print("\n[3/5] Running deduplication & canonicalization...")
    run_dedup(graph)
    print(f"    Post-dedup entities:  {len(graph.entities)}")
    print(f"    Post-dedup claims:    {len(graph.claims)}")
    print(f"    Post-dedup relations: {len(graph.relations)}")
    print(f"    Merge log entries:    {len(MERGE_LOG)}")

    # 4. Save
    print("\n[4/5] Saving final graph...")
    final_path = ROOT / "data" / "processed" / "graph_final.json"
    graph.save(str(final_path))
    merge_log_path = ROOT / "data" / "processed" / "merge_log.json"
    save_merge_log(str(merge_log_path))
    print(f"    Graph  → {final_path}")
    print(f"    Merges → {merge_log_path}")

    # 5. Retrieval
    print("\n[5/5] Running sample retrieval queries...")
    engine = RetrievalEngine(graph)
    query_results = run_sample_queries(engine)
    queries_path = ROOT / "outputs" / "sample_queries.json"
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text(json.dumps(query_results, indent=2))
    print(f"    Results → {queries_path}")

    # Bake data directly into index.html — open file:// without any server
    viz_dir = ROOT / "viz"
    viz_dir.mkdir(exist_ok=True)

    html_path = viz_dir / "index.html"
    html = html_path.read_text()

    if "// __INJECTED_DATA__" not in html:
        print("WARNING: index.html placeholder missing — viz may show no data")
    else:
        graph_json = json.dumps(graph.to_dict())
        merge_json = json.dumps(MERGE_LOG)
        html = html.replace(
            "// __INJECTED_DATA__",
            f"GRAPH = {graph_json};\n  MERGE_LOG = {merge_json};"
        )
        html_path.write_text(html)
        print("    Data baked into viz/index.html")

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(json.dumps(graph.to_dict()["stats"], indent=2))
    print("\nJust open viz/index.html in any browser — no server needed.")

if __name__ == "__main__":
    main()
