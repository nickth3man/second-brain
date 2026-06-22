import json
from pathlib import Path
from second_brain.config import load_config
from second_brain.state import BrainStateStore

cfg = load_config()
store = BrainStateStore.load(cfg)

orphan_sources = [(sid, src) for sid, src in store.state.sources.items() if not src.topics]
print(f"ORPHAN SOURCES ({len(orphan_sources)}) - sources with no assigned topics:")
for sid, src in sorted(orphan_sources):
    print(f"  {sid}  stage={src.stage}  raw={src.raw}")

print()
print("L2 MISMATCHES from latest.json (topic claims source, but source doesn't list that topic):")
latest = Path(".brain/evals/latest.json")
if latest.exists():
    data = json.loads(latest.read_text())
    for m in data.get("l2", {}).get("mismatches", []):
        print(f"  topic={m['topic']}  source={m['source']}")

print()
print("WIKI PAGES citation format check:")
wiki_dir = Path("90-wiki")
pages = list(wiki_dir.glob("*.md"))
print(f"  Total wiki pages: {len(pages)}")
no_sources_section = []
has_sources_no_links = []
for p in sorted(pages):
    text = p.read_text(encoding="utf-8", errors="replace")
    has_source_links = "[source](../50-sources/" in text
    sources_section = "## Sources" in text
    if not sources_section:
        no_sources_section.append(p.stem)
    elif not has_source_links:
        has_sources_no_links.append(p.stem)

print(f"  Pages with no '## Sources' section: {len(no_sources_section)}")
for s in no_sources_section[:10]:
    print(f"    - {s}")
print(f"  Pages with '## Sources' but no properly formatted source links: {len(has_sources_no_links)}")
for s in has_sources_no_links[:10]:
    print(f"    - {s}")

print()
print("DEADLETTER directory:")
deadletter = Path(".brain/deadletter")
dl_files = [f for f in deadletter.glob("*") if f.name != ".gitkeep"]
print(f"  Files in deadletter: {len(dl_files)}")
for f in sorted(dl_files):
    print(f"  - {f.name}")

print()
print("EVALS artifacts:")
evals_dir = Path(".brain/evals")
for f in sorted(evals_dir.glob("*.json")):
    data = json.loads(f.read_text())
    print(f"  {f.name}: {list(data.keys())}")
