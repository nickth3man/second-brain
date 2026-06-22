from pathlib import Path

mdc_files = [
    "codequality.mdc", "database.mdc", "nextjs.mdc",
    "python.mdc", "tailwind.mdc", "typescript.mdc",
]

for name in mdc_files:
    dl = Path(".brain/deadletter") / name
    stem = name.replace(".mdc", "")
    sources_matches = list(Path("50-sources").glob(f"*{stem}*"))
    print(f"{name}:")
    print(f"  deadletter: {dl.exists()}")
    print(f"  50-sources: {[m.name for m in sources_matches]}")
