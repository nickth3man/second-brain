from pathlib import Path

inbox_dir = Path("00-inbox")
files = list(inbox_dir.glob("*"))

tier1_exts = {".vtt", ".csv", ".xlsx", ".md", ".txt", ".mdc"}
tier2_exts = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}
tier3_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".m4a", ".wav"}

tier1 = []
tier2 = []
tier3 = []
other = []

for f in files:
    if f.is_dir():
        continue
    ext = f.suffix.lower()
    if ext in tier1_exts:
        tier1.append(f)
    elif ext in tier2_exts:
        tier2.append(f)
    elif ext in tier3_exts:
        tier3.append(f)
    else:
        other.append(f)

print(f"Total files: {len(files)}")
print(f"Tier 1 (Fast): {len(tier1)}")
for f in sorted(tier1):
    print(f"  - {f.name}")
print(f"Tier 2 (Medium): {len(tier2)}")
for f in sorted(tier2):
    print(f"  - {f.name}")
print(f"Tier 3 (Slow): {len(tier3)}")
for f in sorted(tier3):
    print(f"  - {f.name}")
print(f"Other: {len(other)}")
for f in sorted(other):
    print(f"  - {f.name}")
