import sys
import subprocess
from pathlib import Path

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

inbox_dir = Path("00-inbox")
files = list(inbox_dir.glob("*"))

tier1_exts = {".vtt", ".csv", ".xlsx", ".md", ".txt", ".mdc"}
tier2_exts = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}
tier3_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".m4a", ".wav"}

tier1 = []
tier2 = []
tier3 = []

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

if len(sys.argv) < 2:
    print("Usage: python scripts/batch_ingest.py <tier_number>")
    print("Example: python scripts/batch_ingest.py 1")
    sys.exit(1)

tier_num = int(sys.argv[1])
if tier_num == 1:
    target_files = sorted(tier1)
elif tier_num == 2:
    target_files = sorted(tier2)
elif tier_num == 3:
    target_files = sorted(tier3)
else:
    print(f"Invalid tier number: {tier_num}")
    sys.exit(1)

print(f"Starting ingestion for Tier {tier_num} ({len(target_files)} files)...")

success_count = 0
failed_count = 0

for idx, f in enumerate(target_files, 1):
    print(f"\n[{idx}/{len(target_files)}] Ingesting: {f.name}")
    try:
        # Run uv run brain ingest "<path>"
        result = subprocess.run(
            ["uv", "run", "brain", "ingest", str(f)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        print(result.stdout)
        if result.stderr:
            print("Stderr output:")
            print(result.stderr)
        
        if result.returncode == 0 and "failed" not in result.stdout.lower():
            success_count += 1
        else:
            failed_count += 1
            print(f"Warning: Ingestion of {f.name} might have failed or had warnings.")
    except Exception as e:
        failed_count += 1
        print(f"Error ingesting {f.name}: {e}")

print("\n========================================")
print(f"Tier {tier_num} Ingestion Summary:")
print(f"  - Total processed: {len(target_files)}")
print(f"  - Successfully ingested: {success_count}")
print(f"  - Failed/Warnings: {failed_count}")
print("========================================")
