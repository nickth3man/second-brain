"""Ingest all VTT files from 00-inbox/ with real-time streamed output."""
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

vtt_files = sorted(Path("00-inbox").glob("*.vtt"))
print(f"Found {len(vtt_files)} VTT files to ingest", flush=True)

for idx, f in enumerate(vtt_files, 1):
    print(f"\n{'='*60}", flush=True)
    print(f"[{idx}/{len(vtt_files)}] {f.name}", flush=True)
    print(f"{'='*60}", flush=True)

    proc = subprocess.Popen(
        ["uv", "run", "brain", "ingest", str(f)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"[WARN] Exit code {proc.returncode} for {f.name}", flush=True)

print("\nAll VTTs done.", flush=True)
