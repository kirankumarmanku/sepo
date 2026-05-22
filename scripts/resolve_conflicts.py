"""
resolve_conflicts.py — resolves git merge conflicts in grpo_sepo.py
=====================================================================
Strategy: for every conflict block, keep the 'fdea898 (Generated data
for kuhn)' side and discard the HEAD side. That side contains:
  - SEPO penalty caching (sepo_caches, kl_since_refresh)
  - --sepo-eval-every / --sepo-kl-threshold CLI flags
  - Shared aux episodes (one per opponent group, not per rollout)
  - Per-game KL tracking + refresh prints
  - Variable-length episode handling (min length) for Kuhn

Usage:
  python resolve_conflicts.py
"""

import re
import sys
from pathlib import Path

INPUT = "grpo_sepo.py"
OUTPUT = "grpo_sepo.py"  # in-place
BACKUP = "grpo_sepo.py.preresolve.bak"


def main():
    src = Path(INPUT).read_text()

    # Backup
    Path(BACKUP).write_text(src)
    print(f"Backed up original to {BACKUP}")

    # Match every conflict block:
    #   <<<<<<< HEAD
    #   ...head content...
    #   =======
    #   ...incoming content...
    #   >>>>>>> <commit>
    #
    # Replace with just the incoming content (the part between ======= and >>>>>>>).
    conflict_re = re.compile(
        r"<{7} HEAD\n"      # <<<<<<< HEAD
        r"(.*?)"            # head content (discarded)
        r"={7}\n"           # =======
        r"(.*?)"            # incoming content (kept)
        r">{7} [^\n]*\n",   # >>>>>>> commit
        flags=re.DOTALL,
    )

    matches = conflict_re.findall(src)
    print(f"Found {len(matches)} conflict block(s)")

    resolved = conflict_re.sub(lambda m: m.group(2), src)

    # Sanity: ensure no markers remain
    leftover = re.search(r"^<{7} HEAD$|^>{7} ", resolved, flags=re.MULTILINE)
    if leftover:
        print(f"WARNING: leftover marker at offset {leftover.start()}")
        ctx_start = max(0, leftover.start() - 100)
        ctx_end = min(len(resolved), leftover.start() + 100)
        print(f"Context: ...{resolved[ctx_start:ctx_end]}...")
        sys.exit(1)

    Path(OUTPUT).write_text(resolved)
    print(f"Wrote resolved file to {OUTPUT}")
    print(f"\nVerify with: python -c \"import ast; ast.parse(open('{OUTPUT}').read())\"")


if __name__ == "__main__":
    main()
