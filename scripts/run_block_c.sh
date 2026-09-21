#!/usr/bin/env bash
# Block C single-stage launcher for local or remote Pixi environments.
#
# Usage:
#   scripts/run_block_c.sh --stage c1-sensitivity --jobs 10 --seeds 42 43 44 45 46
#   scripts/run_block_c.sh --stage c2-mismatch --input-experiment-id <c1-id>
#
# The Python runner skips an already completed exact-config campaign. Use
# --rerun only when a new campaign ID is intentionally required.
set -euo pipefail

cd "$(dirname "$0")/.."

export PATH="$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || { echo "--stage requires a value" >&2; exit 2; }
      STAGE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 --stage <c0-smoke|c1-sensitivity|c2-mismatch|c3-mobility|c4-attrition|c5-generalization|c6-random|c6-editorial|c7-midflight|c7-state-matched|c7-timing|c7-severity|c7-editorial> [runner options]"
      echo "Completed exact-config campaigns are skipped; pass --rerun to force a new campaign."
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$STAGE" ]] || { echo "missing --stage; use --help" >&2; exit 2; }
exec pixi run python scripts/audit_block_c.py "$STAGE" "${ARGS[@]}"
