#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:?usage: $0 <dir>}"

for f in "$INPUT_DIR"/*.yaml; do
  sed -i.bak 's|^\([[:space:]]*msa:\).*|\1 empty|' "$f" && rm "${f}.bak"
done
