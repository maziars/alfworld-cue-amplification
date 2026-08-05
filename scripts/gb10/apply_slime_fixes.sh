#!/usr/bin/env bash
# Fix for building slime GB10 image. Upstream slime Dockerfile.gb10 pins
# MEGATRON_COMMIT=3714d81d (stale); slime megatron.patch was regenerated for
# 1dcf0dafa (PR #1867) but the GB10 ARG was never bumped -> git apply conflicts.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/slime/docker/Dockerfile.gb10"
sed -i "s/MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862/MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087/" "$D"
grep -n "^ARG MEGATRON_COMMIT=" "$D"
