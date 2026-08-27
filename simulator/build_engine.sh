#!/usr/bin/env bash
# Build the engine from the C++ source into the Python bindings beside it.
#
# ptcg_engine/ is the whole engine in one place: the C++ sources under src/ and
# the Python bindings that call into them beside it. The bindings need a
# libcg.so; this builds one from src/, so the engine the simulator runs is
# always the engine in this repository and no binary is kept in version control.
#
#   ./build_engine.sh            # builds ptcg_engine/libcg.so
#   ./build_engine.sh /tmp/x.so  # builds somewhere else, to compare first
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/ptcg_engine/src/Export.cpp"
out="${1:-$here/ptcg_engine/libcg.so}"

[ -f "$src" ] || { echo "engine source not found: $src" >&2; exit 1; }

# -include climits: the source reaches for INT_MAX without including it, which
# MSVC supplies transitively and GCC/Clang do not. Compiler flag rather than an
# edit, because the package is competition-licensed and should stay untouched.
${CXX:-g++} -std=c++20 -O2 -fPIC -shared -include climits -o "$out" "$src"
echo "built $out ($(du -h "$out" | cut -f1))"

# Every agent bundle carries its own copy of the bindings, because a bundle runs
# in its own interpreter and must not depend on this directory's. They get the
# same build: one engine everywhere, and no .so in git.
if [ "$out" = "$here/ptcg_engine/libcg.so" ]; then
  for bundle_cg in "$here"/agents/*/cg; do
    [ -d "$bundle_cg" ] || continue
    cp "$out" "$bundle_cg/libcg.so"
    echo "  installed into ${bundle_cg#"$here"/}"
  done
fi
