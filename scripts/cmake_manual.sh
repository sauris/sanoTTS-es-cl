#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/work/piper1-gpl"
export PATH="$HOME/venvs/piper/bin:$PATH"
rm -rf /tmp/piper-build
cmake -B /tmp/piper-build -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -25
echo "CONFIGURE_DONE=$?"
cmake --build /tmp/piper-build -j 8 2>&1 | tail -15
echo "BUILD_DONE=$?"
find /tmp/piper-build -name 'espeakbridge*.so' -o -name 'libespeak-ng.a' | head
