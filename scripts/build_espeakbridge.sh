#!/usr/bin/env bash
# Build the espeakbridge cython/cmake extension in-place (editable installs of
# scikit-build projects don't compile C extensions; TRAINING.md documents this).
set -euo pipefail
cd "$HOME/work/piper1-gpl"
"$HOME/venvs/piper/bin/pip" install -q scikit-build
PATH="$HOME/venvs/piper/bin:$PATH" "$HOME/venvs/piper/bin/python" setup.py build_ext --inplace 2>&1 | tail -8
echo "--- built artifacts ---"
find src/piper -maxdepth 1 -name '*.so' -o -maxdepth 1 -name 'espeakbridge*' | head
echo "--- import test ---"
cd "$HOME/work/piper1-gpl/src"
"$HOME/venvs/piper/bin/python" -c "from piper import espeakbridge; print('espeakbridge OK:', espeakbridge.__file__)"
