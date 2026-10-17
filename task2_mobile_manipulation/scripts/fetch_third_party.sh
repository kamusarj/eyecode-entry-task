#!/usr/bin/env bash
# Fetch the official HoMMI code (+ submodules) and the MuJoCo Menagerie Panda at the exact commits used.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party" && cd "$ROOT/third_party"

if [ ! -d hommi ]; then
  git clone https://github.com/xxm19/hommi.git
  git -C hommi checkout 2c002aaf211637d1f6fe98331ce3a444e7f57a8b
  git -C hommi submodule update --init deps/rby1-wbc deps/universal_manipulation_interface
  git -C hommi/deps/rby1-wbc checkout aa9e6aef05af067fe0ca016cda166f8eea668239
  git -C hommi/deps/universal_manipulation_interface checkout 8ea6ba767a981be8d7b3be84f1bdbd74dbf2c2ed
  # memory fix for the dataset builder (stream-decode videos; identical output)
  git -C hommi apply "$ROOT/patches/hommi_stream_decode.patch"
fi

if [ ! -d mujoco_menagerie ]; then
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
  git -C mujoco_menagerie checkout 62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35
  git -C mujoco_menagerie sparse-checkout set franka_emika_panda
fi
echo "third_party ready"
