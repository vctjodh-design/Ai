#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-0.5B"
REVISION="060db6499f32faf8b98477b0a26969ef7d8b9987"
DEST="models/qwen2.5-0.5b"
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/${REVISION}"
EXPECTED_SHA256="88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"

cd "$(dirname "$0")/.."
mkdir -p "$DEST"

download_file() {
  local name="$1"
  local target="${DEST}/${name}"
  local partial="${target}.partial"

  if [[ -f "$target" ]]; then
    return
  fi

  echo "Downloading ${name}..."
  if [[ -f "$partial" ]]; then
    curl --location --fail --retry 5 --retry-all-errors --connect-timeout 30 \
      --continue-at - --output "$partial" "${BASE_URL}/${name}"
  else
    curl --location --fail --retry 5 --retry-all-errors --connect-timeout 30 \
      --output "$partial" "${BASE_URL}/${name}"
  fi
  mv "$partial" "$target"
}

for file in \
  LICENSE \
  README.md \
  config.json \
  generation_config.json \
  merges.txt \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  model.safetensors
do
  download_file "$file"
done

actual_sha256="$(sha256sum "${DEST}/model.safetensors" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "Model checksum mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha256}" >&2
  exit 1
fi

echo "Model downloaded and verified in ${DEST}."
