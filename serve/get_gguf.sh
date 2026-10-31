#!/usr/bin/env bash
# Get the GGUF file for MODEL at QUANT into DIR: reuse a checked copy, or convert it from Hugging Face.
#
#   bash serve/get_gguf.sh nanbeige42 q8_0 ~/models/nanbeige42
#   bash serve/get_gguf.sh nanbeige42 q8_0 /content/drive/MyDrive/grc-eval/models/nanbeige42/b82e54b-b12020
#   bash serve/get_gguf.sh nanbeige42 q8_0 DIR  team/nanbeige-sft-v1 <revision>    # a fine-tuned, merged checkpoint
#
# 1. DIR/<file> exists: hash it. When serve/models.json has a reference for this platform, it must match.
# 2. Otherwise convert: convert/setup.sh, download HF_REPO at REVISION, convert/to_gguf.sh, copy to DIR.
#    For the pinned base model the BF16 intermediate must match its reference, so a quantized file that
#    differs from another platform's can only differ in rounding.
# References apply only to the pinned source (serve/models.json source.hf_repo + source.revision).
# Scratch space: $SCRATCH (default: a temp folder), emptied afterwards; Nanbeige needs ~21 GB.
# Needs the pinned llama.cpp (serve/install_llamacpp.sh), git, uv and python3.
# Last line: RESULT file=<path> sha256=<hash> check=<match|new-reference|no-reference>
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/pinned.sh"
usage="usage: get_gguf.sh MODEL QUANT DIR [HF_REPO REVISION]"
model="${1:?$usage}"
quant="${2:?$usage}"
dir="${3:?$usage}"
platform="$(uname -s)-$(uname -m)"

# Progress goes to stderr, so stdout carries only verdict() results and the RESULT line.
log() { printf '[get_gguf %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# Prints: file name, pinned repo, pinned revision, BF16 reference, this quant's reference for this platform.
read -r gguf_name src_repo src_rev bf16_ref quant_ref < <(python3 - "$here/models.json" "$model" "$quant" "$platform" <<'PY'
import json, sys
cfg, model, quant, platform = json.load(open(sys.argv[1])), *sys.argv[2:]
if model not in cfg["models"]:
    sys.exit(f"unknown model {model!r}; known: {', '.join(cfg['models'])}")
m = cfg["models"][model]
if quant not in m["gguf"]:
    sys.exit(f"unknown quant {quant!r} for {model}; known: {', '.join(m['gguf'])}")
refs = m.get("sha256", {})
ref = refs.get(quant)
ref = ref.get(platform) if isinstance(ref, dict) else ref
print(m["gguf"][quant], m["source"]["hf_repo"], m["source"]["revision"], refs.get("bf16") or "-", ref or "-")
PY
)
repo="${4:-$src_repo}"
rev="${5:-$src_rev}"
[ "$bf16_ref" = "-" ] && bf16_ref=""
[ "$quant_ref" = "-" ] && quant_ref=""
if [ "$repo" != "$src_repo" ] || [ "$rev" != "$src_rev" ]; then
  pinned=0; bf16_ref=""; quant_ref=""
  log "source $repo @ $rev is not the pinned $src_repo: no reference hashes, the file's hash is only recorded"
else
  pinned=1
fi
target="$dir/$gguf_name"
log "model $model, $quant -> $target (platform $platform)"

# Prints check=... for a finished file, or fails when it contradicts a reference.
verdict() {
  local file="$1" have="$2" ref="$3"
  if [ -n "$ref" ]; then
    [ "$have" = "$ref" ] || die "$file has sha256 $have, but serve/models.json expects $ref for $platform. Delete the file and run again to convert it afresh; if the new file differs too, the conversion is not reproducible: report it."
    log "sha256 matches the $platform reference"
    echo match
  elif [ "$pinned" = 1 ]; then
    log "no $quant reference for $platform yet. Add this to serve/models.json (sha256.$quant.\"$platform\"): $have"
    echo new-reference
  else
    echo no-reference
  fi
}

if [ -f "$target" ]; then
  log "found a cached copy; checking its sha256 (a few seconds per GB)..."
  have="$(sha "$target")"
  check="$(verdict "$target" "$have" "$quant_ref")"
  log "using the cached copy"
  echo "RESULT file=$target sha256=$have check=$check"
  exit 0
fi

log "no cached copy: converting from Hugging Face (first time only, ~5-10 min plus the download)"
require_llama_build llama-quantize
log "step 1/4: converter setup (convert/setup.sh)"
bash "$here/../convert/setup.sh"

scratch="${SCRATCH:-$(mktemp -d)}"
# Named after the repo, as convert_hf_to_gguf.py puts the folder name into the file (general.name):
# any other name gives a different file and hash.
hf_dir="$scratch/hf/${repo##*/}"
out_dir="$scratch/gguf/$model"
mkdir -p "$hf_dir" "$out_dir"
trap 'rm -rf "$hf_dir" "$out_dir"; [ -n "${SCRATCH:-}" ] || rm -rf "$scratch"' EXIT
log "step 2/4: downloading $repo @ $rev into $hf_dir"
started=$SECONDS
"$here/../convert/.venv/bin/hf" download "$repo" --revision "$rev" --local-dir "$hf_dir"
log "downloaded $(du -sh "$hf_dir" | cut -f1) in $((SECONDS - started)) s"
[ -f "$hf_dir/config.json" ] || die "$repo has no config.json: not a full Hugging Face checkpoint"
if ! ls "$hf_dir"/*.safetensors >/dev/null 2>&1; then
  [ -f "$hf_dir/adapter_config.json" ] && die "$repo holds a LoRA adapter, not full weights: merge it into the base model and upload the merged checkpoint"
  die "$repo has no .safetensors weight files"
fi

# GGUF file names end in -<QUANT>.gguf (e.g. Nanbeige4.2-3B-Q8_0.gguf); to_gguf.sh takes the part before it.
suffix="${gguf_name##*-}"
quant_up="${suffix%.gguf}"
name="${gguf_name%-*}"
if [ "$quant_up" = "BF16" ]; then quants=""; plan="BF16"; else quants="$quant_up"; plan="BF16, then $quant_up"; fi
log "step 3/4: converting to $plan (convert/to_gguf.sh)"
started=$SECONDS
QUANTS="$quants" bash "$here/../convert/to_gguf.sh" "$hf_dir" "$out_dir" "$name"
log "converted in $((SECONDS - started)) s"

if [ "$pinned" = 1 ] && [ -n "$bf16_ref" ]; then
  log "checking the BF16 intermediate against the reference..."
  bf16_have="$(sha "$out_dir/$name-BF16.gguf")"
  [ "$bf16_have" = "$bf16_ref" ] || die "BF16 sha256 is $bf16_have, expected $bf16_ref: the download or the converter differs from the pinned setup"
  log "BF16 matches the reference, so the converter's input and output are identical to the pinned setup"
fi
have="$(sha "$out_dir/$gguf_name")"
check="$(verdict "$out_dir/$gguf_name" "$have" "$quant_ref")"

log "step 4/4: copying $(du -h "$out_dir/$gguf_name" | cut -f1) to $dir"
mkdir -p "$dir"
cp "$out_dir/$gguf_name" "$target.partial"
mv "$target.partial" "$target"
log "done; scratch files removed"
echo "RESULT file=$target sha256=$have check=$check"
