#!/usr/bin/env bash
# Download the public Stack Exchange dumps used by the benchmark and unpack them into $OPS_BENCH_DIR.
# serverfault -> $OPS_BENCH_DIR/sf/{raw,xml}      dba -> $OPS_BENCH_DIR/{raw,xml}
# Licence of the data: CC BY-SA 4.0 (see DATA.md). Needs curl and a Python with py7zr (pip install py7zr).
# usage: OPS_BENCH_DIR=$HOME/ops-benchmark bash scripts/download_data.sh [serverfault|dba|all]
set -euo pipefail
BENCH="${OPS_BENCH_DIR:-$HOME/ops-benchmark}"
WHAT="${1:-serverfault}"
PY="${PYTHON:-python3}"

fetch() {  # fetch <site file name> <target dir>
  mkdir -p "$2/raw" "$2/xml"
  [ -f "$2/raw/$1" ] || curl -L --fail -o "$2/raw/$1" "https://archive.org/download/stackexchange/$1"
  "$PY" -I -c "import py7zr,sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" "$2/raw/$1" "$2/xml"
}
case "$WHAT" in
  serverfault) fetch serverfault.com.7z "$BENCH/sf" ;;
  dba)         fetch dba.stackexchange.com.7z "$BENCH" ;;
  all)         fetch serverfault.com.7z "$BENCH/sf"; fetch dba.stackexchange.com.7z "$BENCH" ;;
  *) echo "usage: $0 [serverfault|dba|all]" >&2; exit 2 ;;
esac
echo "done: $BENCH"
