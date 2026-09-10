#!/usr/bin/env bash
# Compile the report. With an argument, also drop a copy of the PDF there.
#
#   ./scripts/build_report.sh
#   ./scripts/build_report.sh ~/somewhere/else
set -euo pipefail
cd "$(dirname "$0")/../report"

tectonic main.tex

if [ $# -ge 1 ]; then
  cp main.pdf "$1/Report DLAI.pdf"
  echo "copy written to $1/Report DLAI.pdf"
fi
