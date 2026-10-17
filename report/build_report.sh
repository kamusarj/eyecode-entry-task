#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
pdflatex -output-directory "$build_dir" -interaction=nonstopmode -halt-on-error technical_report.tex > "$build_dir/build.log"
pdflatex -output-directory "$build_dir" -interaction=nonstopmode -halt-on-error technical_report.tex >> "$build_dir/build.log"
cp "$build_dir/technical_report.pdf" technical_report.pdf
pdfinfo technical_report.pdf | sed -n '/^Pages:/p'
