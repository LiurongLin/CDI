#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error poster.tex >/tmp/spie_poster_build.log
pdflatex -interaction=nonstopmode -halt-on-error poster.tex >/tmp/spie_poster_build.log
printf 'Built %s/poster.pdf\n' "$(pwd)"
