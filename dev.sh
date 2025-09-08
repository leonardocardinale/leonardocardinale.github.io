#!/bin/bash
# -*- coding: utf-8 -*-
# +
#!/usr/bin/env bash
set -euo pipefail

# Clean site output
rm -rf _site fr/_site el/_site it/_site

# --- 0) Ensure per-language assets exist *before* rendering subprojects ---
mkdir -p fr/assets el/assets it/assets
cp -f assets/background.jpeg fr/assets/ || true
cp -f assets/background.jpeg el/assets/ || true
cp -f assets/background.jpeg it/assets/ || true

# --- 1) Root FIRST (so it does its clean once) ---
quarto render

# --- 2) Then each language (write into _site/<lang>) ---
( cd fr && quarto render )
( cd el && quarto render )
( cd it && quarto render )

# --- 3) Copy static folders into built site (optional safety) ---
mkdir -p _site/cv _site/flags
cp -f cv/* _site/cv/ || true
cp -f flags/* _site/flags/ || true

# --- 4) Serve statically ---
python3 -m http.server -d _site 4701

# -


