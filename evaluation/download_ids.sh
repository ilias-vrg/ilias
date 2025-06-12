#!/bin/bash

# parse arguments and default values
DOWNLOAD_DIR=${1:-"./ilias"}
BASE_URL="https://vrg.fel.cvut.cz/ilias_data/"

# create main download directory if it does not exist
mkdir -p "$DOWNLOAD_DIR"

# List of files to download
IMAGEID_URLS=(
  "$BASE_URL/image_ids/image_query_ids.txt"
  "$BASE_URL/image_ids/text_query_ids.txt"
  "$BASE_URL/image_ids/positive_ids.txt"
  "$BASE_URL/image_ids/distractor_ids.txt.gz"
)

echo "Downloading Image ID files..."
for URL in "${IMAGEID_URLS[@]}"; do
  # Remove the base URL to get the path
  RELPATH="${URL#"$BASE_URL/"}"
  # Extract the directory portion
  DIR="$DOWNLOAD_DIR/$(dirname "$RELPATH")"
  # Create local directory if needed
  mkdir -p "$DIR"
  # Download into that directory, preserving filename
  echo "  - $(basename "$RELPATH") -> $DIR/"
  wget -q --show-progress -O "$DIR/$(basename "$RELPATH")" "$URL"
done
echo "Image ID files lists downloaded."
