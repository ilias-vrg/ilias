#!/bin/bash

# parse arguments and default values
DOWNLOAD_DIR=${1:-"./ilias"}
MAX_FILES=${2:-0}
MAX_RETRIES=${3:-1}
CHECKSUMS_URL="https://vrg.fel.cvut.cz/ilias_data/checksums.txt"
CHECKSUMS_FILE="$DOWNLOAD_DIR/checksums.txt"

echo "Downloading files to: $DOWNLOAD_DIR"
[ "$MAX_FILES" -gt 0 ] && echo "Limit downloaded YFCC100M shards to $MAX_FILES."
echo "Using $MAX_RETRIES max retries per file."

# create main download directory if it does not exist
mkdir -p "$DOWNLOAD_DIR"

# download checksums.txt file
echo "Downloading checksums.txt from $CHECKSUMS_URL..."
wget -q -O "$CHECKSUMS_FILE" "$CHECKSUMS_URL" || { echo "ERROR: Failed to download files.txt"; exit 1; }
echo "Downloaded checksums.txt successfully."

downloaded_count=-1

# read each line from checksums.txt and process it sequentially
while read -r checksum url; do
    filename=$(basename "$url")

    # stop downloading if the limit is reached
    if [ "$MAX_FILES" -gt 0 ] && [ "$downloaded_count" -ge "$MAX_FILES" ]; then
        echo "Reached download limit of $MAX_FILES YFCC100M shards. Stop downloading."
        break
    fi

    # check if file is a yfcc100m file
    if [[ "$filename" == *"yfcc100m"* ]]; then
        subdir="$DOWNLOAD_DIR/yfcc100m"
    else
        subdir="$DOWNLOAD_DIR"
    fi

    mkdir -p "$subdir"

    file="$subdir/$filename"
    attempt=0

    while [ $attempt -lt $MAX_RETRIES ]; do
        # download file
        wget -q --show-progress -O "$file" "$url"

        # verify checksum
        result=$(echo "$checksum  $file" | sha256sum -c - 2>&1)

        if echo "$result" | grep -q ': OK'; then
            # check if file is ilias core
            if [[ "$filename" == *"ilias_core.tar" ]]; then
                echo "Extracting $file..."
                tar -xf "$file" -C "$subdir" && echo "Extracted: $file"
                rm "$file" && echo "Removed: $file"
            fi
            ((downloaded_count++))  # increment downloaded file count
            break
        else
            echo "Checksum failed for $file (attempt $((attempt + 1)))"
            ((attempt++))  # increment downloaded file count
            rm -f "$file"  # remove corrupted file
        fi
    done

    # if all attempts failed, report error
    if [ $attempt -eq $MAX_RETRIES ]; then
        echo "ERROR: Failed to download $file correctly after $MAX_RETRIES attempts."
    fi
done < $CHECKSUMS_FILE
