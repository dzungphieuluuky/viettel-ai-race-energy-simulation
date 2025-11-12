#!/bin/bash

# --- 1. Find the latest ZIP file (which is the basis for the JSON file)
LATEST_ZIP=$(ls -t ppo_*.zip 2>/dev/null | head -n 1)

if [ -z "$LATEST_ZIP" ]; then
    echo "No ppo_*.zip files found to determine the latest model. Exiting."
    exit 0
fi

# --- 2. Calculate the base name for the JSON file
BASENAME=${LATEST_ZIP/.zip/}

echo "Keeping: $LATEST_ZIP and ${BASENAME}_metadata.json"
echo "---"
echo "Removing all other files..."

# --- 3. Use find to delete all files that DO NOT match the latest two files
# -maxdepth 1 limits the search to the current directory
# -type f ensures only files are targeted (not directories)
# ! -name excludes the files we want to keep
# -delete removes them directly (fastest method)
find . -maxdepth 1 -type f \
    ! -name "$LATEST_ZIP" \
    ! -name "${BASENAME}_metadata.json" \
    ! -name "rollout_buffer.pkl" \
    -delete

echo "Cleanup complete."