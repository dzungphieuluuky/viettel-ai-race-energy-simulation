#!/bin/bash

# ====================================================================================
# === Focused RL Agent Training Loop for Private Leaderboard Submission ============
# ====================================================================================
# This script runs a training loop focused on three key scenarios.
#
# It accepts two optional command-line arguments:
# 1. Number of Epochs (e.g., 500)
# 2. Path to a specific model .pth file to continue training from.

# --- Configuration ---
# Use the first command-line argument ($1) for NUM_EPOCHS. Default to 100 if not provided.
NUM_EPOCHS=${1:-100}

# Use the second command-line argument ($2) for the specific model path.
MODEL_PATH=${2:-""}

MCR_ROOT="/opt/mcr/R2025a"

# --- Pre-run Checks and Information ---
echo "Starting focused training loop for $NUM_EPOCHS epochs."

if [ -n "$MODEL_PATH" ]; then
    # Check if the specified model file actually exists
    if [ -f "$MODEL_PATH" ]; then
        echo "--> Will resume training from specific model: $MODEL_PATH"
    else
        echo "!!! ERROR: Specified model file not found at: $MODEL_PATH !!!"
        echo "Aborting."
        exit 1
    fi
else
    echo "--> No model specified. The agent will load the latest model automatically."
fi

echo "Scenario Probabilities: Extreme Rural (50%), Highway (30%), Urban Macro (20%)"
echo "================================================================================"

# --- Main Training Loop ---
for (( i=1; i<=$NUM_EPOCHS; i++ ))
do
    # --- Weighted Random Scenario Selection ---
    # Generate a random number between 0 and 99.
    RANDOM_NUM=$((RANDOM % 100))
    SCENARIO=""

    # Probabilities: extreme_rural(50%), highway(30%), urban_macro(20%)
    if [ $RANDOM_NUM -lt 50 ]; then          # 0-49 (50%)
        SCENARIO="extreme_rural"
    elif [ $RANDOM_NUM -lt 80 ]; then        # 50-79 (30%)
        SCENARIO="highway"
    else                                     # 80-99 (20%)
        SCENARIO="urban_macro"
    fi

    echo -e "\n--- Starting Epoch $i of $NUM_EPOCHS with scenario: $SCENARIO ---"

    # --- Pre-simulation model selection ---
    # If a model path was provided, update its timestamp to make it the 'latest' file.
    # The Python agent will then automatically load it.
    if [ -n "$MODEL_PATH" ]; then
        echo "Touching model file to mark it as latest: $MODEL_PATH"
        touch "$MODEL_PATH"
    fi

    # --- Execute Simulation and Capture Exit Code ---
    # The command is the same; the 'touch' trick handles the model loading.
    ./run_runSimulationWithAnimation.sh "$MCR_ROOT" "$SCENARIO" 2> /dev/null
    EXIT_CODE=$? # Immediately save the exit code of the last command

    # Check the exit code to decide what to do
    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! WARNING: Epoch $i with scenario '$SCENARIO' finished with a non-zero exit code ($EXIT_CODE). !!!"
        echo "Assuming this is a non-critical MATLAB crash. Continuing training."
    else
        echo "--- Epoch $i completed successfully ---"
    fi
done

echo -e "\n================================================================================"
echo "Training loop finished after $NUM_EPOCHS epochs."