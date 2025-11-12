#!/bin/bash\n

# ==============================================================================
# === Simplified RL Agent Training Loop ========================================
# ==============================================================================
# This script runs the simulation in a simple loop.
# The Python agent (rl_agent.py) is now responsible for automatically
# loading the latest model and its experience buffer to continue training.

# --- Configuration ---
NUM_EPOCHS=100
MCR_ROOT="/opt/mcr/R2025a"

echo "Starting simplified training loop for $NUM_EPOCHS epochs."
echo "The Python agent will handle all model and buffer loading."
echo "======================================================="

# --- Main Training Loop ---
for (( i=1; i<=$NUM_EPOCHS; i++ ))
do
    echo -e "\n--- Starting Epoch $i of $NUM_EPOCHS ---"

    # We no longer need to pass any model arguments. The agent handles it.
    ./run_main_run_scenarios.sh "$MCR_ROOT"

    if [ $? -ne 0 ]; then
        echo "!!! Simulation run failed on epoch $i. Aborting training loop. !!!"
        exit 1
    fi

    echo "--- Epoch $i completed ---"
done

echo -e "\n======================================================="
echo "Training loop finished after $NUM_EPOCHS epochs."