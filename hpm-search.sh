#!/bin/bash

# Directory setup
MNIST_PATH="/path/to/mnist"
BASE_OUTPUT_DIR="experiments/vit_hyperparameter_search"
mkdir -p "$BASE_OUTPUT_DIR"

# Log file setup
LOG_FILE="$BASE_OUTPUT_DIR/hyperparameter_search.log"
ERROR_LOG="$BASE_OUTPUT_DIR/failed_experiments.log"
echo "Starting hyperparameter search at $(date)" > "$LOG_FILE"
echo "Failed experiments log - $(date)" > "$ERROR_LOG"

# Hyperparameter search spaces - focused on learning rate and patch size
# Learning rates from 1e-6 to 1e-2 with more granularity
LEARNING_RATES=(1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2)
PATCH_SIZES=(4 7 14)

# Fixed hyperparameters
BATCH_SIZE=16
NUM_BLOCKS=2
NUM_HEADS=2
HIDDEN_DIM=8

# Counter for experiments
TOTAL_EXPERIMENTS=$((${#LEARNING_RATES[@]} * ${#PATCH_SIZES[@]}))
CURRENT_EXPERIMENT=0
FAILED_EXPERIMENTS=0

# Function to log command and results
log_experiment() {
    echo "Running experiment with parameters:" >> "$LOG_FILE"
    echo "$1" >> "$LOG_FILE"
    echo "Started at $(date)" >> "$LOG_FILE"
}

# Function to log failed experiments
log_failure() {
    local exp_name=$1
    local cmd=$2
    local error_msg=$3
    
    echo "----------------------------------------" >> "$ERROR_LOG"
    echo "Failed experiment: $exp_name" >> "$ERROR_LOG"
    echo "Command: $cmd" >> "$ERROR_LOG"
    echo "Error message: $error_msg" >> "$ERROR_LOG"
    echo "Failed at: $(date)" >> "$ERROR_LOG"
    echo "----------------------------------------" >> "$ERROR_LOG"
    
    ((FAILED_EXPERIMENTS++))
}

# Function to run experiment with error handling
run_experiment() {
    local cmd=$1
    local exp_name=$2
    
    ((CURRENT_EXPERIMENT++))
    echo "Progress: Experiment $CURRENT_EXPERIMENT of $TOTAL_EXPERIMENTS"
    echo "Running experiment: $exp_name"
    
    # Run the command and capture both stdout and stderr
    if output=$(eval "$cmd" 2>&1); then
        echo "Experiment completed successfully: $exp_name" >> "$LOG_FILE"
        echo "$output" >> "$LOG_FILE"
        return 0
    else
        log_failure "$exp_name" "$cmd" "$output"
        return 1
    fi
}

# Grid search through hyperparameters
for lr in "${LEARNING_RATES[@]}"; do
    for patch_size in "${PATCH_SIZES[@]}"; do
        # Create descriptive experiment name
        EXPERIMENT_NAME="lr${lr}_p${patch_size}"
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$EXPERIMENT_NAME"
        
        # Construct command with fixed hyperparameters
        CMD="python train_vit.py \
            --path_to_mnist $MNIST_PATH \
            --output_dir $OUTPUT_DIR \
            --batch_size $BATCH_SIZE \
            --epochs 10 \
            --learning_rate $lr \
            --hidden_dim $HIDDEN_DIM \
            --patch_size $patch_size \
            --num_heads $NUM_HEADS \
            --num_blocks $NUM_BLOCKS \
            --test_epoch_interval 1"
        
        # Log and run experiment with error handling
        log_experiment "$CMD"
        run_experiment "$CMD" "$EXPERIMENT_NAME"
        
        # Optional: add sleep to prevent GPU memory issues
        sleep 5
    done
done

# Print summary
echo "----------------------------------------" >> "$LOG_FILE"
echo "Hyperparameter search completed at $(date)" >> "$LOG_FILE"
echo "Total experiments: $TOTAL_EXPERIMENTS" >> "$LOG_FILE"
echo "Failed experiments: $FAILED_EXPERIMENTS" >> "$LOG_FILE"
echo "----------------------------------------" >> "$LOG_FILE"

# Also print to terminal
echo "----------------------------------------"
echo "Hyperparameter search completed!"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Failed experiments: $FAILED_EXPERIMENTS"
echo "See $ERROR_LOG for details on failed experiments"
echo "----------------------------------------"

# Analyze results (only for successful experiments)
echo "Best performing models:" >> "$LOG_FILE"
find "$BASE_OUTPUT_DIR" -name "training_history.json" -exec python - {} \; <<EOF
import json
import sys
from pathlib import Path

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
        max_acc = max(data['test_acc']) if data['test_acc'] else 0
        exp_name = Path(sys.argv[1]).parent.name
        print(f"Model: {exp_name}")
        print(f"Max accuracy: {max_acc:.4f}")
        print(f"Hyperparameters: {json.dumps(data['hyperparameters'], indent=2)}")
        print("---")
except Exception as e:
    print(f"Error analyzing {sys.argv[1]}: {str(e)}")
    print("---")
EOF