#!/bin/bash

# Create results directory and log files
RESULTS_DIR="vit_grid_search_results"
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$RESULTS_DIR/grid_search_${TIMESTAMP}.log"
RESULTS_CSV="$RESULTS_DIR/results_${TIMESTAMP}.csv"
EXPERIMENT_LOGS_DIR="$RESULTS_DIR/experiment_logs_${TIMESTAMP}"
mkdir -p $EXPERIMENT_LOGS_DIR

# Write CSV header
echo "learning_rate,patch_size,test_accuracy,inference_time_ms,status" > $RESULTS_CSV

# Function to parse accuracy from log file
parse_accuracy() {
    local log_file=$1
    local accuracy=$(grep -o "Test Accuracy: [0-9.]\+" "$log_file" | tail -n1 | cut -d' ' -f3)
    echo "$accuracy"
}

# Function to calculate average inference time
calculate_inference_time() {
    local log_file=$1
    # Extract all lines containing "Testing:" and the timestamp
    local start_times=($(grep "Testing:" "$log_file" | grep -o "[0-9.]\+s" | cut -d's' -f1))
    local end_times=($(grep "Test Loss:" "$log_file" | grep -o "[0-9.]\+s" | cut -d's' -f1))
    
    # Calculate average inference time if we have matching start and end times
    if [ ${#start_times[@]} -eq ${#end_times[@]} ] && [ ${#start_times[@]} -gt 0 ]; then
        local total_time=0
        local count=${#start_times[@]}
        for ((i=0; i<count; i++)); do
            local time_diff=$(echo "${end_times[$i]} - ${start_times[$i]}" | bc)
            total_time=$(echo "$total_time + $time_diff" | bc)
        done
        # Convert to milliseconds and calculate average
        echo "scale=2; ($total_time / $count) * 1000" | bc
    else
        echo "NA"
    fi
}

# Common parameters
MNIST_PATH="/home/michel/repos/scratch_vit/data"  # Replace with actual path
NUM_HEADS=4
NUM_BLOCKS=3
HIDDEN_DIM=384
EPOCHS=5
TEST_INTERVAL=1

# Learning rates to test
learning_rates=(1e-6 1e-7 1e-8 1e-9)
patch_sizes=(4 7 14)

# Log start time and parameters
{
    echo "==================================================="
    echo "Vision Transformer Grid Search"
    echo "Started at: $(date)"
    echo "==================================================="
    echo
    echo "Fixed Parameters:"
    echo "- Number of heads: $NUM_HEADS"
    echo "- Number of blocks: $NUM_BLOCKS"
    echo "- Hidden dimension: $HIDDEN_DIM"
    echo "- Epochs: $EPOCHS"
    echo "- Test interval: $TEST_INTERVAL"
    echo
    echo "Grid Search Parameters:"
    echo "- Learning rates: ${learning_rates[*]}"
    echo "- Patch sizes: ${patch_sizes[*]}"
    echo
    echo "==================================================="
} | tee "$MAIN_LOG"

# Initialize arrays to store best results
best_accuracy=0
best_lr=""
best_patch_size=""
best_inference_time=""

# Run grid search
total_experiments=${#learning_rates[@]}*${#patch_sizes[@]}
current_experiment=0

for lr in "${learning_rates[@]}"; do
    for patch_size in "${patch_sizes[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo -e "\nExperiment $current_experiment/$total_experiments (lr=$lr, patch_size=$patch_size)" | tee -a "$MAIN_LOG"
        
        # Create experiment-specific log file
        EXPERIMENT_LOG="$EXPERIMENT_LOGS_DIR/experiment_lr${lr}_patch${patch_size}.log"
        
        # Log start time of experiment
        {
            echo "==================================================="
            echo "Experiment Configuration:"
            echo "- Learning Rate: $lr"
            echo "- Patch Size: $patch_size"
            echo "- Start Time: $(date)"
            echo "==================================================="
        } | tee "$EXPERIMENT_LOG"
        
        # Run the training script
        {
            time python train_new.py \
                --path_to_mnist "$MNIST_PATH" \
                --batch_size 64 \
                --learning_rate "$lr" \
                --patch_size "$patch_size" \
                --num_heads "$NUM_HEADS" \
                --num_blocks "$NUM_BLOCKS" \
                --hidden_dim "$HIDDEN_DIM" \
                --epochs "$EPOCHS" \
                --test_epoch_interval "$TEST_INTERVAL" \
                --init_method "he"
        } 2>&1 | tee -a "$EXPERIMENT_LOG"
        
        # Check if the run was successful
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            # Extract metrics
            accuracy=$(parse_accuracy "$EXPERIMENT_LOG")
            inference_time=$(calculate_inference_time "$EXPERIMENT_LOG")
            
            if [ -n "$accuracy" ]; then
                echo "$lr,$patch_size,$accuracy,$inference_time,success" >> "$RESULTS_CSV"
                
                # Update best results if necessary
                if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                    best_accuracy=$accuracy
                    best_lr=$lr
                    best_patch_size=$patch_size
                    best_inference_time=$inference_time
                fi
                
                echo "Success - Test Accuracy: $accuracy, Inference Time: ${inference_time}ms" | tee -a "$MAIN_LOG"
            else
                echo "$lr,$patch_size,NA,$inference_time,failed_no_accuracy" >> "$RESULTS_CSV"
                echo "Failed - No accuracy found in output" | tee -a "$MAIN_LOG"
            fi
        else
            echo "$lr,$patch_size,NA,NA,failed_error" >> "$RESULTS_CSV"
            echo "Failed - Error during execution" | tee -a "$MAIN_LOG"
        fi
        
        echo "---------------------------------------------------" | tee -a "$MAIN_LOG"
    done
done

# Log final results
{
    echo
    echo "==================================================="
    echo "Grid Search Results Summary"
    echo "==================================================="
    echo "Best Configuration:"
    echo "- Learning Rate: $best_lr"
    echo "- Patch Size: $best_patch_size"
    echo "- Test Accuracy: $best_accuracy"
    echo "- Inference Time: ${best_inference_time}ms"
    echo
    echo "All Results (sorted by accuracy):"
    echo "---------------------------------------------------"
    sort -t',' -k3 -nr "$RESULTS_CSV" | column -t -s','
    echo
    echo "==================================================="
    echo "Grid search completed at $(date)"
    echo "Logs saved in: $EXPERIMENT_LOGS_DIR"
    echo "Results saved in: $RESULTS_CSV"
    echo "==================================================="
} | tee -a "$MAIN_LOG"

# Create visualization
cat << EOF > plot_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read results
df = pd.read_csv("$RESULTS_CSV")

# Convert learning_rate to numeric and create readable format
df['learning_rate'] = pd.to_numeric(df['learning_rate'])
df['lr_exp'] = df['learning_rate'].apply(lambda x: f'1e{int(np.log10(x))}')

# Create accuracy heatmap
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
pivot_acc = df.pivot(index='patch_size', columns='lr_exp', values='test_accuracy')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='viridis')
plt.title('Grid Search Results: Test Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Patch Size')

# Create inference time heatmap
plt.subplot(2, 1, 2)
pivot_time = df.pivot(index='patch_size', columns='lr_exp', values='inference_time_ms')
sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='rocket_r')
plt.title('Grid Search Results: Inference Time (ms)')
plt.xlabel('Learning Rate')
plt.ylabel('Patch Size')

plt.tight_layout()
plt.savefig("$RESULTS_DIR/grid_search_results.png")
EOF

python plot_results.py