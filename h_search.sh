#!/bin/bash

# Create results directory and log files
RESULTS_DIR="vit_architecture_search_results"
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$RESULTS_DIR/architecture_search_${TIMESTAMP}.log"
RESULTS_CSV="$RESULTS_DIR/results_${TIMESTAMP}.csv"
EXPERIMENT_LOGS_DIR="$RESULTS_DIR/experiment_logs_${TIMESTAMP}"
mkdir -p $EXPERIMENT_LOGS_DIR

# Write CSV header
echo "num_heads,num_blocks,hidden_dim,test_accuracy,inference_time_ms,status" > $RESULTS_CSV

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
LEARNING_RATE=1e-4
PATCH_SIZE=4
EPOCHS=5
TEST_INTERVAL=1

# Architecture parameters to test
num_heads=(4 8 12)
num_blocks=(4 8 12)
hidden_dims=(256 512 768)

# Log start time and parameters
{
    echo "==================================================="
    echo "Vision Transformer Architecture Search"
    echo "Started at: $(date)"
    echo "==================================================="
    echo
    echo "Fixed Parameters:"
    echo "- Learning rate: $LEARNING_RATE"
    echo "- Patch size: $PATCH_SIZE"
    echo "- Epochs: $EPOCHS"
    echo "- Test interval: $TEST_INTERVAL"
    echo
    echo "Search Parameters:"
    echo "- Number of heads: ${num_heads[*]}"
    echo "- Number of blocks: ${num_blocks[*]}"
    echo "- Hidden dimensions: ${hidden_dims[*]}"
    echo
    echo "==================================================="
} | tee "$MAIN_LOG"

# Initialize arrays to store best results
best_accuracy=0
best_heads=""
best_blocks=""
best_hidden_dim=""
best_inference_time=""

# Run architecture search
total_experiments=${#num_heads[@]}*${#num_blocks[@]}*${#hidden_dims[@]}
current_experiment=0

for hidden_dim in "${hidden_dims[@]}"; do
    for heads in "${num_heads[@]}"; do
        for blocks in "${num_blocks[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo -e "\nExperiment $current_experiment/$total_experiments (heads=$heads, blocks=$blocks, hidden_dim=$hidden_dim)" | tee -a "$MAIN_LOG"
            
            # Create experiment-specific log file
            EXPERIMENT_LOG="$EXPERIMENT_LOGS_DIR/experiment_heads${heads}_blocks${blocks}_dim${hidden_dim}.log"
            
            # Log start time of experiment
            {
                echo "==================================================="
                echo "Experiment Configuration:"
                echo "- Number of Heads: $heads"
                echo "- Number of Blocks: $blocks"
                echo "- Hidden Dimension: $hidden_dim"
                echo "- Start Time: $(date)"
                echo "==================================================="
            } | tee "$EXPERIMENT_LOG"
            
            # Run the training script
            {
                time python train_new.py \
                    --path_to_mnist "$MNIST_PATH" \
                    --batch_size 64 \
                    --learning_rate "$LEARNING_RATE" \
                    --patch_size "$PATCH_SIZE" \
                    --num_heads "$heads" \
                    --num_blocks "$blocks" \
                    --hidden_dim "$hidden_dim" \
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
                    echo "$heads,$blocks,$hidden_dim,$accuracy,$inference_time,success" >> "$RESULTS_CSV"
                    
                    # Update best results if necessary
                    if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                        best_accuracy=$accuracy
                        best_heads=$heads
                        best_blocks=$blocks
                        best_hidden_dim=$hidden_dim
                        best_inference_time=$inference_time
                    fi
                    
                    echo "Success - Test Accuracy: $accuracy, Inference Time: ${inference_time}ms" | tee -a "$MAIN_LOG"
                else
                    echo "$heads,$blocks,$hidden_dim,NA,$inference_time,failed_no_accuracy" >> "$RESULTS_CSV"
                    echo "Failed - No accuracy found in output" | tee -a "$MAIN_LOG"
                fi
            else
                echo "$heads,$blocks,$hidden_dim,NA,NA,failed_error" >> "$RESULTS_CSV"
                echo "Failed - Error during execution" | tee -a "$MAIN_LOG"
            fi
            
            echo "---------------------------------------------------" | tee -a "$MAIN_LOG"
        done
    done
done

# Log final results
{
    echo
    echo "==================================================="
    echo "Architecture Search Results Summary"
    echo "==================================================="
    echo "Best Configuration:"
    echo "- Number of Heads: $best_heads"
    echo "- Number of Blocks: $best_blocks"
    echo "- Hidden Dimension: $best_hidden_dim"
    echo "- Test Accuracy: $best_accuracy"
    echo "- Inference Time: ${best_inference_time}ms"
    echo
    echo "All Results (sorted by accuracy):"
    echo "---------------------------------------------------"
    sort -t',' -k4 -nr "$RESULTS_CSV" | column -t -s','
    echo
    echo "==================================================="
    echo "Architecture search completed at $(date)"
    echo "Logs saved in: $EXPERIMENT_LOGS_DIR"
    echo "Results saved in: $RESULTS_CSV"
    echo "==================================================="
} | tee -a "$MAIN_LOG"