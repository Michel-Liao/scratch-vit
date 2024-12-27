#!/bin/bash

# Create results directory and log files
RESULTS_DIR="vit_cifar10_small_search_results"
mkdir -p $RESULTS_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$RESULTS_DIR/architecture_search_${TIMESTAMP}.log"
RESULTS_CSV="$RESULTS_DIR/results_${TIMESTAMP}.csv"
EXPERIMENT_LOGS_DIR="$RESULTS_DIR/experiment_logs_${TIMESTAMP}"
mkdir -p $EXPERIMENT_LOGS_DIR

# Write CSV header
echo "hidden_dim,num_heads,num_blocks,patch_size,learning_rate,test_accuracy,inference_time_ms,status" > $RESULTS_CSV

# Function to parse accuracy from log file
parse_accuracy() {
    local log_file=$1
    local accuracy=$(grep -o "Test Accuracy: [0-9.]\+" "$log_file" | tail -n1 | cut -d' ' -f3)
    echo "$accuracy"
}

# Function to calculate average inference time
calculate_inference_time() {
    local log_file=$1
    local start_times=($(grep "Testing:" "$log_file" | grep -o "[0-9.]\+s" | cut -d's' -f1))
    local end_times=($(grep "Test Loss:" "$log_file" | grep -o "[0-9.]\+s" | cut -d's' -f1))
    
    if [ ${#start_times[@]} -eq ${#end_times[@]} ] && [ ${#start_times[@]} -gt 0 ]; then
        local total_time=0
        local count=${#start_times[@]}
        for ((i=0; i<count; i++)); do
            local time_diff=$(echo "${end_times[$i]} - ${start_times[$i]}" | bc)
            total_time=$(echo "$total_time + $time_diff" | bc)
        done
        echo "scale=2; ($total_time / $count) * 1000" | bc
    else
        echo "NA"
    fi
}

# Function to generate random number in range
random_range() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}

# Common parameters
CIFAR_PATH="/home/michel/repos/scratch_vit/data/cifar-10/cifar-10-batches-py"  # Replace with actual path
EPOCHS=7
TEST_INTERVAL=1
NUM_TRIALS=20  # Number of random trials per learning rate

# Parameter ranges for random search
hidden_dim_min=64
hidden_dim_max=512
num_heads_min=2
num_heads_max=6
num_blocks_min=2
num_blocks_max=6
patch_sizes=(1 2 4)  # Discrete choices for patch size
learning_rates=(1e-3 1e-4)  # Grid search values

# Log start time and parameters
{
    echo "==================================================="
    echo "Vision Transformer Architecture Search for CIFAR-10"
    echo "Started at: $(date)"
    echo "==================================================="
    echo
    echo "Fixed Parameters:"
    echo "- Epochs: $EPOCHS"
    echo "- Test interval: $TEST_INTERVAL"
    echo "- Number of random trials per learning rate: $NUM_TRIALS"
    echo
    echo "Search Parameters:"
    echo "- Hidden dimension range: $hidden_dim_min to $hidden_dim_max"
    echo "- Number of heads range: $num_heads_min to $num_heads_max"
    echo "- Number of blocks range: $num_blocks_min to $num_blocks_max"
    echo "- Patch sizes: ${patch_sizes[*]}"
    echo "- Learning rates: ${learning_rates[*]}"
    echo
    echo "==================================================="
} | tee "$MAIN_LOG"

# Initialize arrays to store best results
best_accuracy=0
best_hidden_dim=""
best_heads=""
best_blocks=""
best_patch_size=""
best_learning_rate=""
best_inference_time=""

# Calculate total experiments
total_experiments=$((${#learning_rates[@]} * NUM_TRIALS))
current_experiment=0

# Run architecture search
for lr in "${learning_rates[@]}"; do
    for ((trial=1; trial<=NUM_TRIALS; trial++)); do
        current_experiment=$((current_experiment + 1))
        
        # Generate random architecture parameters
        hidden_dim=$(($(random_range $hidden_dim_min $hidden_dim_max) / 32 * 32))  # Round to nearest multiple of 32
        num_heads=$(random_range $num_heads_min $num_heads_max)
        num_blocks=$(random_range $num_blocks_min $num_blocks_max)
        patch_size=${patch_sizes[$((RANDOM % ${#patch_sizes[@]}))]}
        
        echo -e "\nExperiment $current_experiment/$total_experiments" | tee -a "$MAIN_LOG"
        echo "Configuration: hidden_dim=$hidden_dim, heads=$num_heads, blocks=$num_blocks, patch_size=$patch_size, lr=$lr" | tee -a "$MAIN_LOG"
        
        # Create experiment-specific log file
        EXPERIMENT_LOG="$EXPERIMENT_LOGS_DIR/experiment_dim${hidden_dim}_heads${num_heads}_blocks${num_blocks}_patch${patch_size}_lr${lr}.log"
        
        # Log start time of experiment
        {
            echo "==================================================="
            echo "Experiment Configuration:"
            echo "- Hidden Dimension: $hidden_dim"
            echo "- Number of Heads: $num_heads"
            echo "- Number of Blocks: $num_blocks"
            echo "- Patch Size: $patch_size"
            echo "- Learning Rate: $lr"
            echo "- Start Time: $(date)"
            echo "==================================================="
        } | tee "$EXPERIMENT_LOG"
        
        # Run the training script
        {
            time python train_cifar.py \
                --path_to_cifar "$CIFAR_PATH" \
                --batch_size 128 \
                --learning_rate "$lr" \
                --patch_size "$patch_size" \
                --num_heads "$num_heads" \
                --num_blocks "$num_blocks" \
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
                echo "$hidden_dim,$num_heads,$num_blocks,$patch_size,$lr,$accuracy,$inference_time,success" >> "$RESULTS_CSV"
                
                # Update best results if necessary
                if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
                    best_accuracy=$accuracy
                    best_hidden_dim=$hidden_dim
                    best_heads=$num_heads
                    best_blocks=$num_blocks
                    best_patch_size=$patch_size
                    best_learning_rate=$lr
                    best_inference_time=$inference_time
                fi
                
                echo "Success - Test Accuracy: $accuracy, Inference Time: ${inference_time}ms" | tee -a "$MAIN_LOG"
            else
                echo "$hidden_dim,$num_heads,$num_blocks,$patch_size,$lr,NA,$inference_time,failed_no_accuracy" >> "$RESULTS_CSV"
                echo "Failed - No accuracy found in output" | tee -a "$MAIN_LOG"
            fi
        else
            echo "$hidden_dim,$num_heads,$num_blocks,$patch_size,$lr,NA,NA,failed_error" >> "$RESULTS_CSV"
            echo "Failed - Error during execution" | tee -a "$MAIN_LOG"
        fi
        
        echo "---------------------------------------------------" | tee -a "$MAIN_LOG"
    done
done

# Log final results
{
    echo
    echo "==================================================="
    echo "Architecture Search Results Summary"
    echo "==================================================="
    echo "Best Configuration:"
    echo "- Hidden Dimension: $best_hidden_dim"
    echo "- Number of Heads: $best_heads"
    echo "- Number of Blocks: $best_blocks"
    echo "- Patch Size: $best_patch_size"
    echo "- Learning Rate: $best_learning_rate"
    echo "- Test Accuracy: $best_accuracy"
    echo "- Inference Time: ${best_inference_time}ms"
    echo
    echo "All Results (sorted by accuracy):"
    echo "---------------------------------------------------"
    sort -t',' -k6 -nr "$RESULTS_CSV" | column -t -s','
    echo
    echo "==================================================="
    echo "Architecture search completed at $(date)"
    echo "Logs saved in: $EXPERIMENT_LOGS_DIR"
    echo "Results saved in: $RESULTS_CSV"
    echo "==================================================="
} | tee -a "$MAIN_LOG"