#!/bin/bash
#
# Ollama GPU Tuner Script
# This script systematically tests different Ollama GPU configurations
# to find optimal settings for your NVIDIA GPU.
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RESULTS_DIR="./ollama_gpu_tests"
CONFIG_DIR="/etc/ollama/config"
LOG_FILE="$RESULTS_DIR/gpu_tuning_results.log"
CSV_FILE="$RESULTS_DIR/gpu_test_results.csv"
RECOVERY_TIME=10  # Time in seconds to wait between tests for GPU to recover

# Test models array (from smallest to largest)
# Format: "model_name|prompt|description"
MODELS=(
  "tinyllama:latest|Hello|Smallest model for basic testing"
  "phi:latest|Hello|Small but capable model"
  "llama2:7b|Hello|Medium-sized model"
  "deepseek-coder:6.7b-instruct|Write a hello world function|Coding-specific model"
)

# Default model index (0 = first model in the array)
MODEL_INDEX=0

# Print header
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}   Ollama GPU Configuration Tuner                        ${NC}"
echo -e "${GREEN}   For NVIDIA GeForce GTX 1080                           ${NC}"
echo -e "${GREEN}=========================================================${NC}"
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      MODEL_INDEX="$2"
      shift
      shift
      ;;
    --list-models)
      echo -e "${BLUE}Available test models:${NC}"
      for i in "${!MODELS[@]}"; do
        IFS='|' read -r model prompt description <<< "${MODELS[$i]}"
        echo -e "  $i: $model - $description"
      done
      exit 0
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --model <index>     Use the model at the specified index (default: 0)"
      echo "  --list-models       List available test models"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $key${NC}"
      exit 1
      ;;
  esac
done

# Check if running as root
if [[ $EUID -ne 0 ]]; then
  echo -e "${RED}Please run as root or with sudo${NC}"
  exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$CONFIG_DIR"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if GPU is connected and functioning
check_gpu_connected() {
  if ! nvidia-smi &>/dev/null; then
    echo -e "\n${RED}ERROR: GPU DISCONNECTED FROM SYSTEM${NC}"
    echo -e "${RED}The NVIDIA GPU appears to have disconnected or crashed.${NC}"
    echo -e "${RED}This is likely a hardware or driver stability issue.${NC}"
    echo -e "${RED}Please restart your system before continuing.${NC}"

    # Log the failure
    echo "GPU DISCONNECTED AT $(date)" >> "$LOG_FILE"
    echo "Test aborted - system restart required" >> "$LOG_FILE"

    exit 3  # Special exit code for GPU disconnection
  fi
}

# Function to capture GPU metrics
capture_gpu_metrics() {
  local label="$1"
  local config_name="$2"

  # Print full nvidia-smi output
  echo -e "${BLUE}Full nvidia-smi output:${NC}"
  nvidia-smi
  echo ""

  # Check for Ollama processes
  echo -e "${BLUE}Checking for Ollama processes:${NC}"
  ps aux | grep ollama | grep -v grep
  echo ""

  # Get GPU temperature
  local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

  # Get GPU utilization
  local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

  # Get GPU memory usage
  local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  local mem_percent=$(awk "BEGIN {print ($mem_used/$mem_total)*100}")

  # Get GPU power usage
  local power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)

  # Get GPU clock speeds
  local gpu_clock=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits)
  local mem_clock=$(nvidia-smi --query-gpu=clocks.current.memory --format=csv,noheader,nounits)

  # Get timestamp
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")

  # Write to CSV
  echo "$timestamp,$label,$config_name,$temp,$gpu_util,$mem_used,$mem_total,$mem_percent,$power,$gpu_clock,$mem_clock" >> "$CSV_FILE"

  # Print metrics
  echo -e "${BLUE}GPU Metrics ($label - $config_name):${NC}"
  echo -e "  Temperature:   ${temp}°C"
  echo -e "  Utilization:   ${gpu_util}%"
  echo -e "  Memory Usage:  ${mem_used}MB / ${mem_total}MB (${mem_percent}%)"
  echo -e "  Power Draw:    ${power}W"
  echo -e "  GPU Clock:     ${gpu_clock}MHz"
  echo -e "  Memory Clock:  ${mem_clock}MHz"
}

# Check for NVIDIA GPU
echo -e "${BLUE}Checking for NVIDIA GPU...${NC}"
if ! command_exists nvidia-smi; then
  echo -e "${RED}NVIDIA drivers not installed. Cannot configure GPU mode.${NC}"
  exit 1
fi

# Check if the GPU is a GTX 1080
GPU_INFO=$(nvidia-smi -q | grep "Product Name")
if [[ ! "$GPU_INFO" =~ "GeForce GTX 1080" ]]; then
  echo -e "${YELLOW}Warning: This script is optimized for NVIDIA GeForce GTX 1080.${NC}"
  echo -e "${YELLOW}Detected GPU: $GPU_INFO${NC}"
  echo -e "${YELLOW}Continuing with generic GPU configuration...${NC}"
else
  echo -e "${GREEN}NVIDIA GeForce GTX 1080 detected.${NC}"
fi

# Get GPU memory information
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
echo -e "${BLUE}Total VRAM: ${TOTAL_VRAM}MB${NC}"

# Check if Ollama is installed
echo -e "${BLUE}Checking Ollama installation...${NC}"
if ! command_exists ollama; then
  echo -e "${RED}Ollama is not installed. Please install Ollama first.${NC}"
  exit 1
else
  echo -e "${GREEN}Ollama is installed:${NC}"
  ollama --version
fi

# Get the selected model information
if [[ $MODEL_INDEX -ge ${#MODELS[@]} ]]; then
  echo -e "${RED}Invalid model index: $MODEL_INDEX. Use --list-models to see available models.${NC}"
  exit 1
fi

IFS='|' read -r TEST_MODEL TEST_PROMPT TEST_DESCRIPTION <<< "${MODELS[$MODEL_INDEX]}"

echo -e "${BLUE}Selected test model: ${TEST_MODEL} - ${TEST_DESCRIPTION}${NC}"
echo -e "${BLUE}Test prompt: ${TEST_PROMPT}${NC}"

# Check if test model is available
echo -e "${BLUE}Checking for test model ${TEST_MODEL}...${NC}"
if ! ollama list | grep -q "$TEST_MODEL"; then
  echo -e "${YELLOW}Test model $TEST_MODEL not found. Pulling it now...${NC}"
  ollama pull "$TEST_MODEL"
fi

# Initialize log file
echo "Ollama GPU Tuning Results - $(date)" > "$LOG_FILE"
echo "GPU: $(nvidia-smi -q | grep "Product Name" | cut -d ":" -f2 | xargs)" >> "$LOG_FILE"
echo "Total VRAM: ${TOTAL_VRAM}MB" >> "$LOG_FILE"
echo "Test Model: $TEST_MODEL ($TEST_DESCRIPTION)" >> "$LOG_FILE"
echo "Test Prompt: $TEST_PROMPT" >> "$LOG_FILE"
echo "=======================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Initialize CSV file with headers
echo "Timestamp,Stage,Configuration,Temperature,GPU_Utilization,Memory_Used_MB,Memory_Total_MB,Memory_Percent,Power_Draw_W,GPU_Clock_MHz,Memory_Clock_MHz,Duration_s,Tokens_per_Second,Success" > "$CSV_FILE"

# Capture baseline GPU metrics
capture_gpu_metrics "Baseline" "None"

# Function to test a configuration
test_configuration() {
  local config_name="$1"
  local gpu_overhead="$2"
  local max_vram="$3"
  local flash_attention="$4"
  local sched_spread="$5"
  local num_parallel="$6"
  local context_length="$7"
  local llm_library="$8"

  echo -e "\n${YELLOW}Testing configuration: $config_name${NC}"
  echo -e "${BLUE}Settings:${NC}"
  echo -e "  GPU Overhead:     ${gpu_overhead}MB"
  echo -e "  Max VRAM:         ${max_vram:-(all available)}MB"
  echo -e "  Flash Attention:  $flash_attention"
  echo -e "  Scheduler Spread: $sched_spread"
  echo -e "  Parallel Requests:$num_parallel"
  echo -e "  Context Length:   $context_length"
  echo -e "  LLM Library:      ${llm_library:-(auto)}"

  # Create systemd override directory
  mkdir -p /etc/systemd/system/ollama.service.d/

  # Build the configuration
  CONFIG="[Service]\nEnvironment=\"CUDA_VISIBLE_DEVICES=0\"\nEnvironment=\"OLLAMA_USE_GPU=1\"\nEnvironment=\"OLLAMA_GPU_OVERHEAD=${gpu_overhead}\"\nEnvironment=\"OLLAMA_CONTEXT_LENGTH=${context_length}\"\nEnvironment=\"OLLAMA_FLASH_ATTENTION=${flash_attention}\"\nEnvironment=\"OLLAMA_SCHED_SPREAD=${sched_spread}\"\nEnvironment=\"OLLAMA_NUM_PARALLEL=${num_parallel}\""

  # Add MAX_VRAM if specified
  if [[ -n "$max_vram" ]]; then
    CONFIG="${CONFIG}\nEnvironment=\"OLLAMA_MAX_VRAM=${max_vram}\""
  fi

  # Add LLM library if specified
  if [[ -n "$llm_library" ]]; then
    CONFIG="${CONFIG}\nEnvironment=\"OLLAMA_LLM_LIBRARY=${llm_library}\""
  fi

  # Write the configuration
  echo -e "$CONFIG" > /etc/systemd/system/ollama.service.d/override.conf

  # Restart Ollama service
  echo -e "${BLUE}Restarting Ollama service...${NC}"
  systemctl daemon-reload
  systemctl restart ollama

  # Wait for Ollama to start
  echo -e "${YELLOW}Waiting for Ollama service to start...${NC}"
  sleep 5

  # Check if Ollama is running
  if ! systemctl is-active --quiet ollama; then
    echo -e "${RED}Ollama service failed to start with this configuration.${NC}"

    # Get Ollama service logs
    echo -e "${YELLOW}Ollama service logs:${NC}"
    journalctl -u ollama -n 20

    echo "Configuration: $config_name - FAILED (service did not start)" >> "$LOG_FILE"
    echo "Settings: GPU_OVERHEAD=$gpu_overhead, MAX_VRAM=$max_vram, FLASH_ATTENTION=$flash_attention, SCHED_SPREAD=$sched_spread, NUM_PARALLEL=$num_parallel, CONTEXT_LENGTH=$context_length, LLM_LIBRARY=$llm_library" >> "$LOG_FILE"
    echo "Error: Service failed to start" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    return 1
  fi

  # Run the test
  echo -e "${BLUE}Running test query...${NC}"
  local temp_file=$(mktemp)

  # Check GPU is still connected before test
  check_gpu_connected

  # Capture GPU metrics before test
  capture_gpu_metrics "Before" "$config_name"

  # Record test start time
  local test_start_time=$(date +%s)

  if timeout 60 ollama run "$TEST_MODEL" "$TEST_PROMPT" --verbose > "$temp_file" 2>&1; then
    # Record test end time
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))

    # Check GPU is still connected after test
    check_gpu_connected

    # Capture GPU metrics after successful test
    capture_gpu_metrics "After-Success" "$config_name"

    echo -e "${GREEN}Test successful!${NC}"

    # Extract performance metrics
    local duration=$(grep "total duration" "$temp_file" | awk '{print $3}')
    local eval_rate=$(grep "eval rate" "$temp_file" | awk '{print $3}')

    # Convert duration from nanoseconds to seconds if needed
    if [[ "$duration" == *"ns"* ]]; then
      duration=$(echo "$duration" | sed 's/ns//' | awk '{print $1 / 1000000000}')
    fi

    # Add performance metrics to CSV
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$timestamp,Results,$config_name,,,,,,,,,${duration},${eval_rate},true" >> "$CSV_FILE"

    echo "Configuration: $config_name - SUCCESS" >> "$LOG_FILE"
    echo "Settings: GPU_OVERHEAD=$gpu_overhead, MAX_VRAM=$max_vram, FLASH_ATTENTION=$flash_attention, SCHED_SPREAD=$sched_spread, NUM_PARALLEL=$num_parallel, CONTEXT_LENGTH=$context_length, LLM_LIBRARY=$llm_library" >> "$LOG_FILE"
    echo "Performance: Duration=$duration, Eval Rate=$eval_rate tokens/s, Test Duration=${test_duration}s" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    # Save successful configuration
    echo -e "$CONFIG" > "$CONFIG_DIR/${config_name}.conf"
    echo -e "${GREEN}Configuration saved to: $CONFIG_DIR/${config_name}.conf${NC}"

    cat "$temp_file"
    rm "$temp_file"

    # Wait a moment before next test to let GPU recover
    echo -e "${YELLOW}Waiting 10 seconds before next test to let GPU recover...${NC}"
    sleep 10

    return 0
  else
    # Record test end time
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))

    # Check if GPU is still connected
    if ! nvidia-smi &>/dev/null; then
      echo -e "\n${RED}ERROR: GPU DISCONNECTED DURING TEST${NC}"
      echo -e "${RED}The NVIDIA GPU appears to have disconnected or crashed.${NC}"
      echo -e "${RED}This is likely a hardware or driver stability issue.${NC}"
      echo -e "${RED}Please restart your system before continuing.${NC}"

      # Log the failure
      echo "Configuration: $config_name - CAUSED GPU DISCONNECT" >> "$LOG_FILE"
      echo "Settings: GPU_OVERHEAD=$gpu_overhead, MAX_VRAM=$max_vram, FLASH_ATTENTION=$flash_attention, SCHED_SPREAD=$sched_spread, NUM_PARALLEL=$num_parallel, CONTEXT_LENGTH=$context_length, LLM_LIBRARY=$llm_library" >> "$LOG_FILE"
      echo "Error: GPU DISCONNECTED - SYSTEM RESTART REQUIRED" >> "$LOG_FILE"
      echo "Test Duration: ${test_duration}s" >> "$LOG_FILE"
      echo "" >> "$LOG_FILE"

      # Add disconnect event to CSV
      local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
      echo "$timestamp,Disconnect,$config_name,,,,,,,,,${test_duration},,false" >> "$CSV_FILE"

      rm "$temp_file"
      exit 3  # Special exit code for GPU disconnection
    fi

    # Capture GPU metrics after failed test
    capture_gpu_metrics "After-Failure" "$config_name"

    echo -e "${RED}Test failed!${NC}"

    # Extract error message
    local error_msg=$(grep "Error" "$temp_file" | head -1)

    # Get Ollama service logs
    echo -e "${YELLOW}Ollama service logs:${NC}"
    journalctl -u ollama -n 20

    # Check if there are any core dumps
    echo -e "${YELLOW}Checking for core dumps:${NC}"
    ls -la /var/crash/

    # Add failure metrics to CSV
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$timestamp,Results,$config_name,,,,,,,,,${test_duration},,false" >> "$CSV_FILE"

    echo "Configuration: $config_name - FAILED" >> "$LOG_FILE"
    echo "Settings: GPU_OVERHEAD=$gpu_overhead, MAX_VRAM=$max_vram, FLASH_ATTENTION=$flash_attention, SCHED_SPREAD=$sched_spread, NUM_PARALLEL=$num_parallel, CONTEXT_LENGTH=$context_length, LLM_LIBRARY=$llm_library" >> "$LOG_FILE"
    echo "Error: $error_msg" >> "$LOG_FILE"
    echo "Test Duration: ${test_duration}s" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    cat "$temp_file"
    rm "$temp_file"

    # Wait a moment before next test to let GPU recover
    echo -e "${YELLOW}Waiting 10 seconds before next test to let GPU recover...${NC}"
    sleep 10

    return 1
  fi
}

# Initial GPU check
check_gpu_connected

# Test different configurations
echo -e "\n${GREEN}Starting GPU configuration tests...${NC}"
echo -e "${YELLOW}Results will be logged to $LOG_FILE${NC}"
echo -e "${YELLOW}If the GPU disconnects during testing, you will need to restart your system${NC}"

# Function to test CPU-only mode
test_cpu_mode() {
  echo -e "\n${YELLOW}Testing CPU-only mode for comparison...${NC}"
  # Create CPU-only override
  echo -e "${YELLOW}Creating CPU-only configuration...${NC}"
  mkdir -p /etc/systemd/system/ollama.service.d/
  cat > /etc/systemd/system/ollama.service.d/override.conf << EOF
[Service]
Environment="CUDA_VISIBLE_DEVICES=-1"
Environment="OLLAMA_USE_GPU=0"
EOF

  # Restart Ollama service
  echo -e "${BLUE}Restarting Ollama service in CPU mode...${NC}"
  systemctl daemon-reload
  systemctl restart ollama

  # Wait for Ollama to start
  echo -e "${YELLOW}Waiting for Ollama service to start...${NC}"
  sleep 5

  # Check if Ollama is running
  if ! systemctl is-active --quiet ollama; then
    echo -e "${RED}Ollama service failed to start in CPU mode.${NC}"
    journalctl -u ollama -n 20
    exit 1
  fi

  # Run the test
  echo -e "${BLUE}Running CPU-only test query...${NC}"
  local temp_file=$(mktemp)

  # Capture metrics before test
  capture_gpu_metrics "Before-CPU" "cpu_only"

  # Record test start time
  local test_start_time=$(date +%s)

  if timeout 60 ollama run "$TEST_MODEL" "$TEST_PROMPT" --verbose > "$temp_file" 2>&1; then
    # Record test end time
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))

    # Capture metrics after successful test
    capture_gpu_metrics "After-CPU-Success" "cpu_only"

    echo -e "${GREEN}CPU test successful!${NC}"

    # Extract performance metrics
    local duration=$(grep "total duration" "$temp_file" | awk '{print $3}')
    local eval_rate=$(grep "eval rate" "$temp_file" | awk '{print $3}')

    # Add performance metrics to CSV
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$timestamp,Results,cpu_only,,,,,,,,,${test_duration},${eval_rate},true" >> "$CSV_FILE"

    echo "Configuration: cpu_only - SUCCESS" >> "$LOG_FILE"
    echo "Performance: Duration=$duration, Eval Rate=$eval_rate tokens/s, Test Duration=${test_duration}s" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    cat "$temp_file"
    rm "$temp_file"
    return 0
  else
    # Record test end time
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))

    # Capture metrics after failed test
    capture_gpu_metrics "After-CPU-Failure" "cpu_only"

    echo -e "${RED}CPU test failed!${NC}"

    # Get Ollama service logs
    echo -e "${YELLOW}Ollama service logs:${NC}"
    journalctl -u ollama -n 20

    # Add failure metrics to CSV
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$timestamp,Results,cpu_only,,,,,,,,,${test_duration},,false" >> "$CSV_FILE"

    echo "Configuration: cpu_only - FAILED" >> "$LOG_FILE"
    echo "Error: $(grep "Error" "$temp_file" | head -1)" >> "$LOG_FILE"
    echo "Test Duration: ${test_duration}s" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"

    cat "$temp_file"
    rm "$temp_file"
    return 1
  fi
}

# Test 0: CPU-only mode (for comparison)
test_cpu_mode

# Wait a moment before next test
echo -e "${YELLOW}Waiting 10 seconds before next test...${NC}"
sleep 10

# Test 1: Very conservative settings (safest)
test_configuration "very_conservative" 768 "" "false" "false" 1 2048 "cuda_v11"

# Uncomment these tests one by one after confirming the very conservative test works
# without disconnecting your GPU

# # Test 2: Conservative settings
# test_configuration "conservative" 512 "" "false" "false" 1 2048 "cuda_v11"
#
# # Test 3: Balanced settings
# test_configuration "balanced" 384 "" "false" "false" 1 4096 "cuda_v11"
#
# # Test 4: Memory-optimized settings
# MAX_VRAM_70=$(( TOTAL_VRAM * 70 / 100 ))
# test_configuration "memory_optimized" 512 "$MAX_VRAM_70" "false" "false" 1 4096 "cuda_v11"
#
# # Test 5: GTX 1080 specific settings
# test_configuration "gtx1080_optimized" 512 "6144" "false" "false" 1 4096 "cuda_v11"
#
# # Only run these if previous tests were successful
# echo -e "\n${YELLOW}Checking if GPU is still stable before continuing with more aggressive tests...${NC}"
# check_gpu_connected

# # Test 6: Slightly more aggressive settings
# test_configuration "moderate" 256 "" "false" "false" 1 4096 "cuda_v11"
#
# # Test 7: Performance settings (most aggressive, run last)
# test_configuration "performance" 256 "" "true" "false" 1 4096 "cuda_v11"

# Final message
echo -e "\n${GREEN}=========================================================${NC}"
echo -e "${GREEN}   GPU Configuration Testing Complete                     ${NC}"
echo -e "${GREEN}=========================================================${NC}"
echo -e "\n${YELLOW}Results have been logged to: $LOG_FILE${NC}"
echo -e "\n${YELLOW}Successful configurations have been saved to: $CONFIG_DIR${NC}"
echo -e "\n${YELLOW}To apply a successful configuration, run:${NC}"
echo -e "  sudo cp $CONFIG_DIR/<config_name>.conf /etc/systemd/system/ollama.service.d/override.conf"
echo -e "  sudo systemctl daemon-reload"
echo -e "  sudo systemctl restart ollama"
