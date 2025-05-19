#!/bin/bash
#
# Ollama GPU/CPU Configuration Script for Ubuntu
# This script configures Ollama to use NVIDIA GPU or CPU mode
# Optimized for NVIDIA GeForce GTX 1080
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="gpu"
GPU_OVERHEAD=256  # in MB
MAX_VRAM=""       # in MB, empty means use all available
FLASH_ATTENTION="false"
SCHED_SPREAD="false"
NUM_PARALLEL=0    # 0 means auto
CONTEXT_LENGTH=4096
LLM_LIBRARY=""
CONFIG_DIR="/etc/ollama/config"
PROFILE="balanced"  # balanced, aggressive, conservative

# Print header
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}   Ollama Configuration Script for Ubuntu                ${NC}"
echo -e "${GREEN}   Optimized for NVIDIA GeForce GTX 1080                ${NC}"
echo -e "${GREEN}=========================================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root or with sudo${NC}"
  exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --mode)
      MODE="$2"
      shift
      shift
      ;;
    --gpu-overhead)
      GPU_OVERHEAD="$2"
      shift
      shift
      ;;
    --max-vram)
      MAX_VRAM="$2"
      shift
      shift
      ;;
    --flash-attention)
      FLASH_ATTENTION="$2"
      shift
      shift
      ;;
    --sched-spread)
      SCHED_SPREAD="$2"
      shift
      shift
      ;;
    --num-parallel)
      NUM_PARALLEL="$2"
      shift
      shift
      ;;
    --context-length)
      CONTEXT_LENGTH="$2"
      shift
      shift
      ;;
    --llm-library)
      LLM_LIBRARY="$2"
      shift
      shift
      ;;
    --profile)
      PROFILE="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --mode <gpu|cpu>              Set Ollama to use GPU or CPU (default: gpu)"
      echo "  --gpu-overhead <MB>           Set GPU memory overhead in MB (default: 256)"
      echo "  --max-vram <MB>               Set maximum VRAM usage in MB (default: all available)"
      echo "  --flash-attention <true|false> Enable flash attention for faster token generation (default: false)"
      echo "  --sched-spread <true|false>   Distribute model evenly across GPUs (default: false)"
      echo "  --num-parallel <num>          Set concurrent request handling (default: 0 = auto)"
      echo "  --context-length <length>     Set context window size (default: 4096)"
      echo "  --llm-library <library>       Set LLM library (e.g., cuda_v11)"
      echo "  --profile <name>              Use a predefined profile (balanced, aggressive, conservative)"
      echo "  --help                        Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $key${NC}"
      exit 1
      ;;
  esac
done

# Validate mode
if [[ "$MODE" != "gpu" && "$MODE" != "cpu" ]]; then
  echo -e "${RED}Invalid mode: $MODE. Must be 'gpu' or 'cpu'.${NC}"
  exit 1
fi

# Apply profile settings if specified
if [[ "$PROFILE" == "aggressive" ]]; then
  # Aggressive profile - maximize performance, use more VRAM
  GPU_OVERHEAD=128
  FLASH_ATTENTION="true"
  NUM_PARALLEL=2
  CONTEXT_LENGTH=8192
  echo -e "${YELLOW}Using aggressive profile for maximum performance${NC}"
elif [[ "$PROFILE" == "conservative" ]]; then
  # Conservative profile - prioritize stability over performance
  GPU_OVERHEAD=512
  FLASH_ATTENTION="false"
  NUM_PARALLEL=1
  CONTEXT_LENGTH=2048
  echo -e "${YELLOW}Using conservative profile for maximum stability${NC}"
elif [[ "$PROFILE" == "balanced" ]]; then
  # Balanced profile - default settings
  echo -e "${YELLOW}Using balanced profile (default settings)${NC}"
elif [[ "$PROFILE" == "gtx1080" ]]; then
  # GTX 1080-specific optimizations
  GPU_OVERHEAD=200
  MAX_VRAM=7900  # Reserve ~100MB for system
  FLASH_ATTENTION="true"
  NUM_PARALLEL=1
  CONTEXT_LENGTH=4096
  echo -e "${YELLOW}Using GTX 1080 optimized profile${NC}"
else
  echo -e "${RED}Invalid profile: $PROFILE. Must be 'balanced', 'aggressive', 'conservative', or 'gtx1080'.${NC}"
  exit 1
fi

# Create config directory
mkdir -p "$CONFIG_DIR"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Check for NVIDIA GPU
echo -e "${BLUE}Step 1: Checking for NVIDIA GPU...${NC}"
if ! command_exists lspci; then
    echo -e "${YELLOW}Installing pciutils...${NC}"
    apt-get update && apt-get install -y pciutils
fi

NVIDIA_GPU=$(lspci | grep -i nvidia)
if [ -z "$NVIDIA_GPU" ]; then
    echo -e "${RED}No NVIDIA GPU detected. This script requires an NVIDIA GPU.${NC}"
    exit 1
else
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    echo "$NVIDIA_GPU"
fi

# Step 2: Install NVIDIA drivers
echo -e "\n${BLUE}Step 2: Installing NVIDIA drivers...${NC}"
if command_exists nvidia-smi; then
    echo -e "${GREEN}NVIDIA drivers already installed:${NC}"
    nvidia-smi
else
    echo -e "${YELLOW}Installing NVIDIA drivers...${NC}"
    apt-get update
    apt-get install -y nvidia-driver-535

    echo -e "${YELLOW}Checking driver installation...${NC}"
    if ! command_exists nvidia-smi; then
        echo -e "${RED}NVIDIA driver installation failed. Please install manually.${NC}"
        exit 1
    fi

    echo -e "${GREEN}NVIDIA drivers installed successfully:${NC}"
    nvidia-smi
fi

# Step 3: Install CUDA Toolkit
echo -e "\n${BLUE}Step 3: Installing CUDA Toolkit...${NC}"
if command_exists nvcc; then
    echo -e "${GREEN}CUDA Toolkit already installed:${NC}"
    nvcc --version
else
    echo -e "${YELLOW}Installing CUDA Toolkit...${NC}"
    apt-get update
    apt-get install -y nvidia-cuda-toolkit

    echo -e "${YELLOW}Checking CUDA installation...${NC}"
    if ! command_exists nvcc; then
        echo -e "${RED}CUDA Toolkit installation failed. Please install manually.${NC}"
        exit 1
    fi

    echo -e "${GREEN}CUDA Toolkit installed successfully:${NC}"
    nvcc --version
fi

# Step 4: Check if Ollama is installed
echo -e "\n${BLUE}Step 4: Checking Ollama installation...${NC}"
if ! command_exists ollama; then
    echo -e "${RED}Ollama is not installed. Please install Ollama first.${NC}"
    echo -e "${YELLOW}You can install Ollama with: curl -fsSL https://ollama.com/install.sh | sh${NC}"
    exit 1
else
    echo -e "${GREEN}Ollama is installed:${NC}"
    ollama --version
fi

# Configure Ollama based on selected mode
if [[ "$MODE" == "gpu" ]]; then
    echo -e "\n${BLUE}Configuring Ollama for GPU mode (NVIDIA GeForce GTX 1080)...${NC}"

    # Check for NVIDIA GPU
    echo -e "${BLUE}Checking for NVIDIA GeForce GTX 1080...${NC}"
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

        # Get GPU memory information
        TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
        echo -e "${BLUE}Total VRAM: ${TOTAL_VRAM}MB${NC}"

        # If MAX_VRAM is not set, calculate a reasonable value (80% of total)
        if [[ -z "$MAX_VRAM" ]]; then
            MAX_VRAM=$(( TOTAL_VRAM * 80 / 100 ))
            echo -e "${BLUE}Setting MAX_VRAM to 80% of total: ${MAX_VRAM}MB${NC}"
        fi
    fi

    # Create the systemd override directory
    mkdir -p /etc/systemd/system/ollama.service.d/

    # Display configuration settings
    echo -e "${BLUE}Creating GPU configuration with the following settings:${NC}"
    echo -e "${BLUE}  GPU Overhead:     ${GPU_OVERHEAD}MB${NC}"
    echo -e "${BLUE}  Max VRAM:         ${MAX_VRAM:-(all available)}MB${NC}"
    echo -e "${BLUE}  Flash Attention:  ${FLASH_ATTENTION}${NC}"
    echo -e "${BLUE}  Scheduler Spread: ${SCHED_SPREAD}${NC}"
    echo -e "${BLUE}  Parallel Requests:${NUM_PARALLEL}${NC}"
    echo -e "${BLUE}  Context Length:   ${CONTEXT_LENGTH}${NC}"
    echo -e "${BLUE}  LLM Library:      ${LLM_LIBRARY:-(auto)}${NC}"

    # Build the configuration
    CONFIG="[Service]\nEnvironment=\"CUDA_VISIBLE_DEVICES=0\"\nEnvironment=\"OLLAMA_USE_GPU=1\"\nEnvironment=\"OLLAMA_GPU_OVERHEAD=${GPU_OVERHEAD}\"\nEnvironment=\"OLLAMA_CONTEXT_LENGTH=${CONTEXT_LENGTH}\"\nEnvironment=\"OLLAMA_FLASH_ATTENTION=${FLASH_ATTENTION}\"\nEnvironment=\"OLLAMA_SCHED_SPREAD=${SCHED_SPREAD}\"\nEnvironment=\"OLLAMA_NUM_PARALLEL=${NUM_PARALLEL}\""

    # Add MAX_VRAM if specified
    if [[ -n "$MAX_VRAM" ]]; then
        CONFIG="${CONFIG}\nEnvironment=\"OLLAMA_MAX_VRAM=${MAX_VRAM}\""
    fi

    # Add LLM library if specified
    if [[ -n "$LLM_LIBRARY" ]]; then
        CONFIG="${CONFIG}\nEnvironment=\"OLLAMA_LLM_LIBRARY=${LLM_LIBRARY}\""
    fi

    # Write the configuration
    echo -e "$CONFIG" > /etc/systemd/system/ollama.service.d/override.conf

    # Save a copy to the config directory
    echo -e "$CONFIG" > "$CONFIG_DIR/gpu_config.conf"

    echo -e "${GREEN}Created Ollama GPU configuration:${NC}"
    cat /etc/systemd/system/ollama.service.d/override.conf

    # Restart Ollama service
    echo -e "\n${BLUE}Restarting Ollama service...${NC}"
    systemctl daemon-reload
    systemctl restart ollama

    # Wait for Ollama to start
    echo -e "${YELLOW}Waiting for Ollama service to start...${NC}"
    sleep 5

    # Check if Ollama is running
    if systemctl is-active --quiet ollama; then
        echo -e "${GREEN}Ollama service is running.${NC}"
    else
        echo -e "${RED}Ollama service failed to start. Please check the logs:${NC}"
        journalctl -u ollama -n 20
        exit 1
    fi

    # Verify GPU usage
    echo -e "\n${BLUE}Verifying GPU usage...${NC}"
    echo -e "${YELLOW}Checking GPU usage with nvidia-smi:${NC}"
    nvidia-smi

    echo -e "\n${YELLOW}Checking Ollama logs for GPU-related messages:${NC}"
    journalctl -u ollama -n 50 | grep -i "gpu\|cuda"

    # Run a test query
    echo -e "\n${BLUE}Running a test query...${NC}"
    echo -e "${YELLOW}This will load a model and run a simple query to test GPU acceleration.${NC}"
    echo -e "${YELLOW}If the test fails, consider running with --mode cpu instead.${NC}"

    # Check if a model is already pulled
    MODELS=$(ollama list)
    if [[ $MODELS == *"No models found"* ]]; then
        echo -e "${YELLOW}No models found. Pulling a small model for testing...${NC}"
        ollama pull tinyllama
        MODEL="tinyllama"
    else
        # Use the first available model
        MODEL=$(echo "$MODELS" | awk 'NR==2 {print $1}')
        echo -e "${GREEN}Using existing model: $MODEL${NC}"
    fi

    # Run a simple query
    echo -e "${YELLOW}Running test query with model $MODEL...${NC}"
    if ! time ollama run $MODEL "Hello, world! How are you today?" --verbose; then
        echo -e "${RED}GPU test failed. Consider switching to CPU mode with:${NC}"
        echo -e "${YELLOW}  sudo $0 --mode cpu${NC}"
    fi

    # Final message
    echo -e "\n${GREEN}=========================================================${NC}"
    echo -e "${GREEN}   Ollama GPU Configuration Complete                     ${NC}"
    echo -e "${GREEN}=========================================================${NC}"
    echo -e "\n${YELLOW}To verify GPU usage in real-time, run:${NC}"
    echo -e "  nvidia-smi -l 1"
    echo -e "\n${YELLOW}To monitor Ollama logs:${NC}"
    echo -e "  journalctl -u ollama -f"
    echo -e "\n${YELLOW}To benchmark performance:${NC}"
    echo -e "  time ollama run <model> \"Your test prompt here\" --verbose"
    echo -e "\n${YELLOW}If you encounter issues, try CPU mode:${NC}"
    echo -e "  sudo $0 --mode cpu"
    echo -e "\n${GREEN}Configuration saved to: $CONFIG_DIR/gpu_config.conf${NC}"

else
    # CPU mode configuration
    echo -e "\n${BLUE}Configuring Ollama for CPU-only mode...${NC}"

    # Create the systemd override directory
    mkdir -p /etc/systemd/system/ollama.service.d/

    # Create the override.conf file with CPU settings
    echo -e "${BLUE}Creating CPU-only configuration${NC}"

    # Write the configuration
    echo -e "[Service]\nEnvironment=\"CUDA_VISIBLE_DEVICES=-1\"\nEnvironment=\"OLLAMA_USE_GPU=0\"" > /etc/systemd/system/ollama.service.d/override.conf

    # Save a copy to the config directory
    echo -e "[Service]\nEnvironment=\"CUDA_VISIBLE_DEVICES=-1\"\nEnvironment=\"OLLAMA_USE_GPU=0\"" > "$CONFIG_DIR/cpu_config.conf"

    echo -e "${GREEN}Created Ollama CPU configuration:${NC}"
    cat /etc/systemd/system/ollama.service.d/override.conf

    # Restart Ollama service
    echo -e "\n${BLUE}Restarting Ollama service...${NC}"
    systemctl daemon-reload
    systemctl restart ollama

    # Wait for Ollama to start
    echo -e "${YELLOW}Waiting for Ollama service to start...${NC}"
    sleep 5

    # Check if Ollama is running
    if systemctl is-active --quiet ollama; then
        echo -e "${GREEN}Ollama service is running.${NC}"
    else
        echo -e "${RED}Ollama service failed to start. Please check the logs:${NC}"
        journalctl -u ollama -n 20
        exit 1
    fi

    # Run a test query
    echo -e "\n${BLUE}Running a test query...${NC}"
    echo -e "${YELLOW}This will load a model and run a simple query to test CPU mode.${NC}"
    echo -e "${YELLOW}Note: CPU mode will be significantly slower than GPU mode.${NC}"

    # Check if a model is already pulled
    MODELS=$(ollama list)
    if [[ $MODELS == *"No models found"* ]]; then
        echo -e "${YELLOW}No models found. Pulling a small model for testing...${NC}"
        ollama pull tinyllama
        MODEL="tinyllama"
    else
        # Use the first available model
        MODEL=$(echo "$MODELS" | awk 'NR==2 {print $1}')
        echo -e "${GREEN}Using existing model: $MODEL${NC}"
    fi

    # Run a simple query
    echo -e "${YELLOW}Running test query with model $MODEL...${NC}"
    time ollama run $MODEL "Hello, world! How are you today?" --verbose

    # Final message
    echo -e "\n${GREEN}=========================================================${NC}"
    echo -e "${GREEN}   Ollama CPU Configuration Complete                     ${NC}"
    echo -e "${GREEN}=========================================================${NC}"
    echo -e "\n${YELLOW}To monitor Ollama logs:${NC}"
    echo -e "  journalctl -u ollama -f"
    echo -e "\n${YELLOW}To benchmark performance:${NC}"
    echo -e "  time ollama run <model> \"Your test prompt here\" --verbose"
    echo -e "\n${YELLOW}To switch back to GPU mode:${NC}"
    echo -e "  sudo $0 --mode gpu"
    echo -e "\n${GREEN}Configuration saved to: $CONFIG_DIR/cpu_config.conf${NC}"
fi
