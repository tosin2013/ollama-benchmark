# Ollama Benchmark Suite Installation Guide

This guide walks you through installing and setting up the Ollama Benchmark Suite on your system.

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: Python 3.8+ 
- **GPU**: NVIDIA GPU with at least 8GB VRAM (GTX 1080 or better recommended)
- **CUDA**: CUDA 11.4+ and compatible drivers
- **Storage**: At least 20GB free disk space (for models)
- **Memory**: Minimum 16GB RAM recommended

## Step 1: Install NVIDIA Drivers and CUDA

### Check for NVIDIA GPU
```bash
lspci | grep -i nvidia
```

### Install NVIDIA Drivers (Ubuntu)
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### Verify Installation
```bash
nvidia-smi
```

### Install CUDA Toolkit
```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

### Verify CUDA Installation
```bash
nvcc --version
```

## Step 2: Install Ollama

You can install Ollama using our script or manually:

### Using the Script
```bash
sudo ./run_benchmark_workflow.sh --install
```

### Manual Installation
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify Ollama Installation
```bash
ollama --version
```

## Step 3: Clone the Benchmark Repository

```bash
git clone https://github.com/yourusername/ollama-benchmark.git
cd ollama-benchmark
```

## Step 4: Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
# or
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

If you encounter issues with the sqlite3 package, try:
```bash
pip install pydantic ollama
```

## Step 5: Optimize Ollama for Your GPU

```bash
sudo ./run_benchmark_workflow.sh --optimize --profile gtx1080
```

Available profiles:
- `gtx1080`: Optimized for NVIDIA GTX 1080 (8GB VRAM)
- `balanced`: Balanced performance and stability
- `aggressive`: Maximum performance, higher VRAM usage
- `conservative`: Prioritizes stability over performance

## Step 6: Pull Compatible Models

```bash
# Pull models compatible with your GPU
./run_benchmark_workflow.sh --pull-compatible

# Or pull specific models
ollama pull granite-code:3b
ollama pull starcoder2:3b
ollama pull qwen2.5-coder:3b
```

## Step 7: Run Your First Benchmark

```bash
# Run basic benchmark with default models
python benchmark.py

# Run with specific models
python benchmark.py --models granite-code:3b starcoder2:3b

# Run coding challenge test
python benchmark.py --test-coding

# Run with result storage
python benchmark_with_storage.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'ollama'"
```bash
pip install ollama
```

### "CUDA error: no kernel image is available for execution"
This usually means your CUDA version is incompatible with Ollama.
```bash
# Check CUDA version
nvcc --version
# Install compatible version or use CPU mode
sudo ./run_benchmark_workflow.sh --optimize --mode cpu
```

### "No such file or directory: 'nvidia-smi'"
NVIDIA drivers aren't installed or aren't working properly.
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### Permission errors
```bash
# Make scripts executable
chmod +x *.sh
# Run with sudo when needed
sudo ./run_benchmark_workflow.sh --install
```

## Next Steps

- Explore model compatibility with your GPU: `python model_filter.py`
- Get model recommendations: `python model_recommendations.py --use-case coding`
- Compare different benchmark runs: `python benchmark_with_storage.py --list-runs`
- Try the coding challenge feature: `python benchmark.py --test-coding`

See the full [User Guide](./user_guide.md) for more advanced usage scenarios. 