# Ollama Benchmark Suite Documentation

Welcome to the Ollama Benchmark Suite documentation. This comprehensive suite of tools helps you benchmark, optimize, and manage LLMs on NVIDIA GPUs.

## Documentation Contents

- [Installation Guide](installation_guide.md): Step-by-step instructions for setting up the benchmark suite on your system
- [User Guide](user_guide.md): Detailed usage instructions for all features
- [GUI Implementation Guide](gui_implementation_guide.md): Instructions for adding or modifying a graphical interface

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ollama-benchmark.git
cd ollama-benchmark

# Install dependencies
pip install -r requirements.txt

# Install and optimize Ollama
sudo ./run_benchmark_workflow.sh --install --optimize --profile gtx1080

# Pull compatible models
./run_benchmark_workflow.sh --pull-compatible

# Run benchmarks
python benchmark.py --test-coding
```

## System Overview

The Ollama Benchmark Suite consists of several key components:

1. **Workflow Scripts**: Unified interfaces for the benchmarking process
   - `run_benchmark_workflow.sh`: Main orchestration script
   - `optimize_ollama_gpu.sh`: GPU optimization

2. **Benchmarking Tools**: Core benchmarking functionality
   - `benchmark.py`: Core benchmarking script
   - `benchmark_with_storage.py`: Enhanced benchmarking with database storage

3. **Model Management**: Tools for working with models
   - `model_filter.py`: Filter models based on hardware compatibility
   - `model_recommendations.py`: Get personalized model recommendations

4. **Results Analysis**: Database and reporting tools
   - `benchmark_db.py`: SQLite database for result storage and analysis

## Use Cases

- **Hardware compatibility testing**: Find optimal models for your GPU
- **Performance benchmarking**: Compare models on standardized prompts
- **Coding ability evaluation**: Test models on programming challenges
- **Optimization discovery**: Find best parameter settings for your hardware

## Supported Hardware

This suite is optimized for:
- NVIDIA GTX 1080 and other 8GB VRAM GPUs
- Also works on other NVIDIA GPUs with appropriate optimizations
- CPU fallback mode for systems without GPUs

## Additional Resources

- [Ollama's Official Documentation](https://ollama.com/docs)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/) 