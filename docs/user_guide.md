# Ollama Benchmark Suite User Guide

This guide covers all features of the Ollama Benchmark Suite and provides detailed usage instructions for getting the most out of your benchmarking workflow.

## Table of Contents

- [Overview](#overview)
- [Workflow Scripts](#workflow-scripts)
- [GPU Optimization](#gpu-optimization)
- [Model Filtering](#model-filtering)
- [Benchmarking](#benchmarking)
- [Coding Challenge Testing](#coding-challenge-testing)
- [Results Database](#results-database)
- [Model Recommendations](#model-recommendations)
- [Advanced Usage](#advanced-usage)

## Overview

The Ollama Benchmark Suite is a comprehensive toolkit for:

1. Installing and configuring Ollama
2. Optimizing GPU performance for LLMs
3. Filtering models based on hardware compatibility
4. Benchmarking model performance
5. Comparing results across models
6. Testing models on specific tasks like coding challenges
7. Getting recommendations based on use case

## Workflow Scripts

### Unified Workflow

The `run_benchmark_workflow.sh` script provides a unified interface for the entire benchmarking process:

```bash
# Full workflow: install, optimize, pull models, benchmark
sudo ./run_benchmark_workflow.sh --install --optimize --pull-compatible --benchmark-all

# See all options
./run_benchmark_workflow.sh --help
```

### Available Options

| Option | Description |
|--------|-------------|
| `--install` | Install Ollama |
| `--optimize` | Optimize Ollama for GPU |
| `--profile <name>` | Select optimization profile (gtx1080, balanced, aggressive, conservative) |
| `--pull <models>` | Pull specific models (comma-separated) |
| `--pull-compatible` | Pull all models compatible with your GPU |
| `--benchmark <models>` | Benchmark specific models |
| `--benchmark-all` | Benchmark all available models |
| `--verbose` | Show detailed output |

## GPU Optimization

### Using Optimization Profiles

The `optimize_ollama_gpu.sh` script configures Ollama for different hardware:

```bash
# Optimize for GTX 1080
sudo ./optimize_ollama_gpu.sh --mode gpu --profile gtx1080

# Use aggressive settings for maximum performance
sudo ./optimize_ollama_gpu.sh --mode gpu --profile aggressive

# Use CPU mode (no GPU)
sudo ./optimize_ollama_gpu.sh --mode cpu
```

### Custom Optimization

Fine-tune GPU parameters manually:

```bash
sudo ./optimize_ollama_gpu.sh --mode gpu --gpu-overhead 200 --max-vram 7800 --flash-attention true --num-parallel 1
```

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode <gpu|cpu>` | Set to GPU or CPU mode | gpu |
| `--gpu-overhead <MB>` | GPU memory overhead | 256 |
| `--max-vram <MB>` | Maximum VRAM to use | (all available) |
| `--flash-attention <true|false>` | Enable flash attention | false |
| `--sched-spread <true|false>` | Distribute model across GPUs | false |
| `--num-parallel <num>` | Concurrent request handling | 0 (auto) |
| `--context-length <length>` | Context window size | 4096 |

## Model Filtering

The `model_filter.py` script helps you identify models that are compatible with your hardware:

```bash
# Show all compatible models for your GPU
python model_filter.py

# Filter by category (e.g., coding)
python model_filter.py --category coding

# Specify custom VRAM
python model_filter.py --vram 6144

# Filter by model size
python model_filter.py --min-params 1 --max-params 7
```

### Output Formats

```bash
# Get JSON output
python model_filter.py --json

# Get just the comma-separated list for scripts
python model_filter.py | tail -n 1
```

## Benchmarking

### Basic Benchmarking

```bash
# Run with default settings
python benchmark.py

# Benchmark specific models
python benchmark.py --models llama2:7b codellama:7b phi:latest

# Show verbose output
python benchmark.py --verbose
```

### Custom Prompts

```bash
# Use custom prompts
python benchmark.py --prompts "Write a function to calculate Fibonacci numbers." "Explain quantum computing in simple terms."
```

### Enhanced Benchmarking with Storage

```bash
# Run benchmark with result storage
python benchmark_with_storage.py

# List previous benchmark runs
python benchmark_with_storage.py --list-runs

# Compare two benchmark runs
python benchmark_with_storage.py --compare 1 2
```

## Coding Challenge Testing

Test models on their ability to solve programming problems:

```bash
# Run the coding challenge for all available models
python benchmark.py --test-coding

# Test specific models
python benchmark.py --test-coding --models granite-code:3b starcoder2:3b

# Store results in database
python benchmark_with_storage.py --test-coding
```

### Understanding the Results

The coding challenge asks models to implement a function that finds the maximum product of any three integers in an array. Each model's solution is saved to a file named `{model_name}_solution.py`.

To analyze and test the solutions:

```bash
# Review a specific solution
cat granite-code_3b_solution.py

# Run the solution
python granite-code_3b_solution.py
```

## Results Database

The `benchmark_db.py` module provides a SQLite database to store and analyze benchmark results:

```bash
# List recent benchmark runs
python benchmark_db.py list-runs

# Show details of a specific run
python benchmark_db.py run-details 1

# Compare benchmark runs
python benchmark_db.py compare 1 2

# Generate a detailed report
python benchmark_db.py report 1
```

## Model Recommendations

The `model_recommendations.py` script helps you choose the best model for your specific use case:

```bash
# Get recommendations for coding
python model_recommendations.py --use-case coding

# Get recommendations for writing tasks
python model_recommendations.py --use-case writing

# Specify minimum performance
python model_recommendations.py --min-speed 15

# Save as Markdown report
python model_recommendations.py --use-case coding --format markdown
```

### Available Use Cases

| Use Case | Description | Priority |
|----------|-------------|----------|
| `coding` | Code completion and programming | Speed + Quality |
| `chat` | Conversational AI | Speed |
| `writing` | Content creation | Quality |
| `research` | Complex analysis | Quality |

## Advanced Usage

### Custom Optimization for Specific Models

When working with models that have specific requirements:

```bash
# Set up environment for a specific model
OLLAMA_MAX_VRAM=7800 OLLAMA_GPU_OVERHEAD=200 ollama run codellama:7b
```

### Running Multiple Models Simultaneously

To run multiple models on the same GPU:

```bash
# Start Ollama server
ollama serve

# In separate terminal, run benchmarks with reduced VRAM
OLLAMA_MAX_VRAM=3800 python benchmark.py --models phi:latest
```

### Creating Custom Benchmark Prompts

To create domain-specific benchmarks:

1. Create a file with your prompts:
```
# coding_prompts.txt
Write a Python function to implement a binary search tree.
Create a JavaScript function to find the longest palindrome in a string.
...
```

2. Use them in your benchmark:
```bash
python benchmark.py --prompts "$(cat coding_prompts.txt)"
```

### Integrating with Continuous Testing

For automated testing in CI/CD pipelines:

```bash
# Example CI script
#!/bin/bash
./run_benchmark_workflow.sh --optimize --profile conservative
python benchmark.py --models codellama:7b-q4_0 --test-coding
python benchmark_db.py compare $(python benchmark_db.py list-runs | grep -o "Run [0-9]*:" | head -n 2 | cut -d':' -f1 | cut -d' ' -f2 | tr '\n' ' ')
``` 