# Ollama based LLM Benchmark

This tool allows you to get the t/s (tokens per second) of Large Language Models (LLMs) running on your local machine. Currently we only support testing Ollama llms

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Installation Guide](docs/installation_guide.md) - Step-by-step instructions for setting up the benchmark suite
- [User Guide](docs/user_guide.md) - Detailed instructions for all features 
- [GUI Implementation Guide](docs/gui_implementation_guide.md) - How to implement or modify a graphical interface
- [Documentation Overview](docs/README.md) - Summary of all documentation

## Example output

Output on a Nvidia 4090 windows desktop

```bash
Average stats:
(Running on dual 3090 Ti GPU, Epyc 7763 CPU in Ubuntu 22.04)

----------------------------------------------------
        Model: deepseek-r1:70b
        Performance Metrics:
            Prompt Processing:  336.73 tokens/sec
            Generation Speed:   17.65 tokens/sec
            Combined Speed:     18.01 tokens/sec

        Workload Stats:
            Input Tokens:       165
            Generated Tokens:   7673
            Model Load Time:    6.11s
            Processing Time:    0.49s
            Generation Time:    434.70s
            Total Time:         441.31s
----------------------------------------------------

Average stats: 
(Running on single 3090 GPU, 13900KS CPU in WSL2(Ubuntu 22.04) in Windows 11)

----------------------------------------------------
        Model: deepseek-r1:32b
        Performance Metrics:
            Prompt Processing:  399.05 tokens/sec
            Generation Speed:   27.18 tokens/sec
            Combined Speed:     27.58 tokens/sec

        Workload Stats:
            Input Tokens:       168
            Generated Tokens:   10601
            Model Load Time:    15.44s
            Processing Time:    0.42s
            Generation Time:    390.00s
            Total Time:         405.87s
----------------------------------------------------
```

## Getting Started

Follow these instructions to set up and run benchmarks on your system. For more detailed instructions, see the [Installation Guide](docs/installation_guide.md).

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com/) installed and configured
- NVIDIA GPU with appropriate drivers (for GPU acceleration)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/larryhopecode/ollama-benchmark.git
   cd ollama-benchmark
   ```

2. **Set up Python environment**

   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start Ollama Server**

   Ensure the Ollama service is running:

   ```bash
   ollama serve
   ```

2. **Run Benchmarks**

   Basic usage:

   ```bash
   python benchmark.py
   ```

   With options:

   ```bash
   python benchmark.py --verbose --models deepseek-r1:70b --prompts "Write a hello world program" "Explain quantum computing"
   ```

   For coding challenge evaluation:
   
   ```bash
   python benchmark.py --test-coding --models granite-code:3b starcoder2:3b qwen2.5-coder:3b
   ```

### Command Line Options

- `-v, --verbose`: Enable detailed output including streaming responses
- `-m, --models`: Space-separated list of models to benchmark (defaults to all available models)
- `-p, --prompts`: Space-separated list of custom prompts (defaults to a predefined set testing various capabilities)
- `--test-coding`: Run only the coding challenge and save solutions to files

For a complete list of options and advanced usage examples, see the [User Guide](docs/user_guide.md).

## Enhanced Features

This benchmark suite now includes several enhanced features:

- GPU optimization for NVIDIA GPUs (especially GTX 1080)
- Model filtering based on VRAM requirements
- Benchmark result storage and comparison
- Coding challenge evaluation with automatic solution extraction
- Model recommendations based on performance and use case

See the full [User Guide](docs/user_guide.md) for details on using these features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
