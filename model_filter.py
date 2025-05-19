#!/usr/bin/env python3
"""
Ollama Model Filter

A utility for filtering Ollama models based on VRAM requirements and other criteria.
This helps users identify which models will run efficiently on their hardware.

Usage:
    python model_filter.py [--vram AVAILABLE_VRAM_MB] [--category CATEGORY]
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Model database - maps model names to their properties
# Format: "model_name": {"parameters": float (in billions), "min_vram": int (in MB), "category": str}
MODEL_DB = {
    # 1-3B models
    "phi:latest": {"parameters": 2.7, "min_vram": 4000, "category": "general"},
    "tinyllama:latest": {"parameters": 1.1, "min_vram": 2000, "category": "general"},
    "granite-code:3b": {"parameters": 3.0, "min_vram": 4000, "category": "coding"},
    "starcoder2:3b": {"parameters": 3.0, "min_vram": 3500, "category": "coding"},
    "qwen2.5-coder:3b": {"parameters": 3.0, "min_vram": 4000, "category": "coding"},
    
    # 7B models
    "llama2:7b": {"parameters": 7.0, "min_vram": 7000, "category": "general"},
    "codellama:7b": {"parameters": 7.0, "min_vram": 7000, "category": "coding"},
    "mistral:7b": {"parameters": 7.0, "min_vram": 7000, "category": "general"},
    "mistral:7b-instruct": {"parameters": 7.0, "min_vram": 7000, "category": "general"},
    "neural-chat:7b": {"parameters": 7.0, "min_vram": 7000, "category": "general"},
    "deepseek-coder:6.7b": {"parameters": 6.7, "min_vram": 6700, "category": "coding"},
    "deepseek-coder:6.7b-instruct": {"parameters": 6.7, "min_vram": 6700, "category": "coding"},
    
    # Quantized models
    "llama2:7b-q4_0": {"parameters": 7.0, "min_vram": 4000, "category": "general"},
    "mistral:7b-q4_0": {"parameters": 7.0, "min_vram": 4000, "category": "general"},
    "codellama:7b-q4_0": {"parameters": 7.0, "min_vram": 4000, "category": "coding"},
}

def get_available_vram() -> int:
    """
    Detect NVIDIA GPU and return available VRAM in MB.
    Returns 0 if no GPU is detected.
    """
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Parse output to get VRAM in MB
        vram = int(result.stdout.strip())
        print(f"Detected {vram}MB VRAM on NVIDIA GPU")
        return vram
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        print("No NVIDIA GPU detected or nvidia-smi failed")
        return 0

def estimate_vram_requirements(model_name: str) -> int:
    """
    Estimate VRAM requirements for a given model based on parameter count.
    
    Args:
        model_name: Name of the Ollama model
        
    Returns:
        Estimated VRAM requirement in MB, or 0 if unknown
    """
    # Check if model is in our database
    if model_name in MODEL_DB:
        return MODEL_DB[model_name]["min_vram"]
    
    # If we don't have info about this specific model, make a best guess
    # based on the model name (many follow patterns like llama2:13b)
    try:
        # Look for patterns like "7b", "13b", etc. in the model name
        for part in model_name.split(":"):
            if part.endswith("b") and part[:-1].isdigit():
                # Extract the number before "b"
                param_billions = float(part[:-1])
                
                # Rough estimate: 1B params ≈ 1GB VRAM in FP16
                if "q4" in model_name:  # Check if it's a 4-bit quantized model
                    return int(param_billions * 500)  # 4-bit quantized needs ~0.5GB per billion params
                else:
                    return int(param_billions * 1000)  # ~1GB per billion params
    except (ValueError, IndexError):
        pass
        
    # If we couldn't determine, return a very conservative estimate
    print(f"Warning: Couldn't estimate VRAM for unknown model {model_name}")
    return 8000  # Assume it needs 8GB by default

def get_compatible_models(
    available_vram: int = 8192,
    category: Optional[str] = None,
    min_params: float = 0.0,
    max_params: float = float('inf')
) -> List[Tuple[str, dict]]:
    """
    Return a list of Ollama models that should run on the available VRAM.
    
    Args:
        available_vram: Available VRAM in MB (default: 8GB for GTX 1080)
        category: Filter by model category (e.g., "coding", "general")
        min_params: Minimum model size in billions of parameters
        max_params: Maximum model size in billions of parameters
        
    Returns:
        List of (model_name, model_info) tuples for compatible models
    """
    compatible_models = []
    
    for model_name, model_info in MODEL_DB.items():
        # Check VRAM requirement
        if model_info["min_vram"] <= available_vram:
            # Check category if specified
            if category and model_info["category"] != category:
                continue
                
            # Check parameter count
            if model_info["parameters"] < min_params or model_info["parameters"] > max_params:
                continue
                
            compatible_models.append((model_name, model_info))
    
    # Sort by parameter count (descending)
    compatible_models.sort(key=lambda x: x[1]["parameters"], reverse=True)
    
    return compatible_models

def print_model_table(models: List[Tuple[str, dict]], include_header: bool = True) -> None:
    """
    Print a formatted table of models with their properties.
    
    Args:
        models: List of (model_name, model_info) tuples
        include_header: Whether to include the table header
    """
    if not models:
        print("No compatible models found.")
        return
        
    if include_header:
        print(f"{'Model Name':<25} {'Parameters':<12} {'Min VRAM':<12} {'Category':<10}")
        print(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 10}")
        
    for model_name, model_info in models:
        print(f"{model_name:<25} {model_info['parameters']:<12.1f}B {model_info['min_vram']:<12} {model_info['category']:<10}")

def main() -> None:
    """Main function to run the model filter from command line."""
    parser = argparse.ArgumentParser(description="Filter Ollama models based on VRAM and other criteria")
    parser.add_argument("--vram", type=int, help="Available VRAM in MB (default: auto-detect)")
    parser.add_argument("--category", choices=["general", "coding"], help="Filter by model category")
    parser.add_argument("--min-params", type=float, default=0.0, help="Minimum model size in billions of parameters")
    parser.add_argument("--max-params", type=float, default=float('inf'), help="Maximum model size in billions of parameters")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    # Determine available VRAM
    if args.vram:
        available_vram = args.vram
    else:
        available_vram = get_available_vram()
        if available_vram == 0:
            available_vram = 8192  # Default to 8GB if no GPU detected
    
    # Get compatible models
    compatible_models = get_compatible_models(
        available_vram=available_vram,
        category=args.category,
        min_params=args.min_params,
        max_params=args.max_params
    )
    
    # Output results
    if args.json:
        output = {model_name: model_info for model_name, model_info in compatible_models}
        print(json.dumps(output, indent=2))
    else:
        print(f"\nModels compatible with {available_vram}MB VRAM:")
        print_model_table(compatible_models)
        
        # Print comma-separated list for easy use with the workflow script
        model_list = ",".join(model_name for model_name, _ in compatible_models)
        print("\nComma-separated list for workflow script:")
        print(model_list)

if __name__ == "__main__":
    main() 