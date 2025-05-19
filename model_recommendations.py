#!/usr/bin/env python3
"""
Ollama Model Recommendations

A utility for recommending Ollama models based on benchmark results and use case.
This helps users select the best models for their specific needs and hardware.

Usage:
    python model_recommendations.py [--use-case CATEGORY] [--min-speed MIN_TOKENS_PER_SEC]
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple, Any, Union

# Try to import the benchmark database module
try:
    from benchmark_db import BenchmarkDB, detect_hardware_info
except ImportError:
    print("Warning: benchmark_db module not available. Some features will be disabled.")
    BenchmarkDB = None
    detect_hardware_info = None

# Try to import the model filter module
try:
    from model_filter import get_compatible_models, MODEL_DB
except ImportError:
    print("Warning: model_filter module not available. Some features will be disabled.")
    get_compatible_models = None
    MODEL_DB = {}

# Define use case profiles and relevant models
USE_CASES = {
    "coding": {
        "description": "Code completion and programming assistance",
        "weight_factors": {
            "generation_speed": 0.7,  # Generation speed is important
            "coding_category": 0.3,   # Model should be trained for code
        },
        "min_tokens_per_sec": 10,     # Minimum acceptable speed
        "preferred_categories": ["coding"],
    },
    "chat": {
        "description": "Conversational AI and assistant-like interactions",
        "weight_factors": {
            "generation_speed": 0.6,  # Speed matters for responsive chat
            "combined_speed": 0.4,    # Overall throughput for multi-turn
        },
        "min_tokens_per_sec": 15,     # Need faster response for chat
        "preferred_categories": ["general"],
    },
    "writing": {
        "description": "Content creation and creative writing",
        "weight_factors": {
            "generation_speed": 0.4,  # Speed is less critical
            "quality": 0.6,           # Favor larger models for quality
        },
        "min_tokens_per_sec": 5,      # Can be slower for better quality
        "preferred_categories": ["general"],
    },
    "research": {
        "description": "Complex analysis and academic content",
        "weight_factors": {
            "generation_speed": 0.3,  # Speed less important
            "quality": 0.7,           # Quality is paramount
        },
        "min_tokens_per_sec": 3,      # Can be even slower for best results
        "preferred_categories": ["general"],
    },
}

def score_model(
    model_name: str, 
    benchmark_results: Dict[str, Any], 
    use_case: str = "coding", 
    vram_available: int = 8192
) -> float:
    """
    Score a model based on benchmark results and use case requirements.
    
    Args:
        model_name: Name of the model to score
        benchmark_results: Dictionary of benchmark results for the model
        use_case: The intended use case ("coding", "chat", "writing", "research")
        vram_available: Available VRAM in MB
        
    Returns:
        Score between 0 and 100, higher is better
    """
    if use_case not in USE_CASES:
        use_case = "coding"  # Default to coding
        
    profile = USE_CASES[use_case]
    weights = profile["weight_factors"]
    min_tokens_per_sec = profile.get("min_tokens_per_sec", 0)
    preferred_categories = profile.get("preferred_categories", [])
    
    # Check if the model meets minimum token generation speed
    if benchmark_results.get("generation_tokens_per_sec", 0) < min_tokens_per_sec:
        return 0  # Model is too slow for this use case
    
    # Initialize base score
    score = 50.0
    
    # Adjust score based on generation speed
    gen_speed = benchmark_results.get("generation_tokens_per_sec", 0)
    speed_score = min(100, (gen_speed / min_tokens_per_sec) * 50)  # Scale up to 100 max
    score += speed_score * weights.get("generation_speed", 0.5)
    
    # Adjust score based on combined speed if relevant
    if "combined_speed" in weights:
        combined_speed = benchmark_results.get("combined_tokens_per_sec", 0)
        combined_speed_score = min(100, (combined_speed / min_tokens_per_sec) * 50)
        score += combined_speed_score * weights.get("combined_speed", 0)
    
    # Adjust score based on model category if available in MODEL_DB
    if "coding_category" in weights and model_name in MODEL_DB:
        category = MODEL_DB[model_name].get("category", "")
        if category in preferred_categories:
            score += 20 * weights.get("coding_category", 0)
    
    # Adjust score based on quality preference (proxy: model size)
    if "quality" in weights and model_name in MODEL_DB:
        # Larger models generally produce better quality output
        params = MODEL_DB[model_name].get("parameters", 0)
        quality_score = min(100, params * 10)  # 10B params = 100 score
        score += quality_score * weights.get("quality", 0)
    
    # Check VRAM requirements
    if model_name in MODEL_DB:
        vram_required = MODEL_DB[model_name].get("min_vram", 0)
        if vram_required > vram_available:
            # Penalize models that might run out of VRAM
            vram_penalty = ((vram_required - vram_available) / vram_available) * 100
            score -= vram_penalty
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))

def recommend_models(
    benchmark_results: Dict[str, Dict[str, Any]],
    use_case: str = "coding",
    min_tokens_per_sec: float = 0,
    vram_available: int = 8192,
    top_n: int = 3
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Recommend models based on benchmark results and use case.
    
    Args:
        benchmark_results: Dictionary mapping model names to benchmark results
        use_case: The intended use case ("coding", "chat", "writing", "research")
        min_tokens_per_sec: Minimum acceptable token generation speed
        vram_available: Available VRAM in MB
        top_n: Number of top recommendations to return
        
    Returns:
        List of (model_name, score, benchmark_results) tuples for recommended models
    """
    if use_case not in USE_CASES and min_tokens_per_sec <= 0:
        # If no use case or min speed specified, use coding defaults
        use_case = "coding"
        min_tokens_per_sec = USE_CASES["coding"].get("min_tokens_per_sec", 10)
    elif use_case not in USE_CASES:
        # If invalid use case but min speed specified, use custom minimum
        use_case = None
    elif min_tokens_per_sec <= 0:
        # If valid use case but no min speed, use the profile default
        min_tokens_per_sec = USE_CASES[use_case].get("min_tokens_per_sec", 0)
    
    # Filter models that meet the minimum token generation speed
    qualified_models = []
    
    for model_name, results in benchmark_results.items():
        if results.get("generation_tokens_per_sec", 0) >= min_tokens_per_sec:
            if use_case:
                # Score model based on use case
                score = score_model(model_name, results, use_case, vram_available)
            else:
                # Simple scoring based just on generation speed
                gen_speed = results.get("generation_tokens_per_sec", 0)
                score = min(100, (gen_speed / min_tokens_per_sec) * 50)
            
            qualified_models.append((model_name, score, results))
    
    # Sort by score (descending)
    qualified_models.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return qualified_models[:top_n]

def format_recommendations(
    recommendations: List[Tuple[str, float, Dict[str, Any]]],
    use_case: Optional[str] = None,
    format: str = "text"
) -> str:
    """
    Format model recommendations as text or markdown.
    
    Args:
        recommendations: List of (model_name, score, benchmark_results) tuples
        use_case: The use case that was used for recommendations
        format: Output format ('text' or 'markdown')
        
    Returns:
        Formatted recommendations
    """
    if not recommendations:
        return "No models meet the specified requirements."
    
    if format == "markdown":
        output = []
        output.append("# Ollama Model Recommendations")
        
        if use_case:
            case_desc = USE_CASES.get(use_case, {}).get("description", use_case)
            output.append(f"\nUse Case: **{use_case}** ({case_desc})")
        
        output.append("\n## Recommended Models\n")
        output.append("| Rank | Model | Score | Generation Speed | Combined Speed | Load Time |")
        output.append("|------|-------|-------|-----------------|---------------|-----------|")
        
        for i, (model, score, results) in enumerate(recommendations, 1):
            gen_speed = results.get("generation_tokens_per_sec", 0)
            combined_speed = results.get("combined_tokens_per_sec", 0)
            load_time = results.get("model_load_time", 0)
            
            output.append(f"| {i} | {model} | {score:.1f}/100 | {gen_speed:.2f} t/s | {combined_speed:.2f} t/s | {load_time:.2f}s |")
        
        output.append("\n## Model Details\n")
        
        for model, score, results in recommendations:
            output.append(f"### {model} (Score: {score:.1f}/100)\n")
            
            # Add model info if available
            if model in MODEL_DB:
                info = MODEL_DB[model]
                output.append(f"- **Parameters**: {info.get('parameters', 'Unknown')}B")
                output.append(f"- **Category**: {info.get('category', 'Unknown')}")
                output.append(f"- **Min VRAM**: {info.get('min_vram', 'Unknown')}MB\n")
            
            # Add performance metrics
            output.append("#### Performance Metrics\n")
            output.append(f"- **Generation Speed**: {results.get('generation_tokens_per_sec', 0):.2f} tokens/sec")
            output.append(f"- **Prompt Processing**: {results.get('prompt_tokens_per_sec', 0):.2f} tokens/sec")
            output.append(f"- **Combined Speed**: {results.get('combined_tokens_per_sec', 0):.2f} tokens/sec")
            output.append(f"- **Load Time**: {results.get('model_load_time', 0):.2f} seconds")
            output.append("")
        
        return "\n".join(output)
    else:
        # Text format
        output = []
        output.append("OLLAMA MODEL RECOMMENDATIONS")
        output.append("=" * 30)
        
        if use_case:
            case_desc = USE_CASES.get(use_case, {}).get("description", use_case)
            output.append(f"\nUse Case: {use_case} ({case_desc})")
        
        output.append("\nRecommended Models:")
        output.append("-" * 20)
        
        for i, (model, score, results) in enumerate(recommendations, 1):
            gen_speed = results.get("generation_tokens_per_sec", 0)
            combined_speed = results.get("combined_tokens_per_sec", 0)
            
            output.append(f"{i}. {model} (Score: {score:.1f}/100)")
            output.append(f"   Generation Speed: {gen_speed:.2f} tokens/sec")
            output.append(f"   Combined Speed: {combined_speed:.2f} tokens/sec")
            
            if model in MODEL_DB:
                info = MODEL_DB[model]
                output.append(f"   Parameters: {info.get('parameters', 'Unknown')}B")
                output.append(f"   Category: {info.get('category', 'Unknown')}")
            
            output.append("")
        
        return "\n".join(output)

def main() -> None:
    """Main function to run the model recommendations from command line."""
    parser = argparse.ArgumentParser(description="Recommend Ollama models based on benchmark results and use case")
    parser.add_argument("--use-case", choices=list(USE_CASES.keys()), help="Intended use case for the model")
    parser.add_argument("--min-speed", type=float, default=0, help="Minimum acceptable token generation speed")
    parser.add_argument("--vram", type=int, help="Available VRAM in MB (default: auto-detect)")
    parser.add_argument("--top", type=int, default=3, help="Number of recommendations to show")
    parser.add_argument("--format", choices=["text", "markdown"], default="text", help="Output format")
    parser.add_argument("--run-id", type=int, help="Specific benchmark run ID to use (default: most recent)")
    args = parser.parse_args()
    
    # Check if benchmark_db is available
    if BenchmarkDB is None:
        print("Error: benchmark_db module is required for this functionality.")
        sys.exit(1)
    
    # Initialize database
    db = BenchmarkDB()
    
    # Get benchmark run
    if args.run_id:
        run_id = args.run_id
    else:
        # Get most recent run
        runs = db.get_benchmark_runs(limit=1)
        if not runs:
            print("Error: No benchmark runs found in the database.")
            sys.exit(1)
        run_id = runs[0]['id']
    
    # Get benchmark results
    results = db.get_benchmark_results(run_id)
    if not results:
        print(f"Error: No benchmark results found for run ID {run_id}.")
        sys.exit(1)
    
    # Convert to dictionary format
    benchmark_results = {}
    for result in results:
        model_name = result["model_name"]
        benchmark_results[model_name] = result
    
    # Determine available VRAM
    vram_available = args.vram
    if vram_available is None and detect_hardware_info:
        hw_info = detect_hardware_info()
        vram_available = hw_info.get('vram_total', 8192)  # Default to 8GB if detection fails
    elif vram_available is None:
        vram_available = 8192  # Default to 8GB
    
    # Generate recommendations
    recommendations = recommend_models(
        benchmark_results,
        use_case=args.use_case,
        min_tokens_per_sec=args.min_speed,
        vram_available=vram_available,
        top_n=args.top
    )
    
    # Format and print recommendations
    formatted_recommendations = format_recommendations(
        recommendations,
        use_case=args.use_case,
        format=args.format
    )
    print(formatted_recommendations)
    
    # Save recommendations to file
    filename = f"recommendations_{run_id}_{args.use_case or 'custom'}.{args.format == 'markdown' and 'md' or 'txt'}"
    with open(filename, 'w') as f:
        f.write(formatted_recommendations)
    print(f"\nSaved recommendations to {filename}")

if __name__ == "__main__":
    main() 