#!/usr/bin/env python3
"""
Extended Ollama Benchmark Tool with Result Storage

An extension of benchmark.py that adds database storage and reporting capabilities.
"""

import argparse
import os
import sys
from typing import Dict, List

# Import functionality from main benchmark script
from benchmark import (
    run_benchmark, 
    get_benchmark_models, 
    average_stats, 
    inference_stats,
    OllamaResponse,
    extract_and_save_python_code
)

# Import database storage functionality
from benchmark_db import BenchmarkDB, detect_hardware_info

def main() -> None:
    """
    Main execution function for the extended benchmark tool.
    Adds database storage to the original benchmark functionality.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark performance metrics for Ollama models with result storage."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output including streaming responses",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Specific models to benchmark. Tests all available models if not specified.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            # Short analytical question to test basic reasoning
            "Explain the process of photosynthesis in plants, including the key chemical reactions and energy transformations involved.",
            
            # Medium-length creative task
            "Write a detailed story about a time traveler who visits three different historical periods. Include specific details about each era and the protagonist's interactions.",
            
            # Long complex analysis
            "Analyze the potential impact of artificial intelligence on global employment over the next decade. Consider various industries, economic factors, and potential mitigation strategies. Provide specific examples and data-driven reasoning.",
            
            # Technical task with specific requirements
            "Write a Python function that implements a binary search tree with methods for insertion, deletion, and traversal. Include comments explaining the time complexity of each operation.",
            
            # Structured output task
            "Create a detailed business plan for a renewable energy startup. Include sections on market analysis, financial projections, competitive advantages, and risk assessment. Format the response with clear headings and bullet points.",
            
            # Coding challenge for solution.py
            "Create a Python function called 'solve_array_problem' that takes an array of integers and returns the maximum product of any three integers in the array. Include efficient error handling, edge cases (such as arrays with fewer than 3 elements), and add comprehensive docstrings. Write tests to verify your solution works correctly with various inputs.",
        ],
        help="Prompts to use for benchmarking. Multiple prompts can be specified. Default prompts test various capabilities including analysis, creativity, technical knowledge, and structured output.",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Disable storing results in the database",
        default=False,
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only display reports from existing benchmark runs without running new benchmarks",
        default=False,
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List recent benchmark runs from the database",
        default=False,
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=int,
        help="Compare specific benchmark run IDs",
    )
    parser.add_argument(
        "--test-coding",
        action="store_true",
        help="Run only the coding challenge and save solutions",
        default=False,
    )

    args = parser.parse_args()
    
    # Initialize database
    db = BenchmarkDB()
    
    # Report-only mode: list runs or compare existing runs
    if args.list_runs:
        runs = db.get_benchmark_runs()
        if not runs:
            print("No benchmark runs found in the database.")
            return
            
        print(f"Recent benchmark runs ({len(runs)}):")
        print("-" * 40)
        for run in runs:
            print(f"Run {run['id']}:")
            print(f"  Timestamp: {run['timestamp']}")
            print(f"  GPU: {run['gpu_model'] or 'N/A'}")
            print(f"  VRAM: {run['vram_total'] or 'N/A'} MB")
            print("")
        return
        
    if args.compare:
        print(f"Comparing benchmark runs: {args.compare}")
        report = db.generate_comparison_report(args.compare)
        print(report)
        
        # Save report to file
        report_file = f"benchmark_comparison_{'_'.join(map(str, args.compare))}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nSaved comparison report to {report_file}")
        return
        
    if args.report_only:
        # Get the most recent run
        runs = db.get_benchmark_runs(limit=1)
        if not runs:
            print("No benchmark runs found in the database.")
            return
            
        run_id = runs[0]['id']
        report = db.generate_comparison_report([run_id])
        print(report)
        return
    
    # Regular benchmark mode
    print(
        f"\nVerbose: {args.verbose}\nTest models: {args.models}\nPrompts: {args.prompts}"
    )

    model_names = get_benchmark_models(args.models)
    benchmarks: Dict[str, List[OllamaResponse]] = {}

    # Execute benchmarks for each model and prompt
    for model_name in model_names:
        responses: List[OllamaResponse] = []
        
        # Handle test-coding mode specifically
        if args.test_coding:
            print(f"\n\nRunning coding challenge for: {model_name}")
            coding_prompt = "Create a Python function called 'solve_array_problem' that takes an array of integers and returns the maximum product of any three integers in the array. Include efficient error handling, edge cases (such as arrays with fewer than 3 elements), and add comprehensive docstrings. Write tests to verify your solution works correctly with various inputs."
            
            if response := run_benchmark(model_name, coding_prompt, verbose=True):
                # Extract and save the Python code
                extract_and_save_python_code(response.message.content, model_name)
                responses.append(response)
        else:
            # Regular benchmark mode
            for prompt in args.prompts:
                if args.verbose:
                    print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
                
                if response := run_benchmark(model_name, prompt, verbose=args.verbose):
                    responses.append(response)
                    
                    # If this is the coding challenge prompt, save the solution
                    if "solve_array_problem" in prompt:
                        extract_and_save_python_code(response.message.content, model_name)
                    
                    if args.verbose:
                        print(f"Response: {response.message.content}")
                        inference_stats(response)
        
        benchmarks[model_name] = responses

    # Calculate and display average statistics
    for model_name, responses in benchmarks.items():
        average_stats(responses)
    
    # Store benchmark results in database if enabled
    if not args.no_store and benchmarks:
        print("\nStoring benchmark results in database...")
        
        # Detect hardware information
        hw_info = detect_hardware_info()
        print(f"Detected hardware: {hw_info.get('gpu_model', 'Unknown GPU')}, {hw_info.get('vram_total', 'Unknown')}MB VRAM")
        
        # Create a new benchmark run
        run_id = db.create_benchmark_run(hw_info)
        print(f"Created benchmark run with ID: {run_id}")
        
        # Store results for each model
        for model_name, responses in benchmarks.items():
            if not responses:
                continue
                
            # Calculate average metrics
            avg_metrics = {
                'model': model_name,
                'total_duration': sum(r.total_duration for r in responses) // len(responses),
                'load_duration': sum(r.load_duration for r in responses) // len(responses),
                'prompt_eval_count': sum(r.prompt_eval_count for r in responses),
                'prompt_eval_duration': sum(r.prompt_eval_duration for r in responses),
                'eval_count': sum(r.eval_count for r in responses),
                'eval_duration': sum(r.eval_duration for r in responses),
            }
            
            # Store in database
            result_id = db.store_benchmark_result(run_id, model_name, avg_metrics)
            print(f"Stored result for {model_name} with ID: {result_id}")
        
        print("\nGenerated benchmark report:")
        report = db.generate_comparison_report([run_id])
        print(report)
        
        # Save report to file
        report_file = f"benchmark_report_{run_id}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nSaved benchmark report to {report_file}")

if __name__ == "__main__":
    main() 