#!/usr/bin/env python3
"""
Ollama Model Benchmark Tool

A lightweight tool for measuring LLM performance metrics via Ollama:
- Token processing speed (t/s)
- Model load time
- Prompt evaluation time
- Response generation time

Usage:
    python benchmark.py [-v] [-m MODEL_NAMES...] [-p PROMPTS...]

Example:
    python benchmark.py --verbose --models llama2:13b codellama:34b
"""

import argparse
import os
from typing import List, Dict, Optional
from datetime import datetime

import ollama
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a single message in the chat interaction."""
    role: str
    content: str


class OllamaResponse(BaseModel):
    """
    Represents a structured response from the Ollama API.
    Contains performance metrics and message content.
    """
    model: str
    created_at: datetime | None = None
    message: Message
    done: bool
    total_duration: int = Field(default=0)
    load_duration: int = Field(default=0)
    prompt_eval_count: int = Field(default=0)
    prompt_eval_duration: int = Field(default=0)
    eval_count: int = Field(default=0)
    eval_duration: int = Field(default=0)

    @classmethod
    def from_chat_response(cls, response) -> 'OllamaResponse':
        """
        Converts an Ollama API response into an OllamaResponse instance.
        
        Args:
            response: Raw response from Ollama API
        
        Returns:
            OllamaResponse: Structured response object
        """
        return cls(
            model=response.model,
            message=Message(
                role=response.message.role,
                content=response.message.content
            ),
            done=response.done,
            total_duration=getattr(response, 'total_duration', 0),
            load_duration=getattr(response, 'load_duration', 0),
            prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
            prompt_eval_duration=getattr(response, 'prompt_eval_duration', 0),
            eval_count=getattr(response, 'eval_count', 0),
            eval_duration=getattr(response, 'eval_duration', 0)
        )


def run_benchmark(
    model_name: str, 
    prompt: str, 
    verbose: bool
) -> Optional[OllamaResponse]:
    """
    Executes a benchmark run for a specific model and prompt.

    Args:
        model_name: Name of the Ollama model to benchmark
        prompt: Input text to send to the model
        verbose: If True, prints streaming output

    Returns:
        OllamaResponse object containing benchmark results, or None if failed
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        if verbose:
            # For verbose mode, we'll collect the content while streaming
            content = ""
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if hasattr(chunk.message, 'content'):
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)
            
            if not content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response. Please check if:")
                print("1. The model is properly loaded")
                print("2. The Ollama server is functioning correctly")
                print("3. Try running 'ollama run {model_name}' in terminal to verify model output")
                return None
            
            # Make a non-streaming call to get the metrics
            response = ollama.chat(
                model=model_name,
                messages=messages,
            )
            
            # Check if response has content
            if not hasattr(response.message, 'content') or not response.message.content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response in non-streaming mode")
                return None
            
            # Create response with collected content and metrics
            return OllamaResponse(
                model=model_name,
                message=Message(
                    role="assistant",
                    content=content
                ),
                done=True,
                total_duration=getattr(response, 'total_duration', 0),
                load_duration=getattr(response, 'load_duration', 0),
                prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
                prompt_eval_duration=getattr(response, 'prompt_eval_duration', 0),
                eval_count=getattr(response, 'eval_count', 0),
                eval_duration=getattr(response, 'eval_duration', 0)
            )
        else:
            # For non-verbose mode, just make a single non-streaming call
            response = ollama.chat(
                model=model_name,
                messages=messages,
            )
            
            # Check if response has content
            if not hasattr(response.message, 'content') or not response.message.content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response. Please check if:")
                print("1. The model is properly loaded")
                print("2. The Ollama server is functioning correctly")
                print("3. Try running 'ollama run {model_name}' in terminal to verify model output")
                return None
                
            return OllamaResponse.from_chat_response(response)

    except Exception as e:
        print(f"Error benchmarking {model_name}: {str(e)}")
        return None


def nanosec_to_sec(nanosec: int) -> float:
    """Converts nanoseconds to seconds."""
    return nanosec / 1_000_000_000


def inference_stats(model_response: OllamaResponse) -> None:
    """
    Calculates and prints detailed inference statistics for a model response.

    Args:
        model_response: OllamaResponse containing benchmark metrics
    """
    # Calculate tokens per second for different phases
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (
        model_response.prompt_eval_count + model_response.eval_count
    ) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        Model: {model_response.model}
        Performance Metrics:
            Prompt Processing:  {prompt_ts:.2f} tokens/sec
            Generation Speed:   {response_ts:.2f} tokens/sec
            Combined Speed:     {total_ts:.2f} tokens/sec

        Workload Stats:
            Input Tokens:       {model_response.prompt_eval_count}
            Generated Tokens:   {model_response.eval_count}
            Model Load Time:    {nanosec_to_sec(model_response.load_duration):.2f}s
            Processing Time:    {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
            Generation Time:    {nanosec_to_sec(model_response.eval_duration):.2f}s
            Total Time:         {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]) -> None:
    """
    Calculates and prints average statistics across multiple benchmark runs.

    Args:
        responses: List of OllamaResponse objects from multiple runs
    """
    if not responses:
        print("No stats to average")
        return

    # Calculate aggregate metrics
    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def get_benchmark_models(test_models: List[str] = []) -> List[str]:
    """
    Retrieves and validates the list of models to benchmark.

    Args:
        test_models: List of specific models to test

    Returns:
        List of validated model names available for benchmarking
    """
    response = ollama.list()
    available_models = [model.get("model") for model in response.get("models", [])]
    
    if not test_models:
        # Use specified coding models by default
        default_models = ["granite-code:3b", "starcoder2:3b", "qwen2.5-coder:3b"]  # User's coding models
        model_names = [m for m in available_models if m in default_models]
        if not model_names:
            model_names = available_models[:3]  # Take first 3 available models if no defaults found
    else:
        # Filter requested models against available ones
        model_names = [model for model in test_models if model in available_models]
        if len(model_names) < len(test_models):
            missing_models = set(test_models) - set(available_models)
            print(f"Warning: Some requested models are not available: {missing_models}")
    
    if not model_names:
        raise RuntimeError("No valid models found for benchmarking")
        
    print(f"Evaluating models: {model_names}\n")
    return model_names


def extract_and_save_python_code(response_content: str, model_name: str, output_file: str = "solution.py") -> bool:
    """
    Extracts Python code from model response and saves it to a file.
    
    Args:
        response_content: Text content from the model
        model_name: Name of the model that generated the code
        output_file: Path to save the extracted code
        
    Returns:
        True if code was successfully extracted and saved, False otherwise
    """
    # Check if response is empty
    if not response_content.strip():
        print(f"Error: Empty response from {model_name}")
        return False
    
    # Try to extract code from markdown code blocks
    code_blocks = []
    in_code_block = False
    current_block = []
    
    for line in response_content.split('\n'):
        if line.strip().startswith('```python') or line.strip() == '```python':
            in_code_block = True
            current_block = []
        elif line.strip() == '```' and in_code_block:
            in_code_block = False
            if current_block:
                code_blocks.append('\n'.join(current_block))
        elif in_code_block:
            current_block.append(line)
    
    # If no code blocks found, try to extract directly
    if not code_blocks:
        # Check if the whole thing looks like Python code
        if 'def ' in response_content and ('return ' in response_content or 'print(' in response_content):
            code_blocks.append(response_content)
    
    # If still no code found, return failure
    if not code_blocks:
        print(f"Error: Could not extract Python code from {model_name}'s response")
        return False
    
    # Create the output file with model's solution
    try:
        # Create a unique filename for each model
        model_output_file = f"{model_name.replace(':', '_')}_{output_file}"
        with open(model_output_file, 'w') as f:
            f.write(f"# Solution generated by {model_name}\n\n")
            f.write(code_blocks[0])  # Use the first detected code block
        
        print(f"Saved {model_name}'s solution to {model_output_file}")
        return True
    except Exception as e:
        print(f"Error saving solution from {model_name}: {str(e)}")
        return False


def main() -> None:
    """
    Main execution function for the benchmark tool.
    Handles argument parsing and orchestrates the benchmark process.
    """
    # Try to import benchmark_db for results storage
    try:
        from benchmark_db import BenchmarkDB, detect_hardware_info
        has_db = True
    except ImportError:
        has_db = False

    parser = argparse.ArgumentParser(
        description="Benchmark performance metrics for Ollama models."
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
        "--test-coding",
        action="store_true",
        help="Run only the coding challenge and save solutions",
        default=False,
    )

    args = parser.parse_args()
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
    
    # Store benchmark results in database if available
    if has_db and benchmarks:
        print("\nStoring benchmark results in database...")
        db = BenchmarkDB()
        
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