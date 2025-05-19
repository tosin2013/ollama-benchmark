#!/usr/bin/env python3
"""
Benchmark Results Database

A module for storing and retrieving Ollama benchmark results.
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

class BenchmarkDB:
    """Class for managing benchmark results storage and retrieval."""
    
    def __init__(self, db_path: str = "./benchmark_results.db"):
        """
        Initialize the benchmark database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create benchmark runs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            hardware_info TEXT,
            gpu_model TEXT,
            vram_total INTEGER,
            cpu_model TEXT,
            driver_version TEXT
        )
        ''')
        
        # Create benchmark results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            generated_tokens INTEGER NOT NULL,
            model_load_time REAL NOT NULL,
            prompt_processing_time REAL NOT NULL,
            generation_time REAL NOT NULL,
            total_time REAL NOT NULL,
            prompt_tokens_per_sec REAL NOT NULL,
            generation_tokens_per_sec REAL NOT NULL,
            combined_tokens_per_sec REAL NOT NULL,
            raw_metrics TEXT,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)
    
    def create_benchmark_run(self, hardware_info: Dict[str, Any]) -> int:
        """
        Create a new benchmark run entry.
        
        Args:
            hardware_info: Dictionary containing hardware information
                           (gpu_model, vram_total, cpu_model, driver_version)
        
        Returns:
            ID of the created benchmark run
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert new benchmark run
        cursor.execute('''
        INSERT INTO benchmark_runs 
        (timestamp, hardware_info, gpu_model, vram_total, cpu_model, driver_version)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(hardware_info),
            hardware_info.get('gpu_model', ''),
            hardware_info.get('vram_total', 0),
            hardware_info.get('cpu_model', ''),
            hardware_info.get('driver_version', '')
        ))
        
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return run_id
    
    def store_benchmark_result(
        self, 
        run_id: int, 
        model_name: str,
        metrics: Dict[str, Any]
    ) -> int:
        """
        Store a benchmark result for a specific model.
        
        Args:
            run_id: ID of the benchmark run
            model_name: Name of the benchmarked model
            metrics: Dictionary containing benchmark metrics
        
        Returns:
            ID of the stored benchmark result
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Calculate tokens per second
        prompt_tokens = metrics.get('prompt_eval_count', 0)
        generated_tokens = metrics.get('eval_count', 0)
        
        prompt_time = metrics.get('prompt_eval_duration', 0) / 1_000_000_000  # ns to seconds
        generation_time = metrics.get('eval_duration', 0) / 1_000_000_000  # ns to seconds
        load_time = metrics.get('load_duration', 0) / 1_000_000_000  # ns to seconds
        total_time = metrics.get('total_duration', 0) / 1_000_000_000  # ns to seconds
        
        # Avoid division by zero
        prompt_tokens_per_sec = prompt_tokens / prompt_time if prompt_time > 0 else 0
        generation_tokens_per_sec = generated_tokens / generation_time if generation_time > 0 else 0
        
        # Combined tokens per second
        total_tokens = prompt_tokens + generated_tokens
        processing_time = prompt_time + generation_time
        combined_tokens_per_sec = total_tokens / processing_time if processing_time > 0 else 0
        
        # Insert benchmark result
        cursor.execute('''
        INSERT INTO benchmark_results 
        (run_id, model_name, prompt_tokens, generated_tokens, 
         model_load_time, prompt_processing_time, generation_time, total_time,
         prompt_tokens_per_sec, generation_tokens_per_sec, combined_tokens_per_sec,
         raw_metrics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_id,
            model_name,
            prompt_tokens,
            generated_tokens,
            load_time,
            prompt_time,
            generation_time,
            total_time,
            prompt_tokens_per_sec,
            generation_tokens_per_sec,
            combined_tokens_per_sec,
            json.dumps(metrics)
        ))
        
        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return result_id
    
    def get_benchmark_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent benchmark runs.
        
        Args:
            limit: Maximum number of runs to return
        
        Returns:
            List of benchmark run dictionaries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM benchmark_runs
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        runs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return runs
    
    def get_benchmark_results(self, run_id: int) -> List[Dict[str, Any]]:
        """
        Get all benchmark results for a specific run.
        
        Args:
            run_id: ID of the benchmark run
        
        Returns:
            List of benchmark result dictionaries
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM benchmark_results
        WHERE run_id = ?
        ORDER BY generation_tokens_per_sec DESC
        ''', (run_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def generate_comparison_report(
        self, 
        run_ids: List[int] = None, 
        format: str = "markdown"
    ) -> str:
        """
        Generate a comparison report of benchmark results.
        
        Args:
            run_ids: List of run IDs to compare (defaults to last 2 runs)
            format: Output format ('markdown' or 'text')
        
        Returns:
            Formatted comparison report
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # If no run_ids are provided, use the last 2 runs
        if not run_ids:
            cursor.execute('''
            SELECT id FROM benchmark_runs
            ORDER BY timestamp DESC
            LIMIT 2
            ''')
            run_ids = [row['id'] for row in cursor.fetchall()]
        
        # Get run information
        runs_info = {}
        for run_id in run_ids:
            cursor.execute('SELECT * FROM benchmark_runs WHERE id = ?', (run_id,))
            runs_info[run_id] = dict(cursor.fetchone())
        
        # Get results for each run
        all_results = {}
        all_models = set()
        
        for run_id in run_ids:
            cursor.execute('''
            SELECT * FROM benchmark_results 
            WHERE run_id = ?
            ''', (run_id,))
            
            run_results = {}
            for row in cursor.fetchall():
                result = dict(row)
                model_name = result['model_name']
                run_results[model_name] = result
                all_models.add(model_name)
            
            all_results[run_id] = run_results
        
        conn.close()
        
        # Generate the report
        if format == 'markdown':
            return self._generate_markdown_report(runs_info, all_results, all_models)
        else:
            return self._generate_text_report(runs_info, all_results, all_models)
    
    def _generate_markdown_report(
        self, 
        runs_info: Dict[int, Dict], 
        all_results: Dict[int, Dict[str, Dict]], 
        all_models: set
    ) -> str:
        """Generate a Markdown formatted comparison report."""
        report = []
        report.append("# Ollama Benchmark Comparison Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add run information
        report.append("## Hardware Information")
        report.append("")
        report.append("| Run ID | Timestamp | GPU | VRAM | CPU |")
        report.append("|--------|-----------|-----|------|-----|")
        
        for run_id, info in runs_info.items():
            timestamp = info['timestamp'].split('T')[0]
            gpu = info['gpu_model'] or 'N/A'
            vram = f"{info['vram_total']}MB" if info['vram_total'] else 'N/A'
            cpu = info['cpu_model'] or 'N/A'
            report.append(f"| {run_id} | {timestamp} | {gpu} | {vram} | {cpu} |")
        
        report.append("")
        
        # Add comparison table
        report.append("## Performance Comparison")
        report.append("")
        
        # Table header
        header = ["Model"]
        for run_id in runs_info.keys():
            header.append(f"Run {run_id} Generation (t/s)")
        report.append("| " + " | ".join(header) + " |")
        
        # Table separator
        separator = ["-" * len(h) for h in header]
        report.append("| " + " | ".join(separator) + " |")
        
        # Table rows
        for model in sorted(all_models):
            row = [model]
            
            for run_id in runs_info.keys():
                if model in all_results[run_id]:
                    gen_speed = all_results[run_id][model]['generation_tokens_per_sec']
                    row.append(f"{gen_speed:.2f}")
                else:
                    row.append("N/A")
            
            report.append("| " + " | ".join(row) + " |")
        
        report.append("")
        report.append("## Detailed Results")
        
        for model in sorted(all_models):
            report.append("")
            report.append(f"### {model}")
            report.append("")
            report.append("| Metric | " + " | ".join([f"Run {run_id}" for run_id in runs_info.keys()]) + " |")
            report.append("| ------ | " + " | ".join(["-----" for _ in runs_info.keys()]) + " |")
            
            metrics = [
                ("Prompt Tokens", "prompt_tokens"),
                ("Generated Tokens", "generated_tokens"),
                ("Load Time (s)", "model_load_time"),
                ("Prompt Processing (t/s)", "prompt_tokens_per_sec"),
                ("Generation Speed (t/s)", "generation_tokens_per_sec"),
                ("Combined Speed (t/s)", "combined_tokens_per_sec")
            ]
            
            for metric_name, metric_key in metrics:
                row = [metric_name]
                
                for run_id in runs_info.keys():
                    if model in all_results[run_id]:
                        value = all_results[run_id][model][metric_key]
                        
                        # Format based on metric type
                        if metric_key.endswith('_per_sec'):
                            row.append(f"{value:.2f}")
                        elif metric_key.endswith('_time'):
                            row.append(f"{value:.2f}")
                        else:
                            row.append(str(value))
                    else:
                        row.append("N/A")
                
                report.append("| " + " | ".join(row) + " |")
        
        return "\n".join(report)
    
    def _generate_text_report(
        self, 
        runs_info: Dict[int, Dict], 
        all_results: Dict[int, Dict[str, Dict]], 
        all_models: set
    ) -> str:
        """Generate a plain text formatted comparison report."""
        report = []
        report.append("Ollama Benchmark Comparison Report")
        report.append("=" * 40)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add run information
        report.append("Hardware Information:")
        report.append("-" * 20)
        
        for run_id, info in runs_info.items():
            report.append(f"Run {run_id}:")
            report.append(f"  Timestamp: {info['timestamp']}")
            report.append(f"  GPU: {info['gpu_model'] or 'N/A'}")
            report.append(f"  VRAM: {info['vram_total'] or 'N/A'} MB")
            report.append(f"  CPU: {info['cpu_model'] or 'N/A'}")
            report.append("")
        
        # Add comparison table
        report.append("Performance Comparison:")
        report.append("-" * 20)
        
        # Calculate column widths
        model_width = max(len(model) for model in all_models)
        model_width = max(model_width, 10)  # Min width
        
        # Print header
        header = f"{'Model':{model_width}}"
        for run_id in runs_info.keys():
            header += f" | Run {run_id} Gen (t/s)"
        report.append(header)
        report.append("-" * len(header))
        
        # Print rows
        for model in sorted(all_models):
            row = f"{model:{model_width}}"
            
            for run_id in runs_info.keys():
                if model in all_results[run_id]:
                    gen_speed = all_results[run_id][model]['generation_tokens_per_sec']
                    row += f" | {gen_speed:14.2f}"
                else:
                    row += f" | {'N/A':14}"
            
            report.append(row)
        
        report.append("")
        report.append("Detailed Results:")
        report.append("-" * 20)
        
        for model in sorted(all_models):
            report.append(f"\n{model}:")
            
            metrics = [
                ("Prompt Tokens", "prompt_tokens"),
                ("Generated Tokens", "generated_tokens"),
                ("Load Time (s)", "model_load_time"),
                ("Prompt Processing (t/s)", "prompt_tokens_per_sec"),
                ("Generation Speed (t/s)", "generation_tokens_per_sec"),
                ("Combined Speed (t/s)", "combined_tokens_per_sec")
            ]
            
            for metric_name, metric_key in metrics:
                row = f"  {metric_name + ':':<25}"
                
                for run_id in runs_info.keys():
                    if model in all_results[run_id]:
                        value = all_results[run_id][model][metric_key]
                        
                        # Format based on metric type
                        if metric_key.endswith('_per_sec'):
                            row += f" {value:8.2f} |"
                        elif metric_key.endswith('_time'):
                            row += f" {value:8.2f} |"
                        else:
                            row += f" {value:8} |"
                    else:
                        row += f" {'N/A':8} |"
                
                report.append(row)
        
        return "\n".join(report)

def detect_hardware_info() -> Dict[str, Any]:
    """
    Detect system hardware information.
    
    Returns:
        Dictionary with hardware information
    """
    info = {
        'gpu_model': None,
        'vram_total': None,
        'cpu_model': None,
        'driver_version': None,
    }
    
    # Try to get GPU info
    try:
        import subprocess
        
        # Get GPU model
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info['gpu_model'] = result.stdout.strip()
        
        # Get VRAM total
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info['vram_total'] = int(result.stdout.strip())
        
        # Get driver version
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        info['driver_version'] = result.stdout.strip()
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        # GPU info not available
        pass
    
    # Try to get CPU info
    try:
        # Linux
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['cpu_model'] = line.split(':', 1)[1].strip()
                        break
    except Exception:
        # CPU info not available
        pass
    
    return info

def main() -> None:
    """Command-line interface for the benchmark database."""
    
    # Simple command-line parsing
    if len(sys.argv) < 2:
        print("Usage: benchmark_db.py <command> [options]")
        print("Commands:")
        print("  list-runs              List recent benchmark runs")
        print("  run-details <run_id>   Show details for a specific run")
        print("  compare <run1> <run2>  Compare two benchmark runs")
        print("  report <run_id>        Generate report for a specific run")
        return
    
    command = sys.argv[1]
    db = BenchmarkDB()
    
    if command == 'list-runs':
        runs = db.get_benchmark_runs()
        print(f"Recent benchmark runs ({len(runs)}):")
        print("-" * 40)
        for run in runs:
            print(f"Run {run['id']}:")
            print(f"  Timestamp: {run['timestamp']}")
            print(f"  GPU: {run['gpu_model'] or 'N/A'}")
            print(f"  VRAM: {run['vram_total'] or 'N/A'} MB")
            print("")
    
    elif command == 'run-details' and len(sys.argv) >= 3:
        run_id = int(sys.argv[2])
        results = db.get_benchmark_results(run_id)
        
        print(f"Benchmark results for run {run_id} ({len(results)} models):")
        print("-" * 40)
        
        # Sort by generation speed (descending)
        results.sort(key=lambda x: x['generation_tokens_per_sec'], reverse=True)
        
        for result in results:
            print(f"Model: {result['model_name']}")
            print(f"  Generation Speed: {result['generation_tokens_per_sec']:.2f} tokens/sec")
            print(f"  Combined Speed: {result['combined_tokens_per_sec']:.2f} tokens/sec")
            print(f"  Generated Tokens: {result['generated_tokens']}")
            print(f"  Load Time: {result['model_load_time']:.2f}s")
            print("")
    
    elif command == 'compare' and len(sys.argv) >= 4:
        run1 = int(sys.argv[2])
        run2 = int(sys.argv[3])
        report = db.generate_comparison_report([run1, run2])
        print(report)
    
    elif command == 'report' and len(sys.argv) >= 3:
        run_id = int(sys.argv[2])
        report = db.generate_comparison_report([run_id])
        print(report)
    
    else:
        print("Invalid command or missing arguments.")
        return

if __name__ == "__main__":
    main() 