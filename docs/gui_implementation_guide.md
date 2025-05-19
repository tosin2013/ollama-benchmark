# GUI Implementation Guide for Ollama Benchmark Suite

This guide explains how to implement a graphical user interface (GUI) for the Ollama Benchmark Suite or modify an existing one. The documentation covers both web-based and desktop implementations.

## Table of Contents

- [Overview](#overview)
- [Web-based GUI Implementation](#web-based-gui-implementation)
- [Desktop GUI Implementation](#desktop-gui-implementation)
- [Data Visualization](#data-visualization)
- [Real-time Monitoring](#real-time-monitoring)
- [Deployment Options](#deployment-options)

## Overview

The Ollama Benchmark Suite currently operates as a command-line tool. However, a graphical interface can significantly improve usability for many users. There are two main approaches to implementing a GUI:

1. **Web-based GUI**: Using frameworks like Flask, FastAPI, or Django
2. **Desktop GUI**: Using libraries like PyQt, Tkinter, or Electron

## Web-based GUI Implementation

### Flask Implementation

The simplest approach is to create a lightweight Flask application that wraps the benchmark functionality:

#### Step 1: Create Basic Structure

```
ollama-benchmark/
├── static/              # CSS, JS, and images
│   ├── css/
│   ├── js/
│   └── img/
├── templates/           # HTML templates
├── benchmark_web.py     # Flask application
├── benchmark.py         # Original benchmark script
└── ...
```

#### Step 2: Implement Flask App

Here's a minimal Flask app to get started (`benchmark_web.py`):

```python
from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import json
import os
from benchmark import get_benchmark_models
from benchmark_db import BenchmarkDB, detect_hardware_info

app = Flask(__name__)
benchmark_jobs = {}
next_job_id = 1

@app.route('/')
def index():
    # Get list of available models
    models = get_benchmark_models()
    # Get hardware info
    hw_info = detect_hardware_info()
    return render_template('index.html', models=models, hw_info=hw_info)

@app.route('/run_benchmark', methods=['POST'])
def run_benchmark():
    global next_job_id
    
    # Get parameters from form
    selected_models = request.form.getlist('models')
    is_test_coding = 'test_coding' in request.form
    
    # Create command
    cmd = ['python', 'benchmark.py']
    if selected_models:
        cmd.extend(['--models'] + selected_models)
    if is_test_coding:
        cmd.append('--test-coding')
    
    # Create job and run in background
    job_id = next_job_id
    next_job_id += 1
    
    job = {
        'id': job_id,
        'command': ' '.join(cmd),
        'status': 'running',
        'output': '',
        'models': selected_models
    }
    benchmark_jobs[job_id] = job
    
    # Start background task
    threading.Thread(target=run_job, args=(job_id, cmd)).start()
    
    return jsonify({'job_id': job_id})

def run_job(job_id, cmd):
    job = benchmark_jobs[job_id]
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Stream output
        for line in process.stdout:
            job['output'] += line
        
        # Wait for completion
        process.wait()
        job['status'] = 'completed' if process.returncode == 0 else 'failed'
    except Exception as e:
        job['status'] = 'failed'
        job['output'] += f"Error: {str(e)}"

@app.route('/job_status/<int:job_id>')
def job_status(job_id):
    if job_id not in benchmark_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = benchmark_jobs[job_id]
    return jsonify({
        'id': job['id'],
        'status': job['status'],
        'output': job['output']
    })

@app.route('/results')
def results():
    db = BenchmarkDB()
    runs = db.get_benchmark_runs()
    return render_template('results.html', runs=runs)

@app.route('/results/<int:run_id>')
def run_details(run_id):
    db = BenchmarkDB()
    results = db.get_benchmark_results(run_id)
    return render_template('run_details.html', results=results, run_id=run_id)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

#### Step 3: Create HTML Templates

Create a basic template (`templates/index.html`):

```html
<!DOCTYPE html>
<html>
<head>
    <title>Ollama Benchmark Suite</title>
    <link rel="stylesheet" href="/static/css/bootstrap.min.css">
    <script src="/static/js/jquery.min.js"></script>
    <script src="/static/js/bootstrap.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>Ollama Benchmark Suite</h1>
        
        <div class="card mb-4">
            <div class="card-header">
                <h2>Hardware Information</h2>
            </div>
            <div class="card-body">
                <p><strong>GPU:</strong> {{ hw_info.gpu_model or 'N/A' }}</p>
                <p><strong>VRAM:</strong> {{ hw_info.vram_total or 'N/A' }} MB</p>
                <p><strong>Driver:</strong> {{ hw_info.driver_version or 'N/A' }}</p>
            </div>
        </div>
        
        <div class="card mb-4">
            <div class="card-header">
                <h2>Run Benchmark</h2>
            </div>
            <div class="card-body">
                <form id="benchmark-form">
                    <div class="form-group">
                        <label>Select Models:</label>
                        {% for model in models %}
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" name="models" value="{{ model }}" id="model-{{ loop.index }}">
                            <label class="form-check-label" for="model-{{ loop.index }}">
                                {{ model }}
                            </label>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="form-check mb-3">
                        <input class="form-check-input" type="checkbox" name="test_coding" id="test-coding">
                        <label class="form-check-label" for="test-coding">
                            Run Coding Challenge Only
                        </label>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Run Benchmark</button>
                </form>
            </div>
        </div>
        
        <div id="job-status" class="card mb-4" style="display: none;">
            <div class="card-header">
                <h2>Benchmark Progress</h2>
            </div>
            <div class="card-body">
                <div class="progress mb-3">
                    <div id="progress-bar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                </div>
                <pre id="job-output" class="border p-3" style="max-height: 300px; overflow-y: auto;"></pre>
            </div>
        </div>
    </div>
    
    <script>
        let currentJobId = null;
        let statusInterval = null;
        
        $('#benchmark-form').submit(function(e) {
            e.preventDefault();
            
            // Show status card
            $('#job-status').show();
            $('#job-output').text('Starting benchmark...\n');
            
            // Submit form via AJAX
            $.ajax({
                url: '/run_benchmark',
                type: 'POST',
                data: $(this).serialize(),
                success: function(data) {
                    currentJobId = data.job_id;
                    $('#job-output').append(`Job ID: ${currentJobId}\n`);
                    
                    // Start polling for updates
                    statusInterval = setInterval(updateJobStatus, 1000);
                },
                error: function(xhr) {
                    $('#job-output').append('Error starting job: ' + xhr.responseText + '\n');
                }
            });
        });
        
        function updateJobStatus() {
            if (!currentJobId) return;
            
            $.ajax({
                url: `/job_status/${currentJobId}`,
                type: 'GET',
                success: function(data) {
                    $('#job-output').text(data.output);
                    
                    // Update progress based on status
                    if (data.status === 'running') {
                        $('#progress-bar').css('width', '50%');
                    } else if (data.status === 'completed') {
                        $('#progress-bar').css('width', '100%').removeClass('bg-info').addClass('bg-success');
                        clearInterval(statusInterval);
                    } else if (data.status === 'failed') {
                        $('#progress-bar').css('width', '100%').removeClass('bg-info').addClass('bg-danger');
                        clearInterval(statusInterval);
                    }
                    
                    // Auto-scroll to bottom
                    const output = document.getElementById('job-output');
                    output.scrollTop = output.scrollHeight;
                }
            });
        }
    </script>
</body>
</html>
```

#### Step 4: Install Dependencies

```bash
pip install flask
```

#### Step 5: Run the Web Server

```bash
python benchmark_web.py
```

Visit http://localhost:5000 in your browser.

### Enhancing the Web GUI

To enhance the web interface:

1. **Add Bootstrap or Material Design**: Use frontend frameworks for better styling
2. **Implement real-time updates**: Use WebSockets for live streaming of benchmark progress
3. **Add charts**: Integrate Chart.js or Plotly for data visualization
4. **Create a dashboard**: Show system stats, model comparisons, etc.

## Desktop GUI Implementation

### PyQt Implementation

For a desktop application, PyQt provides a robust framework:

#### Step 1: Basic Structure

```
ollama-benchmark/
├── gui/
│   ├── images/
│   ├── __init__.py
│   ├── main_window.py
│   ├── benchmark_tab.py
│   ├── results_tab.py
│   └── ...
├── benchmark_gui.py
└── ...
```

#### Step 2: Main Application

Create a file `benchmark_gui.py`:

```python
import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

#### Step 3: Main Window Implementation

Create `gui/main_window.py`:

```python
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel,
    QStatusBar, QProgressBar, QToolBar, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
import subprocess
import time

from benchmark import get_benchmark_models
from benchmark_db import BenchmarkDB, detect_hardware_info
from .benchmark_tab import BenchmarkTab
from .results_tab import ResultsTab

class BenchmarkRunner(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(int)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
    
    def run(self):
        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True
        )
        
        for line in process.stdout:
            self.progress.emit(line.strip())
        
        process.wait()
        self.completed.emit(process.returncode)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup UI
        self.setWindowTitle("Ollama Benchmark Suite")
        self.setMinimumSize(800, 600)
        
        # Create central widget with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Setup status bar with progress
        self.statusBar = QStatusBar()
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.statusBar.addPermanentWidget(self.progressBar)
        self.setStatusBar(self.statusBar)
        
        # Add tabs
        self.benchmark_tab = BenchmarkTab(self)
        self.results_tab = ResultsTab(self)
        
        self.tabs.addTab(self.benchmark_tab, "Benchmark")
        self.tabs.addTab(self.results_tab, "Results")
        
        # Set up toolbar
        self.init_toolbar()
        
        # Detect hardware info
        self.hw_info = detect_hardware_info()
        self.statusBar.showMessage(f"GPU: {self.hw_info.get('gpu_model', 'N/A')}, VRAM: {self.hw_info.get('vram_total', 'N/A')}MB")
        
        # Connect signals
        self.benchmark_tab.run_button.clicked.connect(self.run_benchmark)
    
    def init_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Add actions
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.refresh_data)
        toolbar.addAction(refresh_action)
    
    def refresh_data(self):
        # Refresh model list
        models = get_benchmark_models()
        self.benchmark_tab.update_model_list(models)
        
        # Refresh results
        self.results_tab.load_results()
    
    def run_benchmark(self):
        # Get parameters from UI
        selected_models = self.benchmark_tab.get_selected_models()
        is_test_coding = self.benchmark_tab.test_coding_checkbox.isChecked()
        
        if not selected_models:
            self.statusBar.showMessage("Error: No models selected")
            return
        
        # Build command
        cmd = "python benchmark.py"
        if selected_models:
            cmd += f" --models {' '.join(selected_models)}"
        if is_test_coding:
            cmd += " --test-coding"
        
        # Update UI
        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        self.benchmark_tab.output_text.clear()
        self.benchmark_tab.output_text.append(f"Running: {cmd}\n")
        
        # Create and start thread
        self.benchmark_thread = BenchmarkRunner(cmd)
        self.benchmark_thread.progress.connect(self.update_progress)
        self.benchmark_thread.completed.connect(self.benchmark_completed)
        self.benchmark_thread.start()
    
    def update_progress(self, line):
        self.benchmark_tab.output_text.append(line)
        cursor = self.benchmark_tab.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.benchmark_tab.output_text.setTextCursor(cursor)
    
    def benchmark_completed(self, code):
        self.progressBar.setValue(100)
        
        if code == 0:
            self.statusBar.showMessage("Benchmark completed successfully")
            # Refresh results tab
            self.results_tab.load_results()
            # Switch to results tab
            self.tabs.setCurrentIndex(1)
        else:
            self.statusBar.showMessage(f"Benchmark failed with code {code}")
```

#### Step 4: Create Benchmark Tab

`gui/benchmark_tab.py`:

```python
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QCheckBox, QLabel, QTextEdit,
    QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt

from benchmark import get_benchmark_models

class BenchmarkTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create splitter for top and bottom sections
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        # Create top section for controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_widget.setLayout(controls_layout)
        splitter.addWidget(controls_widget)
        
        # Create model selection group
        model_group = QGroupBox("Select Models")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        
        # Add model list
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.MultiSelection)
        model_layout.addWidget(self.model_list)
        
        # Add options
        self.test_coding_checkbox = QCheckBox("Run Coding Challenge Only")
        model_layout.addWidget(self.test_coding_checkbox)
        
        # Add run button
        self.run_button = QPushButton("Run Benchmark")
        model_layout.addWidget(self.run_button)
        
        controls_layout.addWidget(model_group)
        
        # Create options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        
        # Add benchmark options
        self.verbose_checkbox = QCheckBox("Verbose Output")
        options_layout.addWidget(self.verbose_checkbox)
        
        self.store_results_checkbox = QCheckBox("Store Results")
        self.store_results_checkbox.setChecked(True)
        options_layout.addWidget(self.store_results_checkbox)
        
        controls_layout.addWidget(options_group)
        
        # Create output section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        
        splitter.addWidget(output_group)
        
        # Load models
        self.update_model_list(get_benchmark_models())
    
    def update_model_list(self, models):
        self.model_list.clear()
        for model in models:
            self.model_list.addItem(model)
    
    def get_selected_models(self):
        selected = []
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            if item.isSelected():
                selected.append(item.text())
        return selected
```

#### Step 5: Create Results Tab

`gui/results_tab.py`:

```python
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QListWidget, QTableWidget, QTableWidgetItem, QLabel,
    QGroupBox, QSplitter, QComboBox
)
from PyQt5.QtCore import Qt

from benchmark_db import BenchmarkDB

class ResultsTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()
        self.load_results()
    
    def setup_ui(self):
        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create splitter for list and details
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Create run list section
        runs_group = QGroupBox("Benchmark Runs")
        runs_layout = QVBoxLayout()
        runs_group.setLayout(runs_layout)
        
        self.runs_list = QListWidget()
        runs_layout.addWidget(self.runs_list)
        
        self.delete_button = QPushButton("Delete Run")
        runs_layout.addWidget(self.delete_button)
        
        splitter.addWidget(runs_group)
        
        # Create details section
        details_group = QGroupBox("Run Details")
        details_layout = QVBoxLayout()
        details_group.setLayout(details_layout)
        
        # Model performance table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Model", "Gen. Speed", "Load Time", "Tokens", "Combined Speed", "Score"
        ])
        details_layout.addWidget(self.results_table)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export Report")
        controls_layout.addWidget(self.export_button)
        
        self.compare_label = QLabel("Compare with:")
        controls_layout.addWidget(self.compare_label)
        
        self.compare_combo = QComboBox()
        controls_layout.addWidget(self.compare_combo)
        
        self.compare_button = QPushButton("Compare")
        controls_layout.addWidget(self.compare_button)
        
        details_layout.addLayout(controls_layout)
        
        splitter.addWidget(details_group)
        
        # Connect signals
        self.runs_list.currentRowChanged.connect(self.show_run_details)
        self.delete_button.clicked.connect(self.delete_run)
        self.compare_button.clicked.connect(self.compare_runs)
    
    def load_results(self):
        # Load run list from database
        db = BenchmarkDB()
        self.runs = db.get_benchmark_runs()
        
        # Update list widget
        self.runs_list.clear()
        self.compare_combo.clear()
        
        for run in self.runs:
            run_id = run['id']
            timestamp = run['timestamp'].split('T')[0]
            gpu = run['gpu_model'] or 'N/A'
            
            item_text = f"Run {run_id}: {timestamp} ({gpu})"
            self.runs_list.addItem(item_text)
            self.compare_combo.addItem(item_text, run_id)
    
    def show_run_details(self, row):
        if row < 0 or row >= len(self.runs):
            return
            
        run_id = self.runs[row]['id']
        
        # Load results for this run
        db = BenchmarkDB()
        results = db.get_benchmark_results(run_id)
        
        # Update table
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(result['model_name']))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{result['generation_tokens_per_sec']:.2f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{result['model_load_time']:.2f}s"))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(result['generated_tokens'])))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{result['combined_tokens_per_sec']:.2f}"))
            
            # Calculate a simple score (placeholder)
            score = result['generation_tokens_per_sec'] * 0.7 + result['combined_tokens_per_sec'] * 0.3
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{score:.1f}"))
    
    def delete_run(self):
        # Implementation for deleting a run
        pass
    
    def compare_runs(self):
        # Implementation for comparing runs
        pass
```

#### Step 6: Install Dependencies

```bash
pip install PyQt5
```

#### Step 7: Run the Application

```bash
python benchmark_gui.py
```

## Data Visualization

### Adding Charts

For effective visualization, add charts to your GUI to display:

1. **Performance comparison**: Bar charts comparing token generation speed across models
2. **Speed vs. model size**: Scatter plot showing relationship between model size and performance
3. **Resource usage**: Line charts showing GPU memory usage during benchmarks

#### Web UI Example (Chart.js)

Add to your HTML:

```html
<canvas id="performance-chart" width="400" height="200"></canvas>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
function createPerformanceChart(models, speeds) {
    const ctx = document.getElementById('performance-chart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [{
                label: 'Generation Speed (tokens/sec)',
                data: speeds,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Tokens per Second'
                    }
                }
            }
        }
    });
    return chart;
}

// Example usage:
// createPerformanceChart(['model1', 'model2'], [24.5, 18.2]);
</script>
```

#### Desktop UI Example (PyQtChart)

Add charts to your PyQt application:

```python
from PyQt5.QtChart import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis
from PyQt5.QtCore import Qt

def create_performance_chart(parent, models, speeds):
    # Create chart
    chart = QChart()
    chart.setTitle("Model Performance Comparison")
    
    # Create bar set
    bar_set = QBarSet("Generation Speed (tokens/sec)")
    for speed in speeds:
        bar_set.append(speed)
    
    # Create series
    series = QBarSeries()
    series.append(bar_set)
    chart.addSeries(series)
    
    # Create axes
    axis_x = QBarCategoryAxis()
    axis_x.append(models)
    chart.addAxis(axis_x, Qt.AlignBottom)
    series.attachAxis(axis_x)
    
    axis_y = QValueAxis()
    axis_y.setTitleText("Tokens per Second")
    axis_y.setRange(0, max(speeds) * 1.1)
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_y)
    
    # Create chart view
    chart_view = QChartView(chart)
    chart_view.setRenderHint(QPainter.Antialiasing)
    
    return chart_view
```

## Real-time Monitoring

### Implementing Live Updates

For a more interactive experience, implement real-time monitoring of benchmark progress:

#### Web UI (WebSockets)

Use Flask-SocketIO to stream benchmark progress:

```python
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import subprocess

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_benchmark')
def start_benchmark(data):
    models = data.get('models', [])
    threading.Thread(target=run_benchmark, args=(models,)).start()

def run_benchmark(models):
    cmd = ['python', 'benchmark.py', '--models'] + models
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        socketio.emit('benchmark_output', {'line': line})
    
    process.wait()
    socketio.emit('benchmark_complete', {'code': process.returncode})

if __name__ == '__main__':
    socketio.run(app, debug=True)
```

#### Desktop UI (QThread with Signals)

Use QThread and signals for real-time updates:

```python
class BenchmarkRunner(QThread):
    progress = pyqtSignal(str)
    gpu_usage = pyqtSignal(float)  # Add signal for GPU usage
    completed = pyqtSignal(int)
    
    def __init__(self, command):
        super().__init__()
        self.command = command
    
    def run(self):
        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True
        )
        
        # Start GPU monitoring thread
        self.running = True
        monitoring_thread = threading.Thread(target=self.monitor_gpu)
        monitoring_thread.start()
        
        for line in process.stdout:
            self.progress.emit(line.strip())
        
        process.wait()
        self.running = False
        monitoring_thread.join()
        self.completed.emit(process.returncode)
    
    def monitor_gpu(self):
        while self.running:
            try:
                # Get GPU usage using nvidia-smi
                output = subprocess.check_output([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'
                ], text=True)
                
                # Parse GPU usage
                if output:
                    values = output.strip().split(',')
                    if len(values) >= 1:
                        gpu_util = float(values[0].strip())
                        self.gpu_usage.emit(gpu_util)
            except:
                pass
            
            time.sleep(1)
```

## Deployment Options

### Packaging Desktop Application

To distribute your desktop application:

```bash
# Install PyInstaller
pip install pyinstaller

# Create standalone executable
pyinstaller --onefile --windowed benchmark_gui.py
```

### Deploying Web UI to Server

For web UI deployment:

```bash
# Install Gunicorn (production WSGI server)
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 benchmark_web:app
```

For production environments, consider using Nginx as a reverse proxy. 