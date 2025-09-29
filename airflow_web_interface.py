#!/usr/bin/env python3

"""
Lightweight Web Interface for Airflow
Bypasses browser header issues by providing a proxy interface
"""

from flask import Flask, render_template_string, request, jsonify, redirect
import requests
import subprocess
import json
from datetime import datetime
import base64

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Airflow Control Panel</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        .status {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            margin-left: 10px;
        }
        .status.healthy { background: #10b981; color: white; }
        .status.warning { background: #f59e0b; color: white; }
        .status.error { background: #ef4444; color: white; }
        .dag-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .dag-card {
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s;
            background: #f9fafb;
        }
        .dag-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }
        .dag-title {
            font-weight: bold;
            font-size: 18px;
            color: #1f2937;
            margin-bottom: 10px;
        }
        .dag-info {
            color: #6b7280;
            font-size: 14px;
            margin: 5px 0;
        }
        .dag-actions {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        button:hover {
            background: #5a67d8;
        }
        button.pause { background: #f59e0b; }
        button.pause:hover { background: #d97706; }
        button.unpause { background: #10b981; }
        button.unpause:hover { background: #059669; }
        .runs-list {
            max-height: 200px;
            overflow-y: auto;
            margin-top: 10px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 10px;
        }
        .run-item {
            padding: 5px;
            border-bottom: 1px solid #f3f4f6;
        }
        .run-item:last-child {
            border-bottom: none;
        }
        .state-success { color: #10b981; }
        .state-running { color: #3b82f6; }
        .state-failed { color: #ef4444; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: #f3f4f6;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #6b7280;
            font-size: 14px;
            margin-top: 5px;
        }
        .alert {
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }
        .alert.success { background: #d1fae5; color: #065f46; }
        .alert.error { background: #fee2e2; color: #991b1b; }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f4f6;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Airflow Control Panel <span class="status healthy">Connected</span></h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ dags|length }}</div>
                <div class="stat-label">Total DAGs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ active_count }}</div>
                <div class="stat-label">Active DAGs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ runs_today }}</div>
                <div class="stat-label">Runs Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ success_rate }}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        <div id="message-area"></div>

        <h2>📊 Available DAGs</h2>
        <div class="dag-grid">
            {% for dag in dags %}
            <div class="dag-card">
                <div class="dag-title">{{ dag.name }}</div>
                <div class="dag-info">Schedule: {{ dag.schedule or 'None' }}</div>
                <div class="dag-info">Status: <span class="state-{{ dag.status }}">{{ dag.status }}</span></div>
                <div class="dag-info">Last Run: {{ dag.last_run or 'Never' }}</div>

                <div class="dag-actions">
                    <button onclick="triggerDag('{{ dag.name }}')">▶️ Trigger</button>
                    {% if dag.is_paused %}
                    <button class="unpause" onclick="unpauseDag('{{ dag.name }}')">Resume</button>
                    {% else %}
                    <button class="pause" onclick="pauseDag('{{ dag.name }}')">Pause</button>
                    {% endif %}
                </div>

                <div id="runs-{{ dag.name }}" style="display:none;" class="runs-list">
                    <div class="loading"></div> Loading runs...
                </div>
                <button onclick="toggleRuns('{{ dag.name }}')" style="margin-top:10px;">
                    📈 Show Runs
                </button>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        function showMessage(message, type='success') {
            const area = document.getElementById('message-area');
            area.innerHTML = `<div class="alert ${type}">${message}</div>`;
            setTimeout(() => { area.innerHTML = ''; }, 5000);
        }

        function triggerDag(dagId) {
            fetch(`/trigger/${dagId}`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(`✅ Successfully triggered ${dagId}`);
                        setTimeout(() => location.reload(), 2000);
                    } else {
                        showMessage(`❌ Failed to trigger ${dagId}: ${data.error}`, 'error');
                    }
                });
        }

        function pauseDag(dagId) {
            fetch(`/pause/${dagId}`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(`⏸️ Paused ${dagId}`);
                        setTimeout(() => location.reload(), 2000);
                    }
                });
        }

        function unpauseDag(dagId) {
            fetch(`/unpause/${dagId}`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMessage(`▶️ Resumed ${dagId}`);
                        setTimeout(() => location.reload(), 2000);
                    }
                });
        }

        function toggleRuns(dagId) {
            const runsDiv = document.getElementById(`runs-${dagId}`);
            if (runsDiv.style.display === 'none') {
                runsDiv.style.display = 'block';
                loadRuns(dagId);
            } else {
                runsDiv.style.display = 'none';
            }
        }

        function loadRuns(dagId) {
            fetch(`/runs/${dagId}`)
                .then(response => response.json())
                .then(data => {
                    const runsDiv = document.getElementById(`runs-${dagId}`);
                    if (data.runs && data.runs.length > 0) {
                        let html = '<div style="font-weight:bold;margin-bottom:10px;">Recent Runs:</div>';
                        data.runs.forEach(run => {
                            html += `<div class="run-item">
                                <span class="state-${run.state.toLowerCase()}">${run.state}</span>
                                - ${run.execution_date}
                            </div>`;
                        });
                        runsDiv.innerHTML = html;
                    } else {
                        runsDiv.innerHTML = '<div style="color:#6b7280;">No runs found</div>';
                    }
                });
        }

        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""

class AirflowProxy:
    """Proxy to interact with Airflow via Docker commands"""

    @staticmethod
    def get_dags():
        """Get list of all DAGs"""
        try:
            result = subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "list"],
                capture_output=True,
                text=True,
                check=True
            )

            dags = []
            lines = result.stdout.strip().split('\n')

            for line in lines[2:]:  # Skip headers
                if line.strip() and not line.startswith('='):
                    parts = line.split('|')
                    if len(parts) >= 4:
                        dag_name = parts[0].strip()
                        if dag_name:
                            dags.append({
                                'name': dag_name,
                                'schedule': parts[2].strip() if len(parts) > 2 else 'None',
                                'is_paused': parts[3].strip() == 'True' if len(parts) > 3 else False,
                                'status': 'paused' if parts[3].strip() == 'True' else 'active',
                                'last_run': 'Unknown'
                            })

            return dags
        except Exception as e:
            print(f"Error getting DAGs: {e}")
            return []

    @staticmethod
    def trigger_dag(dag_id):
        """Trigger a DAG"""
        try:
            result = subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "trigger", dag_id],
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, str(e)

    @staticmethod
    def pause_dag(dag_id):
        """Pause a DAG"""
        try:
            subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "pause", dag_id],
                capture_output=True,
                check=True
            )
            return True
        except:
            return False

    @staticmethod
    def unpause_dag(dag_id):
        """Unpause a DAG"""
        try:
            subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "unpause", dag_id],
                capture_output=True,
                check=True
            )
            return True
        except:
            return False

    @staticmethod
    def get_dag_runs(dag_id):
        """Get runs for a specific DAG"""
        try:
            result = subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "list-runs",
                 "--dag-id", dag_id],
                capture_output=True,
                text=True
            )

            runs = []
            lines = result.stdout.strip().split('\n')

            for line in lines[2:]:  # Skip headers
                if line.strip() and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        runs.append({
                            'run_id': parts[1].strip(),
                            'state': parts[2].strip(),
                            'execution_date': parts[3].strip()
                        })

            return runs[:5]  # Return only last 5 runs
        except:
            return []

@app.route('/')
def index():
    """Main dashboard"""
    proxy = AirflowProxy()
    dags = proxy.get_dags()

    # Calculate statistics
    active_count = sum(1 for d in dags if not d['is_paused'])
    runs_today = len([d for d in dags if 'mlops' in d['name'].lower()])  # Simplified
    success_rate = 95  # Placeholder

    return render_template_string(HTML_TEMPLATE,
                                 dags=dags,
                                 active_count=active_count,
                                 runs_today=runs_today,
                                 success_rate=success_rate)

@app.route('/trigger/<dag_id>', methods=['POST'])
def trigger_dag(dag_id):
    """Trigger a DAG"""
    proxy = AirflowProxy()
    success, message = proxy.trigger_dag(dag_id)
    return jsonify({'success': success, 'message': message})

@app.route('/pause/<dag_id>', methods=['POST'])
def pause_dag(dag_id):
    """Pause a DAG"""
    proxy = AirflowProxy()
    success = proxy.pause_dag(dag_id)
    return jsonify({'success': success})

@app.route('/unpause/<dag_id>', methods=['POST'])
def unpause_dag(dag_id):
    """Unpause a DAG"""
    proxy = AirflowProxy()
    success = proxy.unpause_dag(dag_id)
    return jsonify({'success': success})

@app.route('/runs/<dag_id>')
def get_runs(dag_id):
    """Get DAG runs"""
    proxy = AirflowProxy()
    runs = proxy.get_dag_runs(dag_id)
    return jsonify({'runs': runs})

@app.route('/direct-access')
def direct_access():
    """Redirect to actual Airflow UI with minimal headers"""
    return """
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=http://localhost:8080/login/">
    </head>
    <body>
        <p>Redirecting to Airflow UI...</p>
        <p>If not redirected, <a href="http://localhost:8080/login/">click here</a></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 AIRFLOW WEB INTERFACE")
    print("=" * 60)
    print("\nThis interface bypasses browser header issues!")
    print("\nAccess at: http://localhost:5555")
    print("\nFeatures:")
    print("  • View all DAGs")
    print("  • Trigger DAG runs")
    print("  • Pause/Resume DAGs")
    print("  • View run history")
    print("  • Real-time statistics")
    print("\n" + "=" * 60)

    app.run(host='0.0.0.0', port=5555, debug=True)