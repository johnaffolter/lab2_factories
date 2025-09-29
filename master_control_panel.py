#!/usr/bin/env python3

"""
Master Control Panel for MLOps Platform
Unified interface for managing all platform components
Integrates monitoring, deployment, maintenance, and recovery systems
"""

import os
import sys
import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file
from flask_cors import CORS
import logging

# Import all our components
sys.path.append('.')
from deployment_orchestrator import DeploymentOrchestrator
from cleanup_maintenance_utility import MaintenanceScheduler
from service_health_auto_recovery import ServiceMonitor, HealthCheck, RecoveryStrategy
from airbyte_connector_manager import AirbyteManager
from advanced_realtime_monitoring import RealTimeMonitor
from distributed_tracing_system import DistributedTracer
from anomaly_detection_engine import UnifiedAnomalyDetector
from database_pool_monitor import UnifiedPoolMonitor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)


class MasterControlPanel:
    """Master control panel for entire MLOps platform"""

    def __init__(self):
        self.deployment_orchestrator = DeploymentOrchestrator()
        self.maintenance_scheduler = MaintenanceScheduler()
        self.service_monitor = ServiceMonitor()
        self.airbyte_manager = AirbyteManager()
        self.realtime_monitor = RealTimeMonitor()
        self.tracer = DistributedTracer()
        self.anomaly_detector = UnifiedAnomalyDetector()
        self.pool_monitor = UnifiedPoolMonitor()

        self.platform_status = {
            "initialized": datetime.now().isoformat(),
            "services": {},
            "alerts": [],
            "metrics": {}
        }

        self._initialize_services()

    def _initialize_services(self):
        """Initialize all platform services"""
        logger.info("Initializing MLOps platform services...")

        # Register services for health monitoring
        services = [
            ("monitoring", "http", "http://localhost:8765/health"),
            ("neo4j", "tcp", 7687),
            ("postgres", "tcp", 5432),
            ("redis", "tcp", 6379),
            ("airflow", "http", "http://localhost:8080/health"),
            ("airbyte", "http", "http://localhost:8000/api/v1/health")
        ]

        for service_name, check_type, endpoint_or_port in services:
            if check_type == "http":
                health_check = HealthCheck(type="http", endpoint=endpoint_or_port)
            else:
                health_check = HealthCheck(type="tcp", port=endpoint_or_port)

            self.service_monitor.register_service(
                service_name,
                health_check,
                RecoveryStrategy(restart_attempts=3, restart_delay=10)
            )

        # Start monitoring
        self.service_monitor.start_monitoring()
        self.realtime_monitor.start_monitoring()
        self.pool_monitor.start_monitoring()

        logger.info("Platform services initialized")

    def get_dashboard_data(self) -> dict:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "services": self.service_monitor.get_status_summary(),
            "deployment": self.deployment_orchestrator.get_status(),
            "system_health": self.maintenance_scheduler.get_system_health(),
            "metrics": self.realtime_monitor.get_metrics_dashboard_data(),
            "anomalies": self.anomaly_detector.get_anomaly_summary(),
            "database_pools": self.pool_monitor.get_summary(),
            "traces": len(self.tracer.traces),
            "alerts": self.platform_status["alerts"][-10:]  # Last 10 alerts
        }

    def deploy_service(self, service_name: str, platform: str = "docker") -> dict:
        """Deploy a service"""
        logger.info(f"Deploying {service_name} to {platform}")
        results = self.deployment_orchestrator.deploy_all(
            platform=platform,
            services=[service_name]
        )

        if results:
            result = results[0]
            self._add_alert(f"Deployed {service_name} to {platform}", "info")
            return asdict(result)

        return {"status": "failed", "message": "Deployment failed"}

    def stop_service(self, service_name: str, platform: str = "docker") -> bool:
        """Stop a service"""
        logger.info(f"Stopping {service_name} on {platform}")
        success = self.deployment_orchestrator.stop_all(
            platform=platform,
            services=[service_name]
        )

        if success:
            self._add_alert(f"Stopped {service_name}", "info")

        return success

    def run_maintenance(self, level: str = "daily") -> dict:
        """Run maintenance tasks"""
        logger.info(f"Running {level} maintenance")

        if level == "daily":
            results = self.maintenance_scheduler.run_daily_maintenance()
        elif level == "weekly":
            results = self.maintenance_scheduler.run_weekly_maintenance()
        elif level == "monthly":
            results = self.maintenance_scheduler.run_monthly_maintenance()
        else:
            results = {}

        self._add_alert(f"Completed {level} maintenance", "info")
        return results

    def create_airbyte_connection(self, source: dict, destination: dict) -> str:
        """Create an Airbyte connection"""
        logger.info(f"Creating Airbyte connection: {source['name']} → {destination['name']}")

        # This would create actual Airbyte connections
        # Simplified for demonstration
        connection_id = f"conn_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._add_alert(f"Created Airbyte connection: {connection_id}", "info")
        return connection_id

    def trigger_airbyte_sync(self, connection_id: str) -> str:
        """Trigger an Airbyte sync"""
        logger.info(f"Triggering sync for connection {connection_id}")
        job_id = self.airbyte_manager.trigger_sync(connection_id)

        if job_id:
            self._add_alert(f"Started sync job: {job_id}", "info")
            return job_id

        return None

    def export_configuration(self) -> str:
        """Export platform configuration"""
        config = {
            "timestamp": datetime.now().isoformat(),
            "services": list(self.deployment_orchestrator.services.keys()),
            "deployment_status": self.deployment_orchestrator.get_status(),
            "health_status": self.service_monitor.get_status_summary(),
            "airbyte_connections": self.airbyte_manager.list_connections()
        }

        filepath = f"/tmp/mlops_config_{int(datetime.now().timestamp())}.json"
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Exported configuration to {filepath}")
        return filepath

    def _add_alert(self, message: str, severity: str = "info"):
        """Add an alert to the platform"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "severity": severity
        }
        self.platform_status["alerts"].append(alert)

        # Keep only last 100 alerts
        if len(self.platform_status["alerts"]) > 100:
            self.platform_status["alerts"] = self.platform_status["alerts"][-100:]

    def shutdown(self):
        """Shutdown all platform components"""
        logger.info("Shutting down MLOps platform...")

        self.service_monitor.stop_monitoring()
        self.realtime_monitor.stop_monitoring()
        self.pool_monitor.stop_monitoring()

        logger.info("Platform shutdown complete")


# Initialize control panel
control_panel = MasterControlPanel()


# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    return jsonify(control_panel.get_dashboard_data())


@app.route('/api/deploy', methods=['POST'])
def api_deploy():
    """Deploy a service"""
    data = request.json
    result = control_panel.deploy_service(
        data.get('service'),
        data.get('platform', 'docker')
    )
    return jsonify(result)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop a service"""
    data = request.json
    success = control_panel.stop_service(
        data.get('service'),
        data.get('platform', 'docker')
    )
    return jsonify({"success": success})


@app.route('/api/maintenance', methods=['POST'])
def api_maintenance():
    """Run maintenance"""
    data = request.json
    results = control_panel.run_maintenance(data.get('level', 'daily'))
    return jsonify(results)


@app.route('/api/airbyte/connection', methods=['POST'])
def api_create_connection():
    """Create Airbyte connection"""
    data = request.json
    connection_id = control_panel.create_airbyte_connection(
        data.get('source'),
        data.get('destination')
    )
    return jsonify({"connection_id": connection_id})


@app.route('/api/airbyte/sync', methods=['POST'])
def api_trigger_sync():
    """Trigger Airbyte sync"""
    data = request.json
    job_id = control_panel.trigger_airbyte_sync(data.get('connection_id'))
    return jsonify({"job_id": job_id})


@app.route('/api/export')
def api_export():
    """Export configuration"""
    filepath = control_panel.export_configuration()
    return send_file(filepath, as_attachment=True)


@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MLOps Master Control Panel</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold mb-8 text-center bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            🚀 MLOps Master Control Panel
        </h1>

        <!-- Quick Stats -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">Services</h3>
                <p class="text-3xl font-bold" id="serviceCount">0</p>
                <p class="text-sm text-gray-400" id="healthyServices">0 healthy</p>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">CPU Usage</h3>
                <p class="text-3xl font-bold" id="cpuUsage">0%</p>
                <p class="text-sm text-gray-400">System load</p>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">Memory</h3>
                <p class="text-3xl font-bold" id="memoryUsage">0%</p>
                <p class="text-sm text-gray-400">Used</p>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg">
                <h3 class="text-lg font-semibold mb-2">Anomalies</h3>
                <p class="text-3xl font-bold" id="anomalyCount">0</p>
                <p class="text-sm text-gray-400">Detected</p>
            </div>
        </div>

        <!-- Control Panels -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Deployment Control -->
            <div class="bg-gray-800 p-6 rounded-lg">
                <h2 class="text-xl font-semibold mb-4">Deployment Control</h2>
                <div class="space-y-3">
                    <select id="deployService" class="w-full p-2 bg-gray-700 rounded">
                        <option>monitoring</option>
                        <option>neo4j</option>
                        <option>postgres</option>
                        <option>redis</option>
                        <option>airflow</option>
                    </select>
                    <select id="deployPlatform" class="w-full p-2 bg-gray-700 rounded">
                        <option value="docker">Docker</option>
                        <option value="kubernetes">Kubernetes</option>
                        <option value="aws">AWS</option>
                    </select>
                    <div class="flex space-x-2">
                        <button onclick="deployService()" class="flex-1 px-4 py-2 bg-green-600 rounded hover:bg-green-700">
                            Deploy
                        </button>
                        <button onclick="stopService()" class="flex-1 px-4 py-2 bg-red-600 rounded hover:bg-red-700">
                            Stop
                        </button>
                    </div>
                </div>
            </div>

            <!-- Maintenance Control -->
            <div class="bg-gray-800 p-6 rounded-lg">
                <h2 class="text-xl font-semibold mb-4">Maintenance & Recovery</h2>
                <div class="space-y-3">
                    <button onclick="runMaintenance('daily')" class="w-full px-4 py-2 bg-blue-600 rounded hover:bg-blue-700">
                        Run Daily Maintenance
                    </button>
                    <button onclick="runMaintenance('weekly')" class="w-full px-4 py-2 bg-purple-600 rounded hover:bg-purple-700">
                        Run Weekly Maintenance
                    </button>
                    <button onclick="exportConfig()" class="w-full px-4 py-2 bg-gray-600 rounded hover:bg-gray-700">
                        Export Configuration
                    </button>
                </div>
            </div>
        </div>

        <!-- Service Status -->
        <div class="bg-gray-800 p-6 rounded-lg mb-8">
            <h2 class="text-xl font-semibold mb-4">Service Status</h2>
            <div id="serviceList" class="space-y-2">
                <!-- Services will be listed here -->
            </div>
        </div>

        <!-- Alerts -->
        <div class="bg-gray-800 p-6 rounded-lg">
            <h2 class="text-xl font-semibold mb-4">Recent Alerts</h2>
            <div id="alertList" class="space-y-2 max-h-64 overflow-y-auto">
                <!-- Alerts will be listed here -->
            </div>
        </div>
    </div>

    <script>
        // Update dashboard data
        async function updateDashboard() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();

                // Update stats
                document.getElementById('serviceCount').textContent = data.services.total_services || 0;
                document.getElementById('healthyServices').textContent = `${data.services.healthy || 0} healthy`;
                document.getElementById('cpuUsage').textContent = `${data.system_health?.cpu_percent || 0}%`;
                document.getElementById('memoryUsage').textContent = `${data.system_health?.memory_percent || 0}%`;
                document.getElementById('anomalyCount').textContent = data.anomalies?.total || 0;

                // Update service list
                const serviceList = document.getElementById('serviceList');
                serviceList.innerHTML = '';
                if (data.services?.services) {
                    data.services.services.forEach(service => {
                        const statusColor = service.status === 'healthy' ? 'text-green-400' :
                                          service.status === 'degraded' ? 'text-yellow-400' : 'text-red-400';
                        const div = document.createElement('div');
                        div.className = 'flex justify-between items-center p-3 bg-gray-700 rounded';
                        div.innerHTML = `
                            <span class="font-medium">${service.name}</span>
                            <span class="${statusColor}">${service.status}</span>
                        `;
                        serviceList.appendChild(div);
                    });
                }

                // Update alerts
                const alertList = document.getElementById('alertList');
                alertList.innerHTML = '';
                if (data.alerts) {
                    data.alerts.reverse().forEach(alert => {
                        const div = document.createElement('div');
                        div.className = 'p-3 bg-gray-700 rounded';
                        div.innerHTML = `
                            <div class="flex justify-between">
                                <span>${alert.message}</span>
                                <span class="text-xs text-gray-400">${new Date(alert.timestamp).toLocaleTimeString()}</span>
                            </div>
                        `;
                        alertList.appendChild(div);
                    });
                }
            } catch (error) {
                console.error('Failed to update dashboard:', error);
            }
        }

        // Service deployment
        async function deployService() {
            const service = document.getElementById('deployService').value;
            const platform = document.getElementById('deployPlatform').value;

            const response = await fetch('/api/deploy', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({service, platform})
            });

            const result = await response.json();
            alert(`Deployment ${result.status}: ${result.message || 'Success'}`);
            updateDashboard();
        }

        // Stop service
        async function stopService() {
            const service = document.getElementById('deployService').value;
            const platform = document.getElementById('deployPlatform').value;

            const response = await fetch('/api/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({service, platform})
            });

            const result = await response.json();
            alert(result.success ? 'Service stopped' : 'Failed to stop service');
            updateDashboard();
        }

        // Run maintenance
        async function runMaintenance(level) {
            const response = await fetch('/api/maintenance', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({level})
            });

            const result = await response.json();
            alert(`Maintenance completed: ${JSON.stringify(result).substring(0, 100)}...`);
            updateDashboard();
        }

        // Export configuration
        async function exportConfig() {
            window.location.href = '/api/export';
        }

        // Update dashboard every 5 seconds
        setInterval(updateDashboard, 5000);
        updateDashboard();
    </script>
</body>
</html>
"""


def main():
    """Run the master control panel"""
    print("🚀 MLOPS MASTER CONTROL PANEL")
    print("=" * 50)
    print("Starting unified control interface...")
    print()
    print("Access the control panel at: http://localhost:5000")
    print()
    print("Features:")
    print("  • Service deployment and management")
    print("  • Real-time monitoring and alerts")
    print("  • Automated maintenance and recovery")
    print("  • Airbyte data pipeline management")
    print("  • Comprehensive health checks")
    print()
    print("Press Ctrl+C to stop")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down control panel...")
        control_panel.shutdown()
        print("Shutdown complete")


if __name__ == "__main__":
    main()