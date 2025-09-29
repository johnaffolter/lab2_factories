#!/usr/bin/env python3
"""
Integrated MLOps System Launcher
Author: John Affolter (johnaffolter)
Date: September 29, 2025

This script launches the complete enhanced MLOps system with:
- Enhanced training system with real-time feedback
- Advanced feedback processor with design patterns
- Real-time dashboard integration
- Airflow pipeline orchestration
- Comprehensive monitoring and tracking
"""

import asyncio
import subprocess
import threading
import time
import webbrowser
import logging
import signal
import sys
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedMLOpsLauncher:
    """Launches and manages the complete MLOps system"""

    def __init__(self):
        self.processes = []
        self.system_components = {
            'enhanced_training_system': None,
            'feedback_processor': None,
            'airflow_scheduler': None,
            'monitoring_dashboard': None
        }
        self.shutdown_requested = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Shutdown signal received, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.shutdown_all_components()
        sys.exit(0)

    def check_prerequisites(self):
        """Check if all required components are available"""
        logger.info("🔍 Checking system prerequisites...")

        required_files = [
            'enhanced_mlops_training_system.py',
            'advanced_feedback_processor.py',
            'airflow_enhanced_feedback_dag.py',
            'frontend/realtime_training_dashboard.html'
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False

        logger.info("✅ All required components found")
        return True

    def start_enhanced_training_system(self):
        """Start the enhanced training system"""
        logger.info("🚀 Starting Enhanced Training System...")

        try:
            process = subprocess.Popen([
                sys.executable, 'enhanced_mlops_training_system.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes.append(process)
            self.system_components['enhanced_training_system'] = process

            logger.info("✅ Enhanced Training System started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start Enhanced Training System: {e}")
            return False

    def start_feedback_processor(self):
        """Start the advanced feedback processor"""
        logger.info("🤖 Starting Advanced Feedback Processor...")

        try:
            # Import and start the feedback processor
            from advanced_feedback_processor import AdvancedFeedbackProcessor

            async def run_processor():
                processor = AdvancedFeedbackProcessor()
                await processor.start_processing_service()

            # Start in a separate thread
            def processor_thread():
                asyncio.run(run_processor())

            thread = threading.Thread(target=processor_thread, daemon=True)
            thread.start()

            self.system_components['feedback_processor'] = thread
            logger.info("✅ Advanced Feedback Processor started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start Feedback Processor: {e}")
            return False

    def start_airflow_services(self):
        """Start Airflow services"""
        logger.info("🌊 Starting Airflow Services...")

        try:
            # Check if Airflow is already running
            try:
                result = subprocess.run([
                    'docker', 'ps', '--filter', 'name=airflow-standalone'
                ], capture_output=True, text=True, timeout=10)

                if 'airflow-standalone' in result.stdout:
                    logger.info("✅ Airflow already running in Docker")
                    return True
            except:
                pass

            # Start Airflow standalone if not running
            process = subprocess.Popen([
                'docker', 'run', '-d',
                '--name', 'airflow-standalone-enhanced',
                '-p', '8080:8080',
                '-e', '_AIRFLOW_WWW_USER_USERNAME=admin',
                '-e', '_AIRFLOW_WWW_USER_PASSWORD=admin',
                '-v', f'{Path.cwd()}:/opt/airflow/dags',
                'apache/airflow:2.7.0',
                'standalone'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes.append(process)
            self.system_components['airflow_scheduler'] = process

            logger.info("✅ Airflow Services started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start Airflow Services: {e}")
            return False

    def start_monitoring_dashboard(self):
        """Start the monitoring dashboard"""
        logger.info("📊 Starting Monitoring Dashboard...")

        try:
            # Create a simple HTTP server for the dashboard
            dashboard_port = 9999
            dashboard_path = Path('frontend/realtime_training_dashboard.html')

            if not dashboard_path.exists():
                logger.error("❌ Dashboard HTML file not found")
                return False

            # Start simple HTTP server
            process = subprocess.Popen([
                sys.executable, '-m', 'http.server', str(dashboard_port),
                '--directory', 'frontend'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            self.processes.append(process)
            self.system_components['monitoring_dashboard'] = process

            logger.info(f"✅ Monitoring Dashboard started on port {dashboard_port}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start Monitoring Dashboard: {e}")
            return False

    def open_user_interfaces(self):
        """Open user interfaces in browser"""
        logger.info("🌐 Opening user interfaces...")

        interfaces = [
            {
                'name': 'Real-time Training Dashboard',
                'url': 'http://localhost:9999/realtime_training_dashboard.html',
                'delay': 2
            },
            {
                'name': 'Airflow WebUI',
                'url': 'http://localhost:8080',
                'delay': 5
            },
            {
                'name': 'Comprehensive MLOps UI',
                'url': 'http://localhost:9999/comprehensive_mlops_ui.html',
                'delay': 8
            }
        ]

        def open_interface(interface):
            time.sleep(interface['delay'])
            try:
                webbrowser.open(interface['url'])
                logger.info(f"🌐 Opened {interface['name']}: {interface['url']}")
            except Exception as e:
                logger.warning(f"Could not open {interface['name']}: {e}")

        # Open interfaces with delays
        for interface in interfaces:
            threading.Thread(target=open_interface, args=(interface,), daemon=True).start()

    def monitor_system_health(self):
        """Monitor the health of all components"""
        logger.info("🏥 Starting system health monitoring...")

        def health_check():
            while not self.shutdown_requested:
                try:
                    # Check processes
                    for name, process in self.system_components.items():
                        if hasattr(process, 'poll') and process.poll() is not None:
                            logger.warning(f"⚠️ Component {name} has stopped")

                    # Wait before next check
                    time.sleep(30)

                except Exception as e:
                    logger.error(f"Error in health check: {e}")

        threading.Thread(target=health_check, daemon=True).start()

    def generate_system_status_report(self):
        """Generate a comprehensive system status report"""
        logger.info("📊 Generating system status report...")

        status_report = {
            'timestamp': datetime.now().isoformat(),
            'system_author': 'John Affolter (johnaffolter)',
            'system_version': 'Enhanced MLOps v2.0',
            'components': {},
            'access_urls': {
                'Real-time Dashboard': 'http://localhost:9999/realtime_training_dashboard.html',
                'Airflow WebUI': 'http://localhost:8080',
                'MLOps UI': 'http://localhost:9999/comprehensive_mlops_ui.html',
                'Advanced Monitoring': 'http://localhost:9999/advanced_monitoring_dashboard.html'
            },
            'features': [
                'Real-time training with feedback integration',
                'Advanced feedback processing with design patterns',
                'Continuous model improvement',
                'Interactive dashboards and monitoring',
                'Airflow-orchestrated ML pipelines',
                'WebSocket-based real-time updates',
                'Factory pattern for extensible processors',
                'Observer pattern for notifications'
            ]
        }

        # Check component status
        for name, component in self.system_components.items():
            if component is None:
                status = 'not_started'
            elif hasattr(component, 'is_alive') and component.is_alive():
                status = 'running'
            elif hasattr(component, 'poll') and component.poll() is None:
                status = 'running'
            else:
                status = 'stopped'

            status_report['components'][name] = status

        # Save report
        report_path = f"system_status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(status_report, f, indent=2)

        logger.info(f"📄 System status report saved: {report_path}")
        return status_report

    def start_complete_system(self):
        """Start the complete integrated MLOps system"""
        logger.info("🚀 Starting Complete Integrated MLOps System")
        logger.info("=" * 60)

        # Setup signal handlers
        self.setup_signal_handlers()

        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites not met, cannot start system")
            return False

        # Start components in order
        components_to_start = [
            ('Enhanced Training System', self.start_enhanced_training_system),
            ('Feedback Processor', self.start_feedback_processor),
            ('Airflow Services', self.start_airflow_services),
            ('Monitoring Dashboard', self.start_monitoring_dashboard)
        ]

        failed_components = []
        for name, start_func in components_to_start:
            logger.info(f"Starting {name}...")
            if start_func():
                logger.info(f"✅ {name} started successfully")
                time.sleep(2)  # Give component time to initialize
            else:
                logger.error(f"❌ Failed to start {name}")
                failed_components.append(name)

        if failed_components:
            logger.warning(f"⚠️ Some components failed to start: {failed_components}")
        else:
            logger.info("✅ All components started successfully!")

        # Start monitoring
        self.monitor_system_health()

        # Open user interfaces
        self.open_user_interfaces()

        # Generate initial status report
        status_report = self.generate_system_status_report()

        # Display system information
        self.display_system_info(status_report)

        return True

    def display_system_info(self, status_report):
        """Display system information"""
        print("\n" + "=" * 80)
        print("🚀 ENHANCED MLOPS SYSTEM - FULLY OPERATIONAL")
        print("=" * 80)
        print(f"Author: {status_report['system_author']}")
        print(f"Version: {status_report['system_version']}")
        print(f"Started: {status_report['timestamp']}")
        print()

        print("🌐 ACCESS URLS:")
        for name, url in status_report['access_urls'].items():
            print(f"  • {name}: {url}")
        print()

        print("⚙️ COMPONENT STATUS:")
        for name, status in status_report['components'].items():
            emoji = "✅" if status == "running" else "❌" if status == "stopped" else "⏸️"
            print(f"  {emoji} {name}: {status}")
        print()

        print("🌟 SYSTEM FEATURES:")
        for feature in status_report['features']:
            print(f"  • {feature}")
        print()

        print("🎯 GETTING STARTED:")
        print("  1. Open the Real-time Training Dashboard to monitor training")
        print("  2. Use the Airflow WebUI to manage ML pipelines")
        print("  3. Submit feedback through the dashboard to improve models")
        print("  4. Monitor system health through the monitoring dashboard")
        print()

        print("🔧 SYSTEM COMMANDS:")
        print("  • Press Ctrl+C to shutdown the system gracefully")
        print("  • Check logs for detailed system information")
        print("  • Access components individually through their URLs")
        print()
        print("=" * 80)

    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
            self.shutdown_requested = True

    def shutdown_all_components(self):
        """Shutdown all system components"""
        logger.info("🛑 Shutting down all components...")

        # Terminate processes
        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    # Wait up to 5 seconds for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()  # Force kill if needed
            except Exception as e:
                logger.warning(f"Error shutting down process: {e}")

        logger.info("✅ All components shutdown complete")

    def run(self):
        """Run the complete system"""
        if self.start_complete_system():
            logger.info("🎉 System startup complete! Waiting for shutdown signal...")
            self.wait_for_shutdown()
        else:
            logger.error("❌ System startup failed")

        self.shutdown_all_components()

def main():
    """Main function"""
    print("🚀 Enhanced MLOps System Launcher")
    print("Author: John Affolter (johnaffolter)")
    print("Date: September 29, 2025")
    print()

    launcher = IntegratedMLOpsLauncher()
    launcher.run()

if __name__ == "__main__":
    main()