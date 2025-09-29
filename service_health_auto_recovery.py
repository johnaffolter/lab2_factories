#!/usr/bin/env python3

"""
Service Health Checker with Auto-Recovery
Monitors service health and automatically recovers failed services
Implements circuit breaker pattern and intelligent recovery strategies
"""

import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import requests
import psutil
import subprocess

# Docker support
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enum"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    STOPPED = "stopped"


@dataclass
class HealthCheck:
    """Health check configuration"""
    type: str  # http, tcp, docker, process
    endpoint: str = None
    port: int = None
    timeout: int = 5
    interval: int = 30
    retries: int = 3
    success_threshold: int = 2
    failure_threshold: int = 3


@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_time_ms: float = None
    error_message: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStrategy:
    """Recovery strategy for a service"""
    restart_attempts: int = 3
    restart_delay: int = 10
    escalation_threshold: int = 5
    circuit_breaker_timeout: int = 300
    auto_scale: bool = False
    fallback_service: str = None


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def record_success(self):
        """Record a successful operation"""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed")

    def record_failure(self):
        """Record a failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def can_attempt(self) -> bool:
        """Check if operation can be attempted"""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).seconds
                if time_since_failure > self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker half-open, attempting recovery")
                    return True
            return False

        return self.state == "half_open"


class HealthChecker:
    """Performs health checks on services"""

    def __init__(self):
        self.docker_client = None
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
            except:
                pass

    async def check_http_health(self, endpoint: str, timeout: int = 5) -> Tuple[bool, float, str]:
        """Check HTTP endpoint health"""
        start_time = time.time()
        try:
            response = requests.get(endpoint, timeout=timeout)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return True, response_time, "OK"
            else:
                return False, response_time, f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return False, timeout * 1000, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, 0, "Connection failed"
        except Exception as e:
            return False, 0, str(e)

    async def check_tcp_health(self, host: str, port: int, timeout: int = 5) -> Tuple[bool, float, str]:
        """Check TCP port health"""
        import socket

        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000

            if result == 0:
                return True, response_time, "Port open"
            else:
                return False, response_time, "Port closed"

        except socket.timeout:
            return False, timeout * 1000, "Timeout"
        except Exception as e:
            return False, 0, str(e)
        finally:
            sock.close()

    async def check_docker_health(self, container_name: str) -> Tuple[bool, float, str]:
        """Check Docker container health"""
        if not self.docker_client:
            return False, 0, "Docker not available"

        try:
            container = self.docker_client.containers.get(container_name)
            status = container.status

            if status == "running":
                # Check container health status if available
                health = container.attrs.get("State", {}).get("Health", {})
                if health:
                    health_status = health.get("Status", "none")
                    if health_status == "healthy":
                        return True, 0, "Healthy"
                    elif health_status == "unhealthy":
                        return False, 0, "Unhealthy"
                    else:
                        return True, 0, "Running (no health check)"
                else:
                    return True, 0, "Running"
            else:
                return False, 0, f"Container {status}"

        except docker.errors.NotFound:
            return False, 0, "Container not found"
        except Exception as e:
            return False, 0, str(e)

    async def check_process_health(self, process_name: str) -> Tuple[bool, float, str]:
        """Check if process is running"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if process_name.lower() in proc.info['name'].lower():
                    # Check process responsiveness
                    process = psutil.Process(proc.info['pid'])
                    cpu_percent = process.cpu_percent(interval=0.1)
                    memory_percent = process.memory_percent()

                    if cpu_percent > 90:
                        return False, cpu_percent, "High CPU usage"
                    elif memory_percent > 90:
                        return False, memory_percent, "High memory usage"
                    else:
                        return True, cpu_percent, "Running"

            return False, 0, "Process not found"

        except Exception as e:
            return False, 0, str(e)


class RecoveryManager:
    """Manages service recovery operations"""

    def __init__(self):
        self.docker_client = None
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
            except:
                pass

        self.recovery_history = []

    async def restart_service(self, service_name: str, service_type: str = "docker") -> bool:
        """Restart a service"""
        logger.info(f"Attempting to restart {service_name} ({service_type})")

        if service_type == "docker":
            return await self._restart_docker_container(service_name)
        elif service_type == "systemd":
            return await self._restart_systemd_service(service_name)
        elif service_type == "process":
            return await self._restart_process(service_name)
        else:
            logger.error(f"Unknown service type: {service_type}")
            return False

    async def _restart_docker_container(self, container_name: str) -> bool:
        """Restart Docker container"""
        if not self.docker_client:
            return False

        try:
            container = self.docker_client.containers.get(container_name)
            container.restart()
            logger.info(f"Restarted container {container_name}")

            # Wait for container to be healthy
            for _ in range(30):
                await asyncio.sleep(2)
                if container.status == "running":
                    return True

            return False

        except docker.errors.NotFound:
            logger.error(f"Container {container_name} not found")
            return await self._create_docker_container(container_name)
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False

    async def _create_docker_container(self, container_name: str) -> bool:
        """Create and start a new Docker container"""
        # This would use stored configuration to recreate the container
        logger.info(f"Attempting to create container {container_name}")
        # Implementation would go here based on stored configs
        return False

    async def _restart_systemd_service(self, service_name: str) -> bool:
        """Restart systemd service"""
        try:
            result = subprocess.run(
                ["systemctl", "restart", service_name],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Restarted systemd service {service_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart systemd service: {e}")
            return False

    async def _restart_process(self, process_name: str) -> bool:
        """Restart a process"""
        # This would need process configuration to know how to start it
        logger.info(f"Would restart process {process_name}")
        return False

    async def scale_service(self, service_name: str, replicas: int) -> bool:
        """Scale a service"""
        logger.info(f"Scaling {service_name} to {replicas} replicas")
        # Implementation for Docker Swarm or Kubernetes
        return False

    def record_recovery(self, service_name: str, success: bool, strategy: str):
        """Record recovery attempt"""
        self.recovery_history.append({
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "success": success,
            "strategy": strategy
        })


class ServiceMonitor:
    """Main service monitoring and recovery orchestrator"""

    def __init__(self):
        self.services: Dict[str, Dict] = {}
        self.health_checker = HealthChecker()
        self.recovery_manager = RecoveryManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.monitoring_active = False
        self.alert_callbacks: List[Callable] = []

    def register_service(self, name: str, health_check: HealthCheck,
                        recovery_strategy: RecoveryStrategy = None):
        """Register a service for monitoring"""
        self.services[name] = {
            "health_check": health_check,
            "recovery_strategy": recovery_strategy or RecoveryStrategy()
        }

        self.circuit_breakers[name] = CircuitBreaker(
            failure_threshold=health_check.failure_threshold,
            recovery_timeout=recovery_strategy.circuit_breaker_timeout if recovery_strategy else 60
        )

        self.service_health[name] = ServiceHealth(
            service_name=name,
            status=ServiceStatus.STOPPED,
            last_check=datetime.now()
        )

        logger.info(f"Registered service: {name}")

    async def check_service_health(self, name: str) -> ServiceHealth:
        """Check health of a specific service"""
        if name not in self.services:
            return None

        config = self.services[name]
        health_check = config["health_check"]

        # Perform health check based on type
        if health_check.type == "http":
            healthy, response_time, message = await self.health_checker.check_http_health(
                health_check.endpoint,
                health_check.timeout
            )
        elif health_check.type == "tcp":
            healthy, response_time, message = await self.health_checker.check_tcp_health(
                "localhost",
                health_check.port,
                health_check.timeout
            )
        elif health_check.type == "docker":
            healthy, response_time, message = await self.health_checker.check_docker_health(
                name
            )
        elif health_check.type == "process":
            healthy, response_time, message = await self.health_checker.check_process_health(
                name
            )
        else:
            healthy, response_time, message = False, 0, "Unknown check type"

        # Update health status
        health = self.service_health[name]
        health.last_check = datetime.now()
        health.response_time_ms = response_time
        health.error_message = message if not healthy else None

        if healthy:
            health.consecutive_failures = 0
            health.consecutive_successes += 1

            if health.consecutive_successes >= health_check.success_threshold:
                health.status = ServiceStatus.HEALTHY
                self.circuit_breakers[name].record_success()
        else:
            health.consecutive_successes = 0
            health.consecutive_failures += 1

            if health.consecutive_failures >= health_check.failure_threshold:
                health.status = ServiceStatus.UNHEALTHY
                self.circuit_breakers[name].record_failure()
                await self._trigger_recovery(name)
            elif health.consecutive_failures > 1:
                health.status = ServiceStatus.DEGRADED

        logger.debug(f"Health check {name}: {health.status.value} - {message}")
        return health

    async def _trigger_recovery(self, service_name: str):
        """Trigger recovery for a service"""
        if not self.circuit_breakers[service_name].can_attempt():
            logger.warning(f"Circuit breaker open for {service_name}, skipping recovery")
            return

        health = self.service_health[service_name]
        health.status = ServiceStatus.RECOVERING

        strategy = self.services[service_name]["recovery_strategy"]

        # Try restart
        for attempt in range(strategy.restart_attempts):
            logger.info(f"Recovery attempt {attempt + 1} for {service_name}")

            success = await self.recovery_manager.restart_service(service_name)

            if success:
                await asyncio.sleep(strategy.restart_delay)

                # Verify service is healthy
                check_health = await self.check_service_health(service_name)
                if check_health.status == ServiceStatus.HEALTHY:
                    logger.info(f"Successfully recovered {service_name}")
                    self.recovery_manager.record_recovery(service_name, True, "restart")
                    self._send_alert(f"Service recovered: {service_name}", "info")
                    return

            await asyncio.sleep(strategy.restart_delay)

        # If restart failed, try fallback
        if strategy.fallback_service:
            logger.warning(f"Activating fallback service for {service_name}")
            # Activate fallback service logic here

        # Record failure
        self.recovery_manager.record_recovery(service_name, False, "restart")
        self._send_alert(f"Failed to recover service: {service_name}", "critical")

    async def monitor_all_services(self):
        """Monitor all registered services"""
        while self.monitoring_active:
            tasks = []
            for service_name in self.services:
                tasks.append(self.check_service_health(service_name))

            await asyncio.gather(*tasks)

            # Wait for next check interval
            await asyncio.sleep(10)  # Check every 10 seconds

    def start_monitoring(self):
        """Start monitoring in background thread"""
        self.monitoring_active = True

        def run_async_monitoring():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.monitor_all_services())

        thread = threading.Thread(target=run_async_monitoring, daemon=True)
        thread.start()
        logger.info("Started service monitoring")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("Stopped service monitoring")

    def register_alert_callback(self, callback: Callable):
        """Register alert callback"""
        self.alert_callbacks.append(callback)

    def _send_alert(self, message: str, severity: str):
        """Send alert to registered callbacks"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "severity": severity
        }

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all service statuses"""
        summary = {
            "total_services": len(self.services),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "recovering": 0,
            "services": []
        }

        for name, health in self.service_health.items():
            if health.status == ServiceStatus.HEALTHY:
                summary["healthy"] += 1
            elif health.status == ServiceStatus.DEGRADED:
                summary["degraded"] += 1
            elif health.status == ServiceStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            elif health.status == ServiceStatus.RECOVERING:
                summary["recovering"] += 1

            summary["services"].append({
                "name": name,
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms,
                "consecutive_failures": health.consecutive_failures
            })

        return summary


def setup_default_services(monitor: ServiceMonitor):
    """Set up default service monitoring"""

    # Monitor Neo4j
    monitor.register_service(
        "neo4j",
        HealthCheck(
            type="tcp",
            port=7687,
            interval=30,
            timeout=5
        ),
        RecoveryStrategy(
            restart_attempts=3,
            restart_delay=10
        )
    )

    # Monitor PostgreSQL
    monitor.register_service(
        "postgres",
        HealthCheck(
            type="tcp",
            port=5432,
            interval=30,
            timeout=5
        ),
        RecoveryStrategy(
            restart_attempts=3,
            restart_delay=15
        )
    )

    # Monitor Redis
    monitor.register_service(
        "redis",
        HealthCheck(
            type="tcp",
            port=6379,
            interval=20,
            timeout=3
        ),
        RecoveryStrategy(
            restart_attempts=5,
            restart_delay=5
        )
    )

    # Monitor API
    monitor.register_service(
        "api",
        HealthCheck(
            type="http",
            endpoint="http://localhost:8000/health",
            interval=15,
            timeout=5
        ),
        RecoveryStrategy(
            restart_attempts=3,
            restart_delay=10,
            auto_scale=True
        )
    )

    # Monitor Airflow
    monitor.register_service(
        "airflow",
        HealthCheck(
            type="http",
            endpoint="http://localhost:8080/health",
            interval=60,
            timeout=10
        ),
        RecoveryStrategy(
            restart_attempts=2,
            restart_delay=30
        )
    )


async def demo_monitoring():
    """Demo service monitoring"""
    print("🏥 SERVICE HEALTH MONITORING & AUTO-RECOVERY")
    print("=" * 50)

    monitor = ServiceMonitor()

    # Register alert handler
    def alert_handler(alert):
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }
        icon = severity_icons.get(alert["severity"], "❓")
        print(f"{icon} ALERT: {alert['message']}")

    monitor.register_alert_callback(alert_handler)

    # Set up services
    setup_default_services(monitor)

    # Start monitoring
    monitor.start_monitoring()
    print("✅ Monitoring started for 5 services")
    print()

    # Run for demonstration
    for i in range(6):
        await asyncio.sleep(10)

        # Get status
        status = monitor.get_status_summary()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
        print(f"  Healthy: {status['healthy']}")
        print(f"  Degraded: {status['degraded']}")
        print(f"  Unhealthy: {status['unhealthy']}")
        print(f"  Recovering: {status['recovering']}")

        for service in status["services"][:3]:  # Show first 3
            status_icon = "✅" if service["status"] == "healthy" else "❌"
            response_time = service.get("response_time_ms", 0)
            print(f"    {status_icon} {service['name']}: {service['status']} ({response_time:.1f}ms)")

    # Stop monitoring
    monitor.stop_monitoring()
    print("\n✅ Monitoring stopped")

    # Show recovery history
    if monitor.recovery_manager.recovery_history:
        print("\n📊 Recovery History:")
        for recovery in monitor.recovery_manager.recovery_history:
            result = "Success" if recovery["success"] else "Failed"
            print(f"  {recovery['service']}: {result} ({recovery['strategy']})")


def main():
    """Main entry point"""
    asyncio.run(demo_monitoring())


if __name__ == "__main__":
    main()