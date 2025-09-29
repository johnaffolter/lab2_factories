#!/usr/bin/env python3

"""
Real-Time Monitoring WebSocket Server
Provides live monitoring data via WebSocket connections for dashboard updates
REAL METRICS - NO SIMULATIONS
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Set, Any
import logging

# WebSocket and async imports
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets
    from websockets.server import WebSocketServerProtocol

# System monitoring
import psutil
import platform

# HTTP client for API monitoring
import requests

# AWS SDK for CloudWatch
try:
    import boto3
except ImportError:
    print("Installing boto3...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
    import boto3

# Database monitoring imports
try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseConnectionMonitor:
    """Monitor real database connection pools and performance"""

    def __init__(self):
        self.connections = {}
        self.performance_metrics = {}

    def monitor_neo4j(self) -> Dict[str, Any]:
        """Monitor real Neo4j database connections"""
        if not NEO4J_AVAILABLE:
            return {"status": "not_available", "reason": "neo4j-driver not installed"}

        try:
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            username = os.getenv('NEO4J_USERNAME', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'password')

            driver = GraphDatabase.driver(uri, auth=(username, password))

            # Test connection and measure latency
            start_time = time.time()
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as node_count LIMIT 1")
                node_count = result.single()['node_count']
            latency = (time.time() - start_time) * 1000  # ms

            driver.close()

            return {
                "status": "connected",
                "node_count": node_count,
                "latency_ms": round(latency, 2),
                "uri": uri,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Neo4j monitoring error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def monitor_snowflake(self) -> Dict[str, Any]:
        """Monitor real Snowflake data warehouse connections"""
        if not SNOWFLAKE_AVAILABLE:
            return {"status": "not_available", "reason": "snowflake-connector-python not installed"}

        try:
            # Real Snowflake connection parameters
            conn_params = {
                'account': os.getenv('SNOWFLAKE_ACCOUNT', 'demo_account'),
                'user': os.getenv('SNOWFLAKE_USER', 'demo_user'),
                'password': os.getenv('SNOWFLAKE_PASSWORD', 'demo_pass'),
                'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
                'database': os.getenv('SNOWFLAKE_DATABASE', 'DEMO_DB'),
                'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
            }

            # Measure connection time
            start_time = time.time()
            conn = snowflake.connector.connect(**conn_params)
            connection_time = (time.time() - start_time) * 1000  # ms

            # Run test query
            cursor = conn.cursor()
            query_start = time.time()
            cursor.execute("SELECT CURRENT_TIMESTAMP()")
            result = cursor.fetchone()
            query_time = (time.time() - query_start) * 1000  # ms

            cursor.close()
            conn.close()

            return {
                "status": "connected",
                "warehouse": conn_params['warehouse'],
                "database": conn_params['database'],
                "connection_time_ms": round(connection_time, 2),
                "query_time_ms": round(query_time, 2),
                "server_time": str(result[0]) if result else None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Snowflake monitoring error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def monitor_s3(self) -> Dict[str, Any]:
        """Monitor real AWS S3 operations"""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name='us-west-2'
            )

            # Measure list buckets operation
            start_time = time.time()
            response = s3_client.list_buckets()
            latency = (time.time() - start_time) * 1000  # ms

            bucket_count = len(response.get('Buckets', []))

            return {
                "status": "connected",
                "bucket_count": bucket_count,
                "latency_ms": round(latency, 2),
                "region": "us-west-2",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"S3 monitoring error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class RealtimeMetricsCollector:
    """Collect real system and application metrics"""

    def __init__(self):
        self.db_monitor = DatabaseConnectionMonitor()
        self.metrics_history = []
        self.max_history = 100  # Keep last 100 data points

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect real system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        # Process information
        process_count = len(psutil.pids())

        # System load average (Unix-like systems)
        try:
            load_avg = os.getloadavg()
            load_1, load_5, load_15 = load_avg
        except (AttributeError, OSError):
            load_1 = load_5 = load_15 = 0

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "percent": disk.percent
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "load": {
                "1min": round(load_1, 2),
                "5min": round(load_5, 2),
                "15min": round(load_15, 2)
            },
            "processes": process_count,
            "platform": platform.system()
        }

    def collect_api_metrics(self) -> Dict[str, Any]:
        """Monitor real API endpoints"""
        apis = {
            "openai": {
                "url": "https://api.openai.com/v1/models",
                "headers": {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
            },
            "github": {
                "url": "https://api.github.com/rate_limit",
                "headers": {}
            }
        }

        api_metrics = {}
        for name, config in apis.items():
            try:
                start_time = time.time()
                response = requests.get(
                    config["url"],
                    headers=config["headers"],
                    timeout=5
                )
                latency = (time.time() - start_time) * 1000  # ms

                api_metrics[name] = {
                    "status": "healthy" if response.status_code == 200 else "degraded",
                    "status_code": response.status_code,
                    "latency_ms": round(latency, 2),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                api_metrics[name] = {
                    "status": "error",
                    "error": str(e)[:100],
                    "timestamp": datetime.now().isoformat()
                }

        return api_metrics

    def collect_database_metrics(self) -> Dict[str, Any]:
        """Collect real database connection metrics"""
        return {
            "neo4j": self.db_monitor.monitor_neo4j(),
            "snowflake": self.db_monitor.monitor_snowflake(),
            "s3": self.db_monitor.monitor_s3(),
            "timestamp": datetime.now().isoformat()
        }

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all real metrics"""
        metrics = {
            "system": self.collect_system_metrics(),
            "apis": self.collect_api_metrics(),
            "databases": self.collect_database_metrics(),
            "timestamp": datetime.now().isoformat()
        }

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

        return metrics


class WebSocketMonitoringServer:
    """WebSocket server for real-time monitoring dashboard"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.collector = RealtimeMetricsCollector()
        self.update_interval = 2  # seconds

    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send initial data
        initial_data = {
            "type": "initial",
            "data": self.collector.collect_all_metrics(),
            "history": self.collector.metrics_history[-20:]  # Last 20 data points
        }
        await websocket.send(json.dumps(initial_data))

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast_metrics(self):
        """Broadcast real metrics to all connected clients"""
        while True:
            if self.clients:
                # Collect real metrics
                metrics = self.collector.collect_all_metrics()

                # Create update message
                message = json.dumps({
                    "type": "update",
                    "data": metrics
                })

                # Broadcast to all clients
                disconnected = set()
                for client in self.clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)

                # Remove disconnected clients
                for client in disconnected:
                    await self.unregister_client(client)

            await asyncio.sleep(self.update_interval)

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle client connections and messages"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                # Handle client messages (commands, queries, etc.)
                try:
                    data = json.loads(message)
                    command = data.get("command")

                    if command == "get_history":
                        # Send full metrics history
                        response = {
                            "type": "history",
                            "data": self.collector.metrics_history
                        }
                        await websocket.send(json.dumps(response))

                    elif command == "set_interval":
                        # Update collection interval
                        interval = data.get("interval", 2)
                        self.update_interval = max(1, min(interval, 60))  # 1-60 seconds
                        response = {
                            "type": "config",
                            "update_interval": self.update_interval
                        }
                        await websocket.send(json.dumps(response))

                    elif command == "force_refresh":
                        # Force immediate metrics collection
                        metrics = self.collector.collect_all_metrics()
                        response = {
                            "type": "refresh",
                            "data": metrics
                        }
                        await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def start(self):
        """Start the WebSocket monitoring server"""
        logger.info(f"Starting WebSocket monitoring server on {self.host}:{self.port}")

        # Start broadcast task
        broadcast_task = asyncio.create_task(self.broadcast_metrics())

        # Start WebSocket server
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"WebSocket server running at ws://{self.host}:{self.port}")
            logger.info("Collecting and broadcasting REAL metrics...")
            await asyncio.Future()  # Run forever


def main():
    """Run the real-time monitoring WebSocket server"""
    print("🚀 REAL-TIME MONITORING WEBSOCKET SERVER")
    print("=" * 50)
    print("Collecting REAL system metrics - NO SIMULATIONS")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print()

    # Create and start server
    server = WebSocketMonitoringServer(
        host="localhost",
        port=8765
    )

    # Run async event loop
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down monitoring server...")
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()