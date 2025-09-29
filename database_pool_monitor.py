#!/usr/bin/env python3

"""
Database Connection Pool Monitor
Monitors real database connection pools, query performance, and resource utilization
REAL METRICS FROM ACTUAL DATABASE CONNECTIONS
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import logging

# Database drivers
try:
    import psycopg2
    import psycopg2.pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import snowflake.connector
    from snowflake.connector.pool import ConnectionPool
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for a database connection"""
    database_type: str
    connection_id: str
    created_at: datetime
    last_used: datetime
    queries_executed: int
    total_query_time_ms: float
    avg_query_time_ms: float
    errors: int
    status: str  # 'active', 'idle', 'closed'
    current_query: Optional[str] = None


@dataclass
class PoolMetrics:
    """Metrics for a connection pool"""
    database_type: str
    pool_size: int
    active_connections: int
    idle_connections: int
    waiting_requests: int
    total_connections_created: int
    total_queries: int
    avg_query_time_ms: float
    error_rate: float
    timestamp: datetime


class PostgreSQLPoolMonitor:
    """Monitor PostgreSQL connection pools"""

    def __init__(self):
        if not POSTGRES_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.pools: Dict[str, psycopg2.pool.ThreadedConnectionPool] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        self.query_history = deque(maxlen=1000)

    def create_pool(self, name: str, **kwargs) -> bool:
        """Create a monitored PostgreSQL connection pool"""
        if not self.available:
            return False

        try:
            # Default connection parameters
            params = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': os.getenv('POSTGRES_PORT', 5432),
                'database': os.getenv('POSTGRES_DB', 'postgres'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', 'password'),
                'minconn': kwargs.get('min_connections', 1),
                'maxconn': kwargs.get('max_connections', 10)
            }

            # Create threaded connection pool
            pool = psycopg2.pool.ThreadedConnectionPool(**params)
            self.pools[name] = pool

            logger.info(f"Created PostgreSQL pool '{name}' with {params['maxconn']} max connections")
            return True

        except Exception as e:
            logger.error(f"Failed to create PostgreSQL pool: {e}")
            return False

    def execute_query(self, pool_name: str, query: str) -> Optional[List]:
        """Execute a query using the pool and track metrics"""
        if pool_name not in self.pools:
            return None

        pool = self.pools[pool_name]
        conn = None
        cursor = None
        start_time = time.time()

        try:
            # Get connection from pool
            conn = pool.getconn()
            conn_id = str(id(conn))

            # Track connection metrics
            if conn_id not in self.connection_metrics:
                self.connection_metrics[conn_id] = ConnectionMetrics(
                    database_type="PostgreSQL",
                    connection_id=conn_id,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    queries_executed=0,
                    total_query_time_ms=0,
                    avg_query_time_ms=0,
                    errors=0,
                    status="active"
                )

            metrics = self.connection_metrics[conn_id]
            metrics.current_query = query[:100]  # Truncate for display
            metrics.status = "active"

            # Execute query
            cursor = conn.cursor()
            cursor.execute(query)

            # Fetch results if it's a SELECT query
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
            else:
                conn.commit()
                results = []

            # Update metrics
            query_time = (time.time() - start_time) * 1000  # ms
            metrics.queries_executed += 1
            metrics.total_query_time_ms += query_time
            metrics.avg_query_time_ms = metrics.total_query_time_ms / metrics.queries_executed
            metrics.last_used = datetime.now()
            metrics.current_query = None
            metrics.status = "idle"

            # Record in history
            self.query_history.append({
                'pool': pool_name,
                'query': query[:100],
                'time_ms': query_time,
                'timestamp': datetime.now().isoformat()
            })

            return results

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            if conn_id in self.connection_metrics:
                self.connection_metrics[conn_id].errors += 1
            return None

        finally:
            if cursor:
                cursor.close()
            if conn:
                pool.putconn(conn)

    def get_pool_metrics(self, pool_name: str) -> Optional[PoolMetrics]:
        """Get metrics for a specific pool"""
        if pool_name not in self.pools:
            return None

        pool = self.pools[pool_name]

        # Calculate aggregate metrics
        active = sum(1 for m in self.connection_metrics.values()
                    if m.status == "active")
        total_queries = sum(m.queries_executed for m in self.connection_metrics.values())
        total_errors = sum(m.errors for m in self.connection_metrics.values())
        avg_query_time = sum(m.avg_query_time_ms * m.queries_executed
                           for m in self.connection_metrics.values()) / max(total_queries, 1)

        return PoolMetrics(
            database_type="PostgreSQL",
            pool_size=pool.maxconn,
            active_connections=active,
            idle_connections=pool.maxconn - active,
            waiting_requests=0,  # Would need custom tracking
            total_connections_created=len(self.connection_metrics),
            total_queries=total_queries,
            avg_query_time_ms=avg_query_time,
            error_rate=total_errors / max(total_queries, 1),
            timestamp=datetime.now()
        )


class Neo4jPoolMonitor:
    """Monitor Neo4j connection pools"""

    def __init__(self):
        if not NEO4J_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.drivers: Dict[str, Any] = {}
        self.session_metrics: Dict[str, ConnectionMetrics] = {}
        self.query_history = deque(maxlen=1000)

    def create_driver(self, name: str, **kwargs) -> bool:
        """Create a monitored Neo4j driver"""
        if not self.available:
            return False

        try:
            # Connection parameters
            uri = kwargs.get('uri', os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
            username = kwargs.get('username', os.getenv('NEO4J_USERNAME', 'neo4j'))
            password = kwargs.get('password', os.getenv('NEO4J_PASSWORD', 'password'))

            # Create driver with connection pool
            driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                max_connection_pool_size=kwargs.get('max_pool_size', 50),
                connection_acquisition_timeout=kwargs.get('timeout', 60)
            )

            # Verify connectivity
            driver.verify_connectivity()
            self.drivers[name] = driver

            logger.info(f"Created Neo4j driver '{name}' connected to {uri}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            return False

    def execute_cypher(self, driver_name: str, query: str, parameters: Dict = None) -> Optional[List]:
        """Execute a Cypher query and track metrics"""
        if driver_name not in self.drivers:
            return None

        driver = self.drivers[driver_name]
        start_time = time.time()

        try:
            with driver.session() as session:
                session_id = str(id(session))

                # Track session metrics
                if session_id not in self.session_metrics:
                    self.session_metrics[session_id] = ConnectionMetrics(
                        database_type="Neo4j",
                        connection_id=session_id,
                        created_at=datetime.now(),
                        last_used=datetime.now(),
                        queries_executed=0,
                        total_query_time_ms=0,
                        avg_query_time_ms=0,
                        errors=0,
                        status="active"
                    )

                metrics = self.session_metrics[session_id]
                metrics.current_query = query[:100]
                metrics.status = "active"

                # Execute query
                result = session.run(query, parameters or {})
                records = list(result)

                # Update metrics
                query_time = (time.time() - start_time) * 1000  # ms
                metrics.queries_executed += 1
                metrics.total_query_time_ms += query_time
                metrics.avg_query_time_ms = metrics.total_query_time_ms / metrics.queries_executed
                metrics.last_used = datetime.now()
                metrics.current_query = None
                metrics.status = "idle"

                # Record in history
                self.query_history.append({
                    'driver': driver_name,
                    'query': query[:100],
                    'time_ms': query_time,
                    'record_count': len(records),
                    'timestamp': datetime.now().isoformat()
                })

                return records

        except Exception as e:
            logger.error(f"Cypher execution error: {e}")
            if session_id in self.session_metrics:
                self.session_metrics[session_id].errors += 1
            return None

    def get_driver_metrics(self, driver_name: str) -> Optional[Dict]:
        """Get metrics for a Neo4j driver"""
        if driver_name not in self.drivers:
            return None

        # Aggregate metrics
        active = sum(1 for m in self.session_metrics.values()
                    if m.status == "active")
        total_queries = sum(m.queries_executed for m in self.session_metrics.values())
        avg_query_time = sum(m.avg_query_time_ms * m.queries_executed
                           for m in self.session_metrics.values()) / max(total_queries, 1)

        return {
            "driver": driver_name,
            "active_sessions": active,
            "total_sessions": len(self.session_metrics),
            "total_queries": total_queries,
            "avg_query_time_ms": round(avg_query_time, 2),
            "recent_queries": list(self.query_history)[-10:]
        }


class SnowflakePoolMonitor:
    """Monitor Snowflake connection pools"""

    def __init__(self):
        if not SNOWFLAKE_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.connections: Dict[str, Any] = {}
        self.query_metrics: Dict[str, List] = {}

    def create_connection(self, name: str, **kwargs) -> bool:
        """Create a monitored Snowflake connection"""
        if not self.available:
            return False

        try:
            # Connection parameters
            conn_params = {
                'account': kwargs.get('account', os.getenv('SNOWFLAKE_ACCOUNT')),
                'user': kwargs.get('user', os.getenv('SNOWFLAKE_USER')),
                'password': kwargs.get('password', os.getenv('SNOWFLAKE_PASSWORD')),
                'warehouse': kwargs.get('warehouse', os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')),
                'database': kwargs.get('database', os.getenv('SNOWFLAKE_DATABASE')),
                'schema': kwargs.get('schema', os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC'))
            }

            # Create connection
            conn = snowflake.connector.connect(**conn_params)
            self.connections[name] = conn
            self.query_metrics[name] = []

            logger.info(f"Created Snowflake connection '{name}' to {conn_params['account']}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Snowflake connection: {e}")
            return False

    def execute_query(self, conn_name: str, query: str) -> Optional[List]:
        """Execute a Snowflake query and track metrics"""
        if conn_name not in self.connections:
            return None

        conn = self.connections[conn_name]
        start_time = time.time()

        try:
            cursor = conn.cursor()
            cursor.execute(query)

            # Get query ID for tracking
            query_id = cursor.sfqid

            # Fetch results
            results = cursor.fetchall() if cursor.description else []

            # Get query metrics from Snowflake
            query_time = (time.time() - start_time) * 1000  # ms

            # Store metrics
            metric = {
                'query_id': query_id,
                'query': query[:100],
                'time_ms': query_time,
                'row_count': cursor.rowcount,
                'timestamp': datetime.now().isoformat()
            }
            self.query_metrics[conn_name].append(metric)

            # Keep only last 100 queries
            if len(self.query_metrics[conn_name]) > 100:
                self.query_metrics[conn_name] = self.query_metrics[conn_name][-100:]

            cursor.close()
            return results

        except Exception as e:
            logger.error(f"Snowflake query error: {e}")
            return None

    def get_connection_metrics(self, conn_name: str) -> Optional[Dict]:
        """Get metrics for a Snowflake connection"""
        if conn_name not in self.connections:
            return None

        metrics = self.query_metrics.get(conn_name, [])
        if not metrics:
            return {
                "connection": conn_name,
                "status": "connected",
                "total_queries": 0,
                "avg_query_time_ms": 0,
                "recent_queries": []
            }

        total_time = sum(m['time_ms'] for m in metrics)
        return {
            "connection": conn_name,
            "status": "connected",
            "total_queries": len(metrics),
            "avg_query_time_ms": round(total_time / len(metrics), 2),
            "total_rows_processed": sum(m.get('row_count', 0) for m in metrics),
            "recent_queries": metrics[-10:]
        }


class UnifiedPoolMonitor:
    """Unified monitoring for all database connection pools"""

    def __init__(self):
        self.postgres_monitor = PostgreSQLPoolMonitor()
        self.neo4j_monitor = Neo4jPoolMonitor()
        self.snowflake_monitor = SnowflakePoolMonitor()
        self.monitoring_active = False
        self.monitor_thread = None

    def initialize_all_connections(self):
        """Initialize connections to all available databases"""
        results = {}

        # PostgreSQL
        if self.postgres_monitor.available:
            success = self.postgres_monitor.create_pool("main_pool", max_connections=20)
            results['postgresql'] = "connected" if success else "failed"
        else:
            results['postgresql'] = "not_available"

        # Neo4j
        if self.neo4j_monitor.available:
            success = self.neo4j_monitor.create_driver("main_driver", max_pool_size=50)
            results['neo4j'] = "connected" if success else "failed"
        else:
            results['neo4j'] = "not_available"

        # Snowflake
        if self.snowflake_monitor.available:
            success = self.snowflake_monitor.create_connection("main_connection")
            results['snowflake'] = "connected" if success else "failed"
        else:
            results['snowflake'] = "not_available"

        return results

    def start_monitoring(self, interval: int = 5):
        """Start continuous monitoring in background thread"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started database pool monitoring")

    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.collect_all_metrics()
                self._save_metrics(metrics)

                # Check for issues
                self._check_pool_health(metrics)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            time.sleep(interval)

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all database pools"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'databases': {}
        }

        # PostgreSQL metrics
        if self.postgres_monitor.available:
            for pool_name in self.postgres_monitor.pools:
                pool_metrics = self.postgres_monitor.get_pool_metrics(pool_name)
                if pool_metrics:
                    metrics['databases'][f'postgres_{pool_name}'] = asdict(pool_metrics)

        # Neo4j metrics
        if self.neo4j_monitor.available:
            for driver_name in self.neo4j_monitor.drivers:
                driver_metrics = self.neo4j_monitor.get_driver_metrics(driver_name)
                if driver_metrics:
                    metrics['databases'][f'neo4j_{driver_name}'] = driver_metrics

        # Snowflake metrics
        if self.snowflake_monitor.available:
            for conn_name in self.snowflake_monitor.connections:
                conn_metrics = self.snowflake_monitor.get_connection_metrics(conn_name)
                if conn_metrics:
                    metrics['databases'][f'snowflake_{conn_name}'] = conn_metrics

        return metrics

    def _save_metrics(self, metrics: Dict):
        """Save metrics to file for analysis"""
        filename = f"/tmp/db_pool_metrics_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.debug(f"Saved metrics to {filename}")

    def _check_pool_health(self, metrics: Dict):
        """Check pool health and raise alerts if needed"""
        for db_name, db_metrics in metrics['databases'].items():
            # Check for high error rates
            if isinstance(db_metrics, dict):
                error_rate = db_metrics.get('error_rate', 0)
                if error_rate > 0.1:  # 10% error rate
                    logger.warning(f"High error rate in {db_name}: {error_rate:.2%}")

                # Check for slow queries
                avg_time = db_metrics.get('avg_query_time_ms', 0)
                if avg_time > 1000:  # 1 second
                    logger.warning(f"Slow queries in {db_name}: {avg_time:.0f}ms average")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped database pool monitoring")

    def get_summary(self) -> Dict:
        """Get summary of all database pools"""
        return {
            'timestamp': datetime.now().isoformat(),
            'postgresql': {
                'available': self.postgres_monitor.available,
                'pools': list(self.postgres_monitor.pools.keys()) if self.postgres_monitor.available else [],
                'total_connections': len(self.postgres_monitor.connection_metrics) if self.postgres_monitor.available else 0
            },
            'neo4j': {
                'available': self.neo4j_monitor.available,
                'drivers': list(self.neo4j_monitor.drivers.keys()) if self.neo4j_monitor.available else [],
                'total_sessions': len(self.neo4j_monitor.session_metrics) if self.neo4j_monitor.available else 0
            },
            'snowflake': {
                'available': self.snowflake_monitor.available,
                'connections': list(self.snowflake_monitor.connections.keys()) if self.snowflake_monitor.available else [],
                'total_queries': sum(len(m) for m in self.snowflake_monitor.query_metrics.values()) if self.snowflake_monitor.available else 0
            }
        }


def main():
    """Demo database pool monitoring with real connections"""
    print("🔍 DATABASE CONNECTION POOL MONITOR")
    print("=" * 50)
    print("Monitoring REAL database connections and performance")
    print()

    # Create unified monitor
    monitor = UnifiedPoolMonitor()

    # Initialize connections
    print("Initializing database connections...")
    results = monitor.initialize_all_connections()

    for db, status in results.items():
        icon = "✅" if status == "connected" else "❌" if status == "failed" else "⚠️"
        print(f"{icon} {db.upper()}: {status}")

    print()

    # Start monitoring
    monitor.start_monitoring(interval=5)
    print("Started continuous monitoring (5 second interval)")
    print()

    # Run some test queries if databases are available
    if monitor.postgres_monitor.available and "main_pool" in monitor.postgres_monitor.pools:
        print("Testing PostgreSQL pool...")
        try:
            result = monitor.postgres_monitor.execute_query(
                "main_pool",
                "SELECT version()"
            )
            if result:
                print(f"  PostgreSQL version: {result[0][0][:50]}...")
        except Exception as e:
            print(f"  PostgreSQL test failed: {e}")

    if monitor.neo4j_monitor.available and "main_driver" in monitor.neo4j_monitor.drivers:
        print("Testing Neo4j driver...")
        try:
            result = monitor.neo4j_monitor.execute_cypher(
                "main_driver",
                "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            )
            if result:
                print(f"  Neo4j node count: {result[0]['node_count'] if result else 0}")
        except Exception as e:
            print(f"  Neo4j test failed: {e}")

    if monitor.snowflake_monitor.available and "main_connection" in monitor.snowflake_monitor.connections:
        print("Testing Snowflake connection...")
        try:
            result = monitor.snowflake_monitor.execute_query(
                "main_connection",
                "SELECT CURRENT_TIMESTAMP()"
            )
            if result:
                print(f"  Snowflake timestamp: {result[0][0]}")
        except Exception as e:
            print(f"  Snowflake test failed: {e}")

    print()
    print("Collecting metrics...")
    time.sleep(2)

    # Get and display metrics
    metrics = monitor.collect_all_metrics()
    print("\n📊 Current Pool Metrics:")
    print("-" * 40)

    for db_name, db_metrics in metrics['databases'].items():
        print(f"\n{db_name}:")
        if isinstance(db_metrics, dict):
            for key, value in db_metrics.items():
                if key not in ['timestamp', 'recent_queries', 'query_history']:
                    print(f"  {key}: {value}")

    # Get summary
    summary = monitor.get_summary()
    print("\n📈 Summary:")
    print(json.dumps(summary, indent=2, default=str))

    # Let it run for a bit
    print("\nMonitoring active... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(10)
            current_metrics = monitor.collect_all_metrics()
            active_count = sum(
                1 for db_metrics in current_metrics['databases'].values()
                if isinstance(db_metrics, dict) and db_metrics.get('active_connections', 0) > 0
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Active connections: {active_count}")

    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop_monitoring()
        print("Monitor stopped")


if __name__ == "__main__":
    main()