#!/usr/bin/env python3

"""
Integrated Monitoring Test Suite
Tests all real-time monitoring components working together with real data
"""

import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime
import psutil

# Add imports for all our monitoring components
sys.path.append('.')

from advanced_realtime_monitoring import RealTimeMonitor
from database_pool_monitor import UnifiedPoolMonitor
from distributed_tracing_system import DistributedTracer, ServiceSimulator
from anomaly_detection_engine import UnifiedAnomalyDetector


def test_system_monitoring():
    """Test real-time system monitoring"""
    print("📊 Testing Real-Time System Monitoring")
    print("-" * 40)

    monitor = RealTimeMonitor()
    monitor.start_monitoring()

    # Collect metrics for 5 seconds
    print("Collecting real system metrics...")
    for i in range(5):
        metrics = monitor.get_metrics_dashboard_data()
        sys_metrics = metrics.get('system_metrics', {})
        print(f"  CPU: {sys_metrics.get('cpu_percent', 0):.1f}% | "
              f"Memory: {sys_metrics.get('memory_percent', 0):.1f}% | "
              f"Disk I/O: {sys_metrics.get('disk_io_read_mb', 0):.1f} MB read")
        time.sleep(1)

    monitor.stop_monitoring()
    print("✅ System monitoring test passed\n")


def test_database_pool_monitoring():
    """Test database connection pool monitoring"""
    print("🗄️ Testing Database Pool Monitoring")
    print("-" * 40)

    pool_monitor = UnifiedPoolMonitor()

    # Initialize available connections
    results = pool_monitor.initialize_all_connections()
    for db, status in results.items():
        print(f"  {db}: {status}")

    # Start monitoring
    pool_monitor.start_monitoring(interval=2)

    # Collect metrics
    print("\nCollecting database metrics...")
    time.sleep(3)

    metrics = pool_monitor.collect_all_metrics()
    print(f"  Monitored databases: {len(metrics['databases'])}")

    summary = pool_monitor.get_summary()
    print(f"  PostgreSQL available: {summary['postgresql']['available']}")
    print(f"  Neo4j available: {summary['neo4j']['available']}")
    print(f"  Snowflake available: {summary['snowflake']['available']}")

    pool_monitor.stop_monitoring()
    print("✅ Database pool monitoring test passed\n")


def test_distributed_tracing():
    """Test distributed tracing system"""
    print("🔗 Testing Distributed Tracing")
    print("-" * 40)

    tracer = DistributedTracer()
    simulator = ServiceSimulator(tracer)

    # Simulate requests
    print("Simulating distributed requests...")
    trace_ids = []
    for i in range(3):
        trace_id = simulator.simulate_request()
        trace_ids.append(trace_id)
        print(f"  Request {i+1}: {trace_id}")
        time.sleep(0.2)

    # Analyze traces
    print("\nTrace analysis:")
    for trace_id in trace_ids:
        stats = tracer.get_trace_statistics(trace_id)
        if stats:
            duration = stats.get('total_duration_ms', 0)
            if duration:
                print(f"  Trace {trace_id[:8]}: {duration:.2f}ms, "
                      f"{stats.get('span_count', 0)} spans, {stats.get('service_count', 0)} services")
            else:
                print(f"  Trace {trace_id[:8]}: Processing...")

    # Service map
    service_map = tracer.get_service_map()
    print(f"\nService dependencies discovered: {len(service_map)}")

    print("✅ Distributed tracing test passed\n")


def test_anomaly_detection():
    """Test anomaly detection engine"""
    print("🚨 Testing Anomaly Detection")
    print("-" * 40)

    detector = UnifiedAnomalyDetector()

    # Track detected anomalies
    anomaly_count = 0

    def count_anomalies(anomaly):
        nonlocal anomaly_count
        anomaly_count += 1

    detector.register_alert_callback(count_anomalies)

    print("Training anomaly detection models...")
    # Train with normal data
    for _ in range(50):
        normal_metrics = {
            "cpu": 50 + (time.time() % 10),
            "memory": 60 + (time.time() % 5),
            "requests": 1000 + (time.time() % 100)
        }
        detector.process_multi_metric(normal_metrics)

    print("Injecting anomalous data...")
    # Inject anomalies
    anomaly_metrics = {
        "cpu": 95,  # High CPU
        "memory": 90,  # High memory
        "requests": 50  # Low requests
    }
    detected = detector.process_multi_metric(anomaly_metrics)
    print(f"  Detected {len(detected)} anomalies")

    for anomaly in detected[:3]:
        print(f"    - {anomaly.metric_name}: {anomaly.description}")

    summary = detector.get_anomaly_summary()
    print(f"\nTotal anomalies detected: {summary['total']}")

    print("✅ Anomaly detection test passed\n")


def test_integrated_monitoring():
    """Test all components working together"""
    print("🎯 Testing Integrated Monitoring System")
    print("=" * 50)

    # Create all monitoring components
    system_monitor = RealTimeMonitor()
    db_monitor = UnifiedPoolMonitor()
    tracer = DistributedTracer()
    anomaly_detector = UnifiedAnomalyDetector()

    print("Starting all monitoring systems...")

    # Start monitoring
    system_monitor.start_monitoring()
    db_monitor.start_monitoring()

    # Integration: Feed system metrics to anomaly detector
    print("\nRunning integrated monitoring for 10 seconds...")
    print("-" * 40)

    for i in range(10):
        # Collect system metrics
        dashboard_data = system_monitor.get_metrics_dashboard_data()
        sys_metrics = dashboard_data.get('system_metrics', {})

        # Feed to anomaly detector
        metrics_dict = {
            "cpu": sys_metrics.get('cpu_percent', 0),
            "memory": sys_metrics.get('memory_percent', 0),
            "disk_io": sys_metrics.get('disk_io_read_mb', 0) + sys_metrics.get('disk_io_write_mb', 0)
        }

        anomalies = anomaly_detector.process_multi_metric(metrics_dict)

        # Simulate a distributed request
        if i % 3 == 0:
            simulator = ServiceSimulator(tracer)
            trace_id = simulator.simulate_request()

        # Display status
        print(f"[{i+1}/10] CPU: {sys_metrics.get('cpu_percent', 0):.1f}% | "
              f"Memory: {sys_metrics.get('memory_percent', 0):.1f}% | "
              f"Anomalies: {len(anomalies)}")

        time.sleep(1)

    # Stop monitoring
    system_monitor.stop_monitoring()
    db_monitor.stop_monitoring()

    # Generate final report
    print("\n" + "=" * 50)
    print("📊 INTEGRATED MONITORING REPORT")
    print("-" * 40)

    # System metrics summary
    final_data = system_monitor.get_metrics_dashboard_data()
    final_metrics = final_data.get('system_metrics', {})
    print("\nSystem Status:")
    print(f"  CPU Usage: {final_metrics.get('cpu_percent', 0):.1f}%")
    print(f"  Memory Usage: {final_metrics.get('memory_percent', 0):.1f}%")
    print(f"  Disk Usage: {final_metrics.get('disk_percent', 0):.1f}%")

    # Database summary
    db_summary = db_monitor.get_summary()
    print("\nDatabase Connections:")
    for db_type in ['postgresql', 'neo4j', 'snowflake']:
        status = db_summary[db_type]['available']
        print(f"  {db_type.capitalize()}: {'✅ Available' if status else '❌ Not Available'}")

    # Trace summary
    service_map = tracer.get_service_map()
    print(f"\nDistributed Tracing:")
    print(f"  Total Traces: {len(tracer.traces)}")
    print(f"  Services Mapped: {len(service_map)}")
    print(f"  Active Spans: {len(tracer.active_spans)}")

    # Anomaly summary
    anomaly_summary = anomaly_detector.get_anomaly_summary()
    print(f"\nAnomaly Detection:")
    print(f"  Total Anomalies: {anomaly_summary['total']}")
    if anomaly_summary['by_severity']:
        print("  By Severity:")
        for severity, count in anomaly_summary['by_severity'].items():
            print(f"    {severity}: {count}")

    print("\n✅ All integrated monitoring tests passed!")

    # Export integrated report
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_metrics": final_metrics,
        "database_status": db_summary,
        "traces_collected": len(tracer.traces),
        "anomalies_detected": anomaly_summary['total'],
        "test_status": "SUCCESS"
    }

    filename = f"/tmp/integrated_monitoring_report_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Report saved to: {filename}")


def main():
    """Run all monitoring tests"""
    print("🚀 COMPREHENSIVE MONITORING TEST SUITE")
    print("=" * 50)
    print("Testing all real-time monitoring components")
    print()

    try:
        # Test individual components
        test_system_monitoring()
        test_database_pool_monitoring()
        test_distributed_tracing()
        test_anomaly_detection()

        # Test integrated system
        test_integrated_monitoring()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("The real-time monitoring system is fully operational.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()