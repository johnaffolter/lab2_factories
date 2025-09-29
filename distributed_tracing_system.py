#!/usr/bin/env python3

"""
Distributed Tracing System
Tracks request flows across multiple services and components with real timing data
Implements OpenTelemetry-style tracing with spans, contexts, and correlation
"""

import os
import sys
import json
import time
import uuid
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
import logging

# Visualization support
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a unit of work in distributed tracing"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)
    status: str = "running"  # running, completed, error
    error_message: Optional[str] = None


@dataclass
class Trace:
    """Represents a complete request trace across services"""
    trace_id: str
    root_span_id: str
    spans: List[Span]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    services_involved: List[str] = field(default_factory=list)
    total_spans: int = 0
    error_count: int = 0


class SpanContext:
    """Context for managing spans within a trace"""

    def __init__(self, trace_id: str, parent_span_id: Optional[str] = None):
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.baggage: Dict[str, str] = {}

    def create_child_context(self, span_id: str) -> 'SpanContext':
        """Create a child context for nested spans"""
        child = SpanContext(self.trace_id, span_id)
        child.baggage = self.baggage.copy()
        return child


class DistributedTracer:
    """Main distributed tracing system"""

    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans = deque(maxlen=10000)
        self.span_listeners: List[Callable] = []
        self._lock = threading.Lock()

    def create_trace(self, operation_name: str, service_name: str) -> SpanContext:
        """Start a new trace with root span"""
        trace_id = self._generate_id()
        span_id = self._generate_id()

        root_span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=None,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.now(),
            tags={"span.kind": "server", "http.method": "POST"}
        )

        with self._lock:
            self.active_spans[span_id] = root_span

            trace = Trace(
                trace_id=trace_id,
                root_span_id=span_id,
                spans=[root_span],
                start_time=root_span.start_time,
                services_involved=[service_name],
                total_spans=1
            )
            self.traces[trace_id] = trace

        logger.info(f"Started trace {trace_id} with root span {span_id}")
        return SpanContext(trace_id, None)

    @contextmanager
    def span(self, context: SpanContext, operation_name: str, service_name: str, **tags):
        """Context manager for creating and managing spans"""
        span = self.start_span(context, operation_name, service_name, **tags)
        try:
            yield span
            self.finish_span(span.span_id, "completed")
        except Exception as e:
            self.finish_span(span.span_id, "error", str(e))
            raise

    def start_span(self, context: SpanContext, operation_name: str,
                   service_name: str, **tags) -> Span:
        """Start a new span within a trace"""
        span_id = self._generate_id()

        span = Span(
            span_id=span_id,
            trace_id=context.trace_id,
            parent_span_id=context.parent_span_id,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.now(),
            tags=tags
        )

        with self._lock:
            self.active_spans[span_id] = span

            if context.trace_id in self.traces:
                trace = self.traces[context.trace_id]
                trace.spans.append(span)
                trace.total_spans += 1
                if service_name not in trace.services_involved:
                    trace.services_involved.append(service_name)

        self._notify_listeners("span_started", span)
        return span

    def finish_span(self, span_id: str, status: str = "completed",
                   error_message: Optional[str] = None):
        """Complete a span and calculate duration"""
        with self._lock:
            if span_id not in self.active_spans:
                return

            span = self.active_spans.pop(span_id)
            span.end_time = datetime.now()
            span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
            span.status = status
            span.error_message = error_message

            self.completed_spans.append(span)

            # Update trace
            if span.trace_id in self.traces:
                trace = self.traces[span.trace_id]
                if status == "error":
                    trace.error_count += 1

                # Check if trace is complete
                if span.span_id == trace.root_span_id:
                    trace.end_time = span.end_time
                    trace.duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000

        self._notify_listeners("span_finished", span)
        logger.debug(f"Finished span {span_id} with status {status} ({span.duration_ms:.2f}ms)")

    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add a tag to an active span"""
        with self._lock:
            if span_id in self.active_spans:
                self.active_spans[span_id].tags[key] = value

    def add_span_log(self, span_id: str, message: str, level: str = "info"):
        """Add a log entry to a span"""
        with self._lock:
            if span_id in self.active_spans:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": level,
                    "message": message
                }
                self.active_spans[span_id].logs.append(log_entry)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a complete trace by ID"""
        return self.traces.get(trace_id)

    def get_trace_timeline(self, trace_id: str) -> List[Dict]:
        """Generate timeline view of a trace"""
        trace = self.get_trace(trace_id)
        if not trace:
            return []

        timeline = []
        for span in trace.spans:
            timeline.append({
                "span_id": span.span_id,
                "operation": span.operation_name,
                "service": span.service_name,
                "start": span.start_time.isoformat() if span.start_time else None,
                "end": span.end_time.isoformat() if span.end_time else None,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "parent": span.parent_span_id
            })

        # Sort by start time
        timeline.sort(key=lambda x: x["start"] if x["start"] else "")
        return timeline

    def get_service_map(self) -> Dict[str, List[str]]:
        """Generate service dependency map from traces"""
        service_map = defaultdict(set)

        for trace in self.traces.values():
            spans_by_id = {s.span_id: s for s in trace.spans}

            for span in trace.spans:
                if span.parent_span_id and span.parent_span_id in spans_by_id:
                    parent = spans_by_id[span.parent_span_id]
                    if parent.service_name != span.service_name:
                        service_map[parent.service_name].add(span.service_name)

        # Convert sets to lists
        return {k: list(v) for k, v in service_map.items()}

    def analyze_critical_path(self, trace_id: str) -> List[Span]:
        """Find the critical path (longest duration path) in a trace"""
        trace = self.get_trace(trace_id)
        if not trace or not NETWORKX_AVAILABLE:
            return []

        # Build directed graph
        G = nx.DiGraph()
        spans_by_id = {s.span_id: s for s in trace.spans}

        for span in trace.spans:
            duration = span.duration_ms or 0
            G.add_node(span.span_id, duration=duration, span=span)

            if span.parent_span_id:
                G.add_edge(span.parent_span_id, span.span_id)

        # Find root node
        root = trace.root_span_id

        # Find longest path from root
        if root not in G:
            return []

        try:
            # Use topological sort to find longest path
            longest_path = nx.dag_longest_path(G, weight='duration')
            return [spans_by_id[node] for node in longest_path if node in spans_by_id]
        except:
            return []

    def get_trace_statistics(self, trace_id: str) -> Dict[str, Any]:
        """Calculate statistics for a trace"""
        trace = self.get_trace(trace_id)
        if not trace:
            return {}

        service_durations = defaultdict(float)
        operation_counts = defaultdict(int)

        for span in trace.spans:
            if span.duration_ms:
                service_durations[span.service_name] += span.duration_ms
                operation_counts[span.operation_name] += 1

        return {
            "trace_id": trace_id,
            "total_duration_ms": trace.duration_ms,
            "span_count": trace.total_spans,
            "service_count": len(trace.services_involved),
            "error_count": trace.error_count,
            "error_rate": trace.error_count / max(trace.total_spans, 1),
            "service_durations": dict(service_durations),
            "operation_counts": dict(operation_counts),
            "avg_span_duration": sum(s.duration_ms or 0 for s in trace.spans) / max(len(trace.spans), 1)
        }

    def register_listener(self, callback: Callable):
        """Register a callback for span events"""
        self.span_listeners.append(callback)

    def _notify_listeners(self, event_type: str, span: Span):
        """Notify registered listeners of span events"""
        for listener in self.span_listeners:
            try:
                listener(event_type, span)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    def _generate_id(self) -> str:
        """Generate unique ID for traces and spans"""
        return str(uuid.uuid4())[:16]


class ServiceSimulator:
    """Simulates distributed service calls for testing"""

    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
        self.services = ["api-gateway", "auth-service", "user-service",
                        "database", "cache", "notification-service"]

    def simulate_request(self) -> str:
        """Simulate a complete distributed request"""
        # Start trace at API gateway
        context = self.tracer.create_trace("POST /api/users", "api-gateway")
        trace_id = context.trace_id

        # API Gateway processing
        with self.tracer.span(context, "request_validation", "api-gateway") as span:
            time.sleep(random.uniform(0.01, 0.03))  # Simulate work
            self.tracer.add_span_tag(span.span_id, "http.status", 200)
            self.tracer.add_span_tag(span.span_id, "user.id", "user123")

        # Auth check
        auth_context = context.create_child_context(span.span_id)
        with self.tracer.span(auth_context, "verify_token", "auth-service") as auth_span:
            time.sleep(random.uniform(0.02, 0.05))
            self.tracer.add_span_tag(auth_span.span_id, "auth.valid", True)

        # User service operations
        user_context = context.create_child_context(span.span_id)
        with self.tracer.span(user_context, "get_user", "user-service") as user_span:
            time.sleep(random.uniform(0.01, 0.02))

            # Database query
            db_context = user_context.create_child_context(user_span.span_id)
            with self.tracer.span(db_context, "SELECT user", "database") as db_span:
                time.sleep(random.uniform(0.03, 0.08))
                self.tracer.add_span_tag(db_span.span_id, "db.rows", 1)

            # Cache check
            cache_context = user_context.create_child_context(user_span.span_id)
            with self.tracer.span(cache_context, "cache_lookup", "cache") as cache_span:
                time.sleep(random.uniform(0.001, 0.005))
                hit = random.choice([True, False])
                self.tracer.add_span_tag(cache_span.span_id, "cache.hit", hit)

        # Notification
        if random.choice([True, False]):
            notif_context = context.create_child_context(span.span_id)
            with self.tracer.span(notif_context, "send_email", "notification-service") as notif_span:
                time.sleep(random.uniform(0.05, 0.15))
                self.tracer.add_span_log(notif_span.span_id, "Email sent to user123")

        # Complete root span
        root_span = self.tracer.active_spans.get(context.trace_id)
        if root_span:
            self.tracer.finish_span(root_span.span_id, "completed")

        return trace_id


def visualize_trace(tracer: DistributedTracer, trace_id: str):
    """Visualize a trace as ASCII art"""
    timeline = tracer.get_trace_timeline(trace_id)
    if not timeline:
        return

    print(f"\n📊 Trace Timeline: {trace_id}")
    print("=" * 80)

    # Find min start time for normalization
    start_times = [datetime.fromisoformat(s["start"]) for s in timeline if s["start"]]
    if not start_times:
        return

    min_time = min(start_times)

    # Create visual timeline
    for item in timeline:
        if not item["start"]:
            continue

        start = datetime.fromisoformat(item["start"])
        offset_ms = (start - min_time).total_seconds() * 1000
        duration = item["duration_ms"] or 0

        # Indentation for hierarchy
        indent = "  " * (len([s for s in timeline if s["span_id"] == item["parent"]]))

        # Status indicator
        status_icon = "✅" if item["status"] == "completed" else "❌"

        # Visual bar
        bar_length = min(int(duration / 10), 50)  # Scale to max 50 chars
        bar = "█" * bar_length

        print(f"{indent}{status_icon} {item['service']:20} | {item['operation']:25} "
              f"| {duration:7.2f}ms {bar}")


def main():
    """Demo distributed tracing with simulated services"""
    print("🔍 DISTRIBUTED TRACING SYSTEM")
    print("=" * 50)
    print("Real-time request tracing across multiple services")
    print()

    # Create tracer and simulator
    tracer = DistributedTracer()
    simulator = ServiceSimulator(tracer)

    # Register event listener
    def trace_listener(event_type: str, span: Span):
        if event_type == "span_finished":
            print(f"  [{span.service_name}] {span.operation_name} - {span.duration_ms:.2f}ms")

    tracer.register_listener(trace_listener)

    # Simulate multiple requests
    print("Simulating distributed requests...")
    print("-" * 40)

    trace_ids = []
    for i in range(3):
        print(f"\nRequest {i+1}:")
        trace_id = simulator.simulate_request()
        trace_ids.append(trace_id)
        time.sleep(0.5)

    print("\n" + "=" * 50)

    # Analyze traces
    for trace_id in trace_ids:
        stats = tracer.get_trace_statistics(trace_id)
        print(f"\n📈 Trace Analysis: {trace_id}")
        print(f"  Total Duration: {stats['total_duration_ms']:.2f}ms")
        print(f"  Services Involved: {stats['service_count']}")
        print(f"  Total Spans: {stats['span_count']}")
        print(f"  Error Rate: {stats['error_rate']:.1%}")

        # Show critical path
        critical_path = tracer.analyze_critical_path(trace_id)
        if critical_path:
            print(f"  Critical Path:")
            for span in critical_path[:3]:  # Show top 3
                print(f"    - {span.service_name}/{span.operation_name}: {span.duration_ms:.2f}ms")

    # Service dependency map
    print("\n🗺️ Service Dependencies:")
    service_map = tracer.get_service_map()
    for service, dependencies in service_map.items():
        print(f"  {service} → {', '.join(dependencies)}")

    # Visualize one trace
    if trace_ids:
        visualize_trace(tracer, trace_ids[0])

    # Export trace data
    export_data = {
        "traces": [],
        "service_map": service_map,
        "timestamp": datetime.now().isoformat()
    }

    for trace_id in trace_ids:
        trace = tracer.get_trace(trace_id)
        if trace:
            export_data["traces"].append({
                "trace_id": trace_id,
                "timeline": tracer.get_trace_timeline(trace_id),
                "statistics": tracer.get_trace_statistics(trace_id)
            })

    filename = f"/tmp/distributed_traces_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"\n💾 Trace data exported to: {filename}")


if __name__ == "__main__":
    main()