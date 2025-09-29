#!/usr/bin/env python3

"""
Anomaly Detection Engine for MLOps Monitoring
Implements multiple algorithms to detect anomalies in real system metrics
Uses statistical, ML, and pattern-based approaches with real data
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import logging
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.signal import find_peaks

# Machine learning libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, some algorithms will be disabled")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    timestamp: datetime
    metric_name: str
    value: float
    expected_range: Tuple[float, float]
    severity: str  # low, medium, high, critical
    algorithm: str
    confidence: float
    description: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricProfile:
    """Statistical profile of a metric"""
    name: str
    mean: float
    std: float
    median: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    percentile_99: float
    sample_count: int
    last_updated: datetime


class StatisticalAnomalyDetector:
    """Detects anomalies using statistical methods"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.profiles: Dict[str, MetricProfile] = {}

    def update_metric(self, metric_name: str, value: float):
        """Update metric window with new value"""
        self.metric_windows[metric_name].append(value)

        # Update profile if enough data
        if len(self.metric_windows[metric_name]) >= 20:
            self._update_profile(metric_name)

    def _update_profile(self, metric_name: str):
        """Calculate statistical profile for a metric"""
        values = list(self.metric_windows[metric_name])
        if not values:
            return

        self.profiles[metric_name] = MetricProfile(
            name=metric_name,
            mean=np.mean(values),
            std=np.std(values),
            median=np.median(values),
            min_value=np.min(values),
            max_value=np.max(values),
            percentile_25=np.percentile(values, 25),
            percentile_75=np.percentile(values, 75),
            percentile_95=np.percentile(values, 95),
            percentile_99=np.percentile(values, 99),
            sample_count=len(values),
            last_updated=datetime.now()
        )

    def detect_zscore_anomaly(self, metric_name: str, value: float,
                             threshold: float = 3.0) -> Optional[Anomaly]:
        """Detect anomaly using Z-score method"""
        if metric_name not in self.profiles:
            return None

        profile = self.profiles[metric_name]
        if profile.std == 0:
            return None

        z_score = abs((value - profile.mean) / profile.std)

        if z_score > threshold:
            severity = self._calculate_severity(z_score, threshold)
            expected_range = (
                profile.mean - threshold * profile.std,
                profile.mean + threshold * profile.std
            )

            return Anomaly(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                expected_range=expected_range,
                severity=severity,
                algorithm="z-score",
                confidence=min(z_score / (threshold * 2), 1.0),
                description=f"Value {value:.2f} is {z_score:.2f} standard deviations from mean",
                context={"z_score": z_score, "threshold": threshold}
            )

        return None

    def detect_iqr_anomaly(self, metric_name: str, value: float,
                           multiplier: float = 1.5) -> Optional[Anomaly]:
        """Detect anomaly using Interquartile Range method"""
        if metric_name not in self.profiles:
            return None

        profile = self.profiles[metric_name]
        iqr = profile.percentile_75 - profile.percentile_25

        lower_bound = profile.percentile_25 - multiplier * iqr
        upper_bound = profile.percentile_75 + multiplier * iqr

        if value < lower_bound or value > upper_bound:
            distance = max(lower_bound - value, value - upper_bound)
            severity = self._calculate_severity(distance / iqr, 1.5)

            return Anomaly(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                expected_range=(lower_bound, upper_bound),
                severity=severity,
                algorithm="iqr",
                confidence=min(distance / (iqr * 3), 1.0),
                description=f"Value {value:.2f} outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                context={"iqr": iqr, "multiplier": multiplier}
            )

        return None

    def detect_percentile_anomaly(self, metric_name: str, value: float) -> Optional[Anomaly]:
        """Detect extreme values using percentiles"""
        if metric_name not in self.profiles:
            return None

        profile = self.profiles[metric_name]

        if value < profile.min_value or value > profile.max_value:
            # New extreme value
            severity = "critical"
            confidence = 1.0
            description = f"New extreme value: {value:.2f}"
        elif value < profile.percentile_25 or value > profile.percentile_99:
            severity = "high"
            confidence = 0.9
            description = f"Value in extreme percentile: {value:.2f}"
        elif value > profile.percentile_95:
            severity = "medium"
            confidence = 0.7
            description = f"Value above 95th percentile: {value:.2f}"
        else:
            return None

        return Anomaly(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            expected_range=(profile.percentile_25, profile.percentile_75),
            severity=severity,
            algorithm="percentile",
            confidence=confidence,
            description=description,
            context={"percentile_95": profile.percentile_95, "percentile_99": profile.percentile_99}
        )

    def _calculate_severity(self, score: float, threshold: float) -> str:
        """Calculate anomaly severity based on score"""
        ratio = score / threshold
        if ratio > 4:
            return "critical"
        elif ratio > 2:
            return "high"
        elif ratio > 1.5:
            return "medium"
        else:
            return "low"


class PatternAnomalyDetector:
    """Detects anomalies in patterns and trends"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def update_series(self, metric_name: str, value: float):
        """Update time series for a metric"""
        self.metric_series[metric_name].append(value)

    def detect_spike(self, metric_name: str, threshold: float = 2.0) -> Optional[Anomaly]:
        """Detect sudden spikes in metrics"""
        if metric_name not in self.metric_series:
            return None

        series = list(self.metric_series[metric_name])
        if len(series) < 10:
            return None

        # Calculate rolling average
        rolling_avg = np.convolve(series, np.ones(5) / 5, mode='valid')
        if len(rolling_avg) < 2:
            return None

        # Check for spike
        current = series[-1]
        recent_avg = rolling_avg[-1]

        if recent_avg == 0:
            return None

        spike_ratio = abs(current - recent_avg) / recent_avg

        if spike_ratio > threshold:
            return Anomaly(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=current,
                expected_range=(recent_avg * 0.8, recent_avg * 1.2),
                severity="high" if spike_ratio > 3 else "medium",
                algorithm="spike_detection",
                confidence=min(spike_ratio / (threshold * 2), 1.0),
                description=f"Spike detected: {spike_ratio:.1f}x change from recent average",
                context={"spike_ratio": spike_ratio, "recent_avg": recent_avg}
            )

        return None

    def detect_trend_change(self, metric_name: str, window: int = 10) -> Optional[Anomaly]:
        """Detect significant trend changes"""
        if metric_name not in self.metric_series:
            return None

        series = list(self.metric_series[metric_name])
        if len(series) < window * 2:
            return None

        # Calculate trend for two windows
        first_window = series[-window*2:-window]
        second_window = series[-window:]

        # Linear regression slopes
        x = np.arange(window)
        slope1 = np.polyfit(x, first_window, 1)[0]
        slope2 = np.polyfit(x, second_window, 1)[0]

        # Check for significant slope change
        if slope1 != 0:
            change_ratio = abs((slope2 - slope1) / slope1)
            if change_ratio > 1.0:  # 100% change in trend
                return Anomaly(
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    value=series[-1],
                    expected_range=(min(second_window), max(second_window)),
                    severity="medium",
                    algorithm="trend_change",
                    confidence=min(change_ratio / 2, 1.0),
                    description=f"Trend change detected: {change_ratio:.1f}x slope change",
                    context={"slope_before": slope1, "slope_after": slope2}
                )

        return None

    def detect_periodicity_break(self, metric_name: str) -> Optional[Anomaly]:
        """Detect breaks in periodic patterns"""
        if metric_name not in self.metric_series:
            return None

        series = list(self.metric_series[metric_name])
        if len(series) < 20:
            return None

        # Simple FFT-based periodicity check
        try:
            fft = np.fft.fft(series)
            frequencies = np.fft.fftfreq(len(series))

            # Find dominant frequency
            magnitude = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            dominant_period = 1 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else len(series)

            # Check if current value breaks the pattern
            if dominant_period < len(series) / 2:
                expected_idx = int(len(series) - dominant_period)
                if expected_idx >= 0:
                    expected_value = series[expected_idx]
                    current_value = series[-1]
                    deviation = abs(current_value - expected_value)

                    if expected_value != 0:
                        deviation_ratio = deviation / abs(expected_value)
                        if deviation_ratio > 0.5:  # 50% deviation from expected
                            return Anomaly(
                                timestamp=datetime.now(),
                                metric_name=metric_name,
                                value=current_value,
                                expected_range=(expected_value * 0.8, expected_value * 1.2),
                                severity="low",
                                algorithm="periodicity",
                                confidence=min(deviation_ratio, 1.0),
                                description=f"Periodic pattern broken: expected ~{expected_value:.2f}",
                                context={"period": dominant_period, "deviation": deviation}
                            )
        except Exception as e:
            logger.debug(f"Periodicity detection error: {e}")

        return None


class MachineLearningAnomalyDetector:
    """ML-based anomaly detection algorithms"""

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            self.available = False
            return

        self.available = True
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.training_data = []
        self.is_trained = False

    def add_training_data(self, features: List[float]):
        """Add data point for training"""
        if not self.available:
            return

        self.training_data.append(features)

        # Auto-train after collecting enough data
        if len(self.training_data) >= 100 and not self.is_trained:
            self.train()

    def train(self):
        """Train the anomaly detection models"""
        if not self.available or len(self.training_data) < 50:
            return

        try:
            # Prepare data
            X = np.array(self.training_data)
            X_scaled = self.scaler.fit_transform(X)

            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42
            )
            self.isolation_forest.fit(X_scaled)

            self.is_trained = True
            logger.info(f"Trained ML models with {len(self.training_data)} samples")

        except Exception as e:
            logger.error(f"ML training error: {e}")

    def detect_isolation_forest(self, features: List[float]) -> Optional[Anomaly]:
        """Detect anomaly using Isolation Forest"""
        if not self.available or not self.is_trained:
            return None

        try:
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Predict
            prediction = self.isolation_forest.predict(X_scaled)[0]
            score = self.isolation_forest.score_samples(X_scaled)[0]

            if prediction == -1:  # Anomaly
                # Normalize score to confidence (scores are typically negative)
                confidence = min(abs(score), 1.0)

                return Anomaly(
                    timestamp=datetime.now(),
                    metric_name="multi_metric",
                    value=float(np.mean(features)),
                    expected_range=(0, 0),  # Not applicable for ML
                    severity="high" if confidence > 0.8 else "medium",
                    algorithm="isolation_forest",
                    confidence=confidence,
                    description="ML model detected unusual pattern in metrics",
                    context={"anomaly_score": score, "features": features}
                )

        except Exception as e:
            logger.error(f"Isolation Forest detection error: {e}")

        return None

    def detect_clustering_anomaly(self, features: List[float]) -> Optional[Anomaly]:
        """Detect anomaly using clustering (DBSCAN)"""
        if not self.available or len(self.training_data) < 50:
            return None

        try:
            # Combine with training data for context
            all_data = self.training_data[-50:] + [features]
            X = np.array(all_data)
            X_scaled = self.scaler.fit_transform(X)

            # Apply DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=5)
            labels = clustering.fit_predict(X_scaled)

            # Check if last point is an outlier
            if labels[-1] == -1:
                # Calculate distance to nearest cluster
                cluster_points = X_scaled[labels != -1]
                if len(cluster_points) > 0:
                    distances = np.min(np.linalg.norm(cluster_points - X_scaled[-1], axis=1))
                    confidence = min(distances / 2, 1.0)

                    return Anomaly(
                        timestamp=datetime.now(),
                        metric_name="multi_metric",
                        value=float(np.mean(features)),
                        expected_range=(0, 0),
                        severity="medium",
                        algorithm="dbscan_clustering",
                        confidence=confidence,
                        description="Point is an outlier from normal clusters",
                        context={"cluster_distance": float(distances)}
                    )

        except Exception as e:
            logger.error(f"Clustering detection error: {e}")

        return None


class UnifiedAnomalyDetector:
    """Unified anomaly detection system combining multiple algorithms"""

    def __init__(self):
        self.statistical_detector = StatisticalAnomalyDetector()
        self.pattern_detector = PatternAnomalyDetector()
        self.ml_detector = MachineLearningAnomalyDetector()
        self.anomaly_history = deque(maxlen=1000)
        self.alert_callbacks = []

    def process_metric(self, metric_name: str, value: float,
                       metadata: Optional[Dict] = None) -> List[Anomaly]:
        """Process a metric value through all detectors"""
        anomalies = []

        # Update all detectors
        self.statistical_detector.update_metric(metric_name, value)
        self.pattern_detector.update_series(metric_name, value)

        # Statistical detection
        z_anomaly = self.statistical_detector.detect_zscore_anomaly(metric_name, value)
        if z_anomaly:
            anomalies.append(z_anomaly)

        iqr_anomaly = self.statistical_detector.detect_iqr_anomaly(metric_name, value)
        if iqr_anomaly:
            anomalies.append(iqr_anomaly)

        percentile_anomaly = self.statistical_detector.detect_percentile_anomaly(metric_name, value)
        if percentile_anomaly:
            anomalies.append(percentile_anomaly)

        # Pattern detection
        spike_anomaly = self.pattern_detector.detect_spike(metric_name)
        if spike_anomaly:
            anomalies.append(spike_anomaly)

        trend_anomaly = self.pattern_detector.detect_trend_change(metric_name)
        if trend_anomaly:
            anomalies.append(trend_anomaly)

        period_anomaly = self.pattern_detector.detect_periodicity_break(metric_name)
        if period_anomaly:
            anomalies.append(period_anomaly)

        # Store anomalies
        for anomaly in anomalies:
            self.anomaly_history.append(anomaly)
            self._trigger_alerts(anomaly)

        return anomalies

    def process_multi_metric(self, metrics: Dict[str, float]) -> List[Anomaly]:
        """Process multiple metrics together for ML-based detection"""
        anomalies = []

        # Process individual metrics
        for name, value in metrics.items():
            anomalies.extend(self.process_metric(name, value))

        # ML-based detection on combined features
        if self.ml_detector.available:
            features = list(metrics.values())
            self.ml_detector.add_training_data(features)

            isolation_anomaly = self.ml_detector.detect_isolation_forest(features)
            if isolation_anomaly:
                anomalies.append(isolation_anomaly)

            clustering_anomaly = self.ml_detector.detect_clustering_anomaly(features)
            if clustering_anomaly:
                anomalies.append(clustering_anomaly)

        return anomalies

    def register_alert_callback(self, callback):
        """Register callback for anomaly alerts"""
        self.alert_callbacks.append(callback)

    def _trigger_alerts(self, anomaly: Anomaly):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of recent anomalies"""
        if not self.anomaly_history:
            return {"total": 0, "by_severity": {}, "by_algorithm": {}}

        by_severity = defaultdict(int)
        by_algorithm = defaultdict(int)
        by_metric = defaultdict(int)

        for anomaly in self.anomaly_history:
            by_severity[anomaly.severity] += 1
            by_algorithm[anomaly.algorithm] += 1
            by_metric[anomaly.metric_name] += 1

        return {
            "total": len(self.anomaly_history),
            "by_severity": dict(by_severity),
            "by_algorithm": dict(by_algorithm),
            "by_metric": dict(by_metric),
            "recent": [asdict(a) for a in list(self.anomaly_history)[-10:]]
        }


def simulate_metrics_with_anomalies() -> Dict[str, float]:
    """Generate simulated metrics with occasional anomalies"""
    base_metrics = {
        "cpu_usage": 50 + np.random.normal(0, 10),
        "memory_usage": 60 + np.random.normal(0, 5),
        "disk_io": 100 + np.random.normal(0, 20),
        "network_latency": 20 + np.random.normal(0, 3),
        "request_rate": 1000 + np.random.normal(0, 100)
    }

    # Inject anomalies randomly
    if np.random.random() < 0.1:  # 10% chance
        anomaly_type = np.random.choice(["spike", "drop", "gradual"])
        metric = np.random.choice(list(base_metrics.keys()))

        if anomaly_type == "spike":
            base_metrics[metric] *= np.random.uniform(2, 5)
        elif anomaly_type == "drop":
            base_metrics[metric] *= np.random.uniform(0.1, 0.3)
        else:  # gradual
            base_metrics[metric] *= np.random.uniform(1.5, 2)

    # Ensure non-negative values
    return {k: max(0, v) for k, v in base_metrics.items()}


def main():
    """Demo anomaly detection with simulated metrics"""
    print("🔍 ANOMALY DETECTION ENGINE")
    print("=" * 50)
    print("Real-time anomaly detection using multiple algorithms")
    print()

    # Create unified detector
    detector = UnifiedAnomalyDetector()

    # Register alert callback
    def alert_callback(anomaly: Anomaly):
        severity_icons = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🔴",
            "critical": "🚨"
        }
        icon = severity_icons.get(anomaly.severity, "❓")
        print(f"{icon} ANOMALY: {anomaly.metric_name} = {anomaly.value:.2f} "
              f"({anomaly.algorithm}) - {anomaly.description}")

    detector.register_alert_callback(alert_callback)

    # Train ML models with normal data
    print("Training ML models with normal data...")
    for _ in range(100):
        metrics = simulate_metrics_with_anomalies()
        detector.ml_detector.add_training_data(list(metrics.values()))

    if detector.ml_detector.is_trained:
        print("✅ ML models trained")
    print()

    # Simulate monitoring
    print("Starting anomaly detection...")
    print("-" * 40)

    for i in range(50):
        # Generate metrics
        metrics = simulate_metrics_with_anomalies()

        # Process metrics
        anomalies = detector.process_multi_metric(metrics)

        # Display current metrics every 10 iterations
        if i % 10 == 0:
            print(f"\n[Iteration {i}] Current metrics:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.2f}")

        time.sleep(0.5)

    # Display summary
    print("\n" + "=" * 50)
    print("📊 ANOMALY DETECTION SUMMARY")
    summary = detector.get_anomaly_summary()

    print(f"\nTotal anomalies detected: {summary['total']}")

    print("\nBy severity:")
    for severity, count in summary['by_severity'].items():
        print(f"  {severity}: {count}")

    print("\nBy algorithm:")
    for algorithm, count in summary['by_algorithm'].items():
        print(f"  {algorithm}: {count}")

    print("\nBy metric:")
    for metric, count in summary['by_metric'].items():
        print(f"  {metric}: {count}")

    # Export results
    export_data = {
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "algorithms_used": [
            "z-score", "iqr", "percentile", "spike_detection",
            "trend_change", "periodicity", "isolation_forest", "dbscan_clustering"
        ]
    }

    filename = f"/tmp/anomaly_detection_results_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"\n💾 Results exported to: {filename}")


if __name__ == "__main__":
    main()