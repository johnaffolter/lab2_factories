"""
Metrics Definition Service

Provides SME-friendly interface for defining metrics, thresholds,
binning strategies, and categorization rules.

Design Patterns:
    - Strategy Pattern: Multiple binning/threshold strategies
    - Factory Pattern: Create metrics from configurations
    - Builder Pattern: Fluent metric definition interface
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class BinningStrategy(Enum):
    """Strategies for binning continuous values"""
    EQUAL_WIDTH = "equal_width"  # Equal-sized bins
    EQUAL_FREQUENCY = "equal_frequency"  # Equal number of items per bin
    QUANTILE = "quantile"  # Based on quantiles
    CUSTOM = "custom"  # Custom bin edges
    LOGARITHMIC = "logarithmic"  # Log scale bins
    SEMANTIC = "semantic"  # Business-defined bins


class ThresholdType(Enum):
    """Types of threshold comparisons"""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    BETWEEN = "between"
    EQUALS = "equals"
    IN_LIST = "in_list"
    REGEX_MATCH = "regex_match"


class AggregationMethod(Enum):
    """Methods for aggregating multiple metrics"""
    SUM = "sum"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    MIN = "min"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class MetricDefinition:
    """
    Defines a single metric with SME-friendly configuration.

    Attributes:
        name: Human-readable metric name
        technical_name: Internal feature/column name
        description: What this metric measures
        unit: Unit of measurement (e.g., "emails", "score", "percentage")
        category: Business category (e.g., "engagement", "quality")
        threshold_rules: List of threshold configurations
        binning_config: Optional binning strategy
    """
    name: str
    technical_name: str
    description: str
    unit: str = "count"
    category: str = "general"
    threshold_rules: List[Dict[str, Any]] = field(default_factory=list)
    binning_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "technical_name": self.technical_name,
            "description": self.description,
            "unit": self.unit,
            "category": self.category,
            "threshold_rules": self.threshold_rules,
            "binning_config": self.binning_config,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricDefinition":
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ThresholdRule:
    """
    Defines a threshold rule with SME-friendly labels.

    Example:
        - If impact_score > 0.7, label as "High Impact"
        - If response_time between 0-2 hours, label as "Excellent"
    """
    name: str
    threshold_type: ThresholdType
    value: Union[float, int, str, List]
    label: str
    color: str = "blue"  # For UI visualization
    icon: str = "📊"  # Emoji or icon identifier
    action_required: bool = False
    recommended_action: str = ""

    def evaluate(self, metric_value: Any) -> bool:
        """Evaluate if metric value meets this threshold"""
        if self.threshold_type == ThresholdType.GREATER_THAN:
            return metric_value > self.value
        elif self.threshold_type == ThresholdType.LESS_THAN:
            return metric_value < self.value
        elif self.threshold_type == ThresholdType.BETWEEN:
            return self.value[0] <= metric_value <= self.value[1]
        elif self.threshold_type == ThresholdType.EQUALS:
            return metric_value == self.value
        elif self.threshold_type == ThresholdType.IN_LIST:
            return metric_value in self.value
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "threshold_type": self.threshold_type.value,
            "value": self.value,
            "label": self.label,
            "color": self.color,
            "icon": self.icon,
            "action_required": self.action_required,
            "recommended_action": self.recommended_action
        }


class BinningStrategyExecutor:
    """Executes different binning strategies"""

    @staticmethod
    def equal_width(values: List[float], n_bins: int = 5) -> List[tuple]:
        """Create equal-width bins"""
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        width = (max_val - min_val) / n_bins

        bins = []
        for i in range(n_bins):
            lower = min_val + i * width
            upper = min_val + (i + 1) * width
            bins.append((lower, upper))
        return bins

    @staticmethod
    def quantile(values: List[float], n_quantiles: int = 4) -> List[tuple]:
        """Create quantile-based bins"""
        if not values:
            return []
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        bins = []
        for i in range(n_quantiles):
            lower_idx = int(i * n / n_quantiles)
            upper_idx = int((i + 1) * n / n_quantiles) - 1
            if upper_idx >= n:
                upper_idx = n - 1

            lower = sorted_vals[lower_idx]
            upper = sorted_vals[upper_idx]
            bins.append((lower, upper))
        return bins

    @staticmethod
    def semantic(bin_definitions: List[Dict[str, Any]]) -> List[tuple]:
        """
        Create semantic bins based on business definitions.

        Example:
            [
                {"label": "Low", "range": [0, 0.3]},
                {"label": "Medium", "range": [0.3, 0.7]},
                {"label": "High", "range": [0.7, 1.0]}
            ]
        """
        bins = []
        for bin_def in bin_definitions:
            range_vals = bin_def.get("range", [])
            if len(range_vals) == 2:
                bins.append(tuple(range_vals))
        return bins


class MetricsDefinitionService:
    """
    Main service for managing metric definitions.

    Provides SME-friendly interface for creating, storing, and
    evaluating metrics with thresholds and binning.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.metrics: Dict[str, MetricDefinition] = {}
        self.threshold_rules: Dict[str, List[ThresholdRule]] = {}
        self.binning_executor = BinningStrategyExecutor()

        if config_path:
            self.load_from_config(config_path)
        else:
            self._initialize_default_metrics()

    def _initialize_default_metrics(self):
        """Initialize default metrics for email classification"""

        # Impact Score Metric
        impact_metric = MetricDefinition(
            name="Causal Impact Score",
            technical_name="causal_impact",
            description="Measures how much a feature influences the classification outcome",
            unit="score (0-1)",
            category="causality",
            threshold_rules=[
                {
                    "name": "critical_impact",
                    "type": "greater_than",
                    "value": 0.8,
                    "label": "Critical Impact",
                    "color": "red",
                    "icon": "🔴",
                    "action_required": True,
                    "recommended_action": "Immediate review required - major classification driver"
                },
                {
                    "name": "high_impact",
                    "type": "between",
                    "value": [0.6, 0.8],
                    "label": "High Impact",
                    "color": "orange",
                    "icon": "🟠",
                    "action_required": False,
                    "recommended_action": "Monitor closely - significant influence"
                },
                {
                    "name": "medium_impact",
                    "type": "between",
                    "value": [0.3, 0.6],
                    "label": "Medium Impact",
                    "color": "yellow",
                    "icon": "🟡",
                    "action_required": False,
                    "recommended_action": "Track periodically"
                },
                {
                    "name": "low_impact",
                    "type": "less_than",
                    "value": 0.3,
                    "label": "Low Impact",
                    "color": "green",
                    "icon": "🟢",
                    "action_required": False,
                    "recommended_action": "No action needed"
                }
            ],
            binning_config={
                "strategy": "semantic",
                "bins": [
                    {"label": "Negligible", "range": [0, 0.1]},
                    {"label": "Low", "range": [0.1, 0.3]},
                    {"label": "Medium", "range": [0.3, 0.6]},
                    {"label": "High", "range": [0.6, 0.8]},
                    {"label": "Critical", "range": [0.8, 1.0]}
                ]
            }
        )
        self.add_metric(impact_metric)

        # Classification Confidence Metric
        confidence_metric = MetricDefinition(
            name="Classification Confidence",
            technical_name="prediction_confidence",
            description="Model's confidence in its classification decision",
            unit="percentage",
            category="model_performance",
            threshold_rules=[
                {
                    "name": "high_confidence",
                    "type": "greater_than",
                    "value": 0.8,
                    "label": "High Confidence",
                    "color": "green",
                    "icon": "✅",
                    "action_required": False,
                    "recommended_action": "Proceed with classification"
                },
                {
                    "name": "low_confidence",
                    "type": "less_than",
                    "value": 0.5,
                    "label": "Low Confidence",
                    "color": "red",
                    "icon": "⚠️",
                    "action_required": True,
                    "recommended_action": "Manual review recommended"
                }
            ]
        )
        self.add_metric(confidence_metric)

        # Email Complexity Metric
        complexity_metric = MetricDefinition(
            name="Email Complexity",
            technical_name="word_length_average_word_length",
            description="Average word length - indicator of email complexity",
            unit="characters",
            category="content_analysis",
            binning_config={
                "strategy": "semantic",
                "bins": [
                    {"label": "Simple", "range": [0, 4]},
                    {"label": "Standard", "range": [4, 6]},
                    {"label": "Complex", "range": [6, 100]}
                ]
            }
        )
        self.add_metric(complexity_metric)

    def add_metric(self, metric: MetricDefinition) -> None:
        """Add a metric definition"""
        self.metrics[metric.technical_name] = metric

        # Parse threshold rules
        threshold_rules = []
        for rule_config in metric.threshold_rules:
            rule = ThresholdRule(
                name=rule_config["name"],
                threshold_type=ThresholdType(rule_config["type"]),
                value=rule_config["value"],
                label=rule_config["label"],
                color=rule_config.get("color", "blue"),
                icon=rule_config.get("icon", "📊"),
                action_required=rule_config.get("action_required", False),
                recommended_action=rule_config.get("recommended_action", "")
            )
            threshold_rules.append(rule)

        self.threshold_rules[metric.technical_name] = threshold_rules

    def evaluate_metric(
        self,
        metric_name: str,
        value: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a metric value against defined thresholds.

        Returns SME-friendly result with labels, colors, and recommendations.
        """
        if metric_name not in self.metrics:
            return {"error": f"Unknown metric: {metric_name}"}

        metric = self.metrics[metric_name]
        rules = self.threshold_rules.get(metric_name, [])

        # Find matching threshold
        matched_rule = None
        for rule in rules:
            if rule.evaluate(value):
                matched_rule = rule
                break

        # Get bin label if binning configured
        bin_label = None
        if metric.binning_config:
            bin_label = self._get_bin_label(value, metric.binning_config)

        return {
            "metric_name": metric.name,
            "value": value,
            "unit": metric.unit,
            "category": metric.category,
            "threshold_matched": matched_rule.label if matched_rule else "Unknown",
            "color": matched_rule.color if matched_rule else "gray",
            "icon": matched_rule.icon if matched_rule else "📊",
            "action_required": matched_rule.action_required if matched_rule else False,
            "recommended_action": matched_rule.recommended_action if matched_rule else "",
            "bin_label": bin_label
        }

    def _get_bin_label(self, value: float, binning_config: Dict[str, Any]) -> str:
        """Get bin label for a value"""
        strategy = binning_config.get("strategy")
        bins = binning_config.get("bins", [])

        if strategy == "semantic":
            for bin_def in bins:
                range_vals = bin_def.get("range", [])
                if len(range_vals) == 2 and range_vals[0] <= value <= range_vals[1]:
                    return bin_def.get("label", "Unknown")

        return "Unknown"

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all metric definitions (SME-friendly format)"""
        return [metric.to_dict() for metric in self.metrics.values()]

    def get_metrics_by_category(self, category: str) -> List[MetricDefinition]:
        """Get metrics filtered by category"""
        return [m for m in self.metrics.values() if m.category == category]

    def save_to_config(self, file_path: str) -> None:
        """Save metric definitions to JSON config file"""
        config = {
            "metrics": [m.to_dict() for m in self.metrics.values()],
            "last_updated": datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_from_config(self, file_path: str) -> None:
        """Load metric definitions from JSON config file"""
        with open(file_path, 'r') as f:
            config = json.load(f)

        for metric_data in config.get("metrics", []):
            metric = MetricDefinition.from_dict(metric_data)
            self.add_metric(metric)

    def create_dashboard_summary(
        self,
        metric_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create SME-friendly dashboard summary.

        Args:
            metric_values: Dict of {technical_name: value}

        Returns:
            Dashboard-ready summary with colors, icons, recommendations
        """
        summary = {
            "total_metrics": len(metric_values),
            "by_category": {},
            "action_required": [],
            "all_evaluations": []
        }

        for technical_name, value in metric_values.items():
            evaluation = self.evaluate_metric(technical_name, value)

            if evaluation.get("action_required"):
                summary["action_required"].append(evaluation)

            # Group by category
            category = evaluation.get("category", "general")
            if category not in summary["by_category"]:
                summary["by_category"][category] = []
            summary["by_category"][category].append(evaluation)

            summary["all_evaluations"].append(evaluation)

        return summary