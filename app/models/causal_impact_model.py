"""
Causal Impact Analysis Model

Implements structural causal models (SCM) for understanding cause-effect
relationships in email classification. Follows Pearl's do-calculus framework.

Design Patterns:
    - Strategy Pattern: Multiple causal inference strategies
    - Observer Pattern: Track causal relationships over time
    - Template Method: Standard causal analysis workflow
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import json
from datetime import datetime
from collections import defaultdict


class InterventionType(Enum):
    """Types of causal interventions"""
    DO_OPERATION = "do"  # Pearl's do-operator: do(X=x)
    COUNTERFACTUAL = "counterfactual"  # What if X had been different?
    MEDIATION = "mediation"  # Effect through mediator
    BACKDOOR = "backdoor"  # Adjust for confounders


class ImpactSeverity(Enum):
    """Severity levels for causal impacts"""
    CRITICAL = "critical"  # >0.8 impact
    HIGH = "high"  # 0.6-0.8
    MEDIUM = "medium"  # 0.3-0.6
    LOW = "low"  # 0.1-0.3
    NEGLIGIBLE = "negligible"  # <0.1


@dataclass
class CausalEdge:
    """Represents a causal relationship between variables"""
    source: str
    target: str
    strength: float  # -1 to 1 (negative = inhibitory)
    confidence: float  # 0 to 1
    intervention_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_severity(self) -> ImpactSeverity:
        """Categorize impact severity"""
        abs_strength = abs(self.strength)
        if abs_strength >= 0.8:
            return ImpactSeverity.CRITICAL
        elif abs_strength >= 0.6:
            return ImpactSeverity.HIGH
        elif abs_strength >= 0.3:
            return ImpactSeverity.MEDIUM
        elif abs_strength >= 0.1:
            return ImpactSeverity.LOW
        else:
            return ImpactSeverity.NEGLIGIBLE


@dataclass
class InterventionResult:
    """Result of a causal intervention"""
    intervention_type: InterventionType
    target_variable: str
    original_value: Any
    intervened_value: Any
    outcome_change: float
    confidence: float
    affected_variables: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CausalGraph:
    """
    Maintains causal structure between features and outcomes.

    Uses adjacency list representation for efficient traversal.
    Supports Pearl's do-calculus operations.
    """

    def __init__(self):
        self.edges: Dict[str, List[CausalEdge]] = defaultdict(list)
        self.nodes: set = set()
        self._transitive_closure: Optional[Dict[str, set]] = None

    def add_edge(self, edge: CausalEdge) -> None:
        """Add causal edge to graph"""
        self.edges[edge.source].append(edge)
        self.nodes.add(edge.source)
        self.nodes.add(edge.target)
        self._transitive_closure = None  # Invalidate cache

    def get_children(self, node: str) -> List[str]:
        """Get direct causal descendants"""
        return [edge.target for edge in self.edges.get(node, [])]

    def get_parents(self, node: str) -> List[str]:
        """Get direct causal ancestors"""
        parents = []
        for source, edges in self.edges.items():
            if any(edge.target == node for edge in edges):
                parents.append(source)
        return parents

    def get_descendants(self, node: str) -> set:
        """Get all causal descendants (transitive closure)"""
        if self._transitive_closure is None:
            self._compute_transitive_closure()
        return self._transitive_closure.get(node, set())

    def _compute_transitive_closure(self) -> None:
        """Compute transitive closure using DFS"""
        self._transitive_closure = {}
        for node in self.nodes:
            visited = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    stack.extend(self.get_children(current))
            visited.discard(node)  # Don't include self
            self._transitive_closure[node] = visited

    def find_backdoor_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Find minimal backdoor adjustment set.

        A set Z satisfies backdoor criterion if:
        1. Z blocks all backdoor paths from treatment to outcome
        2. Z contains no descendants of treatment
        """
        # Get all parents of treatment (potential confounders)
        confounders = set(self.get_parents(treatment))

        # Remove descendants of treatment
        treatment_descendants = self.get_descendants(treatment)
        confounders -= treatment_descendants

        return list(confounders)

    def get_edge_strength(self, source: str, target: str) -> float:
        """Get causal strength between two variables"""
        for edge in self.edges.get(source, []):
            if edge.target == target:
                return edge.strength
        return 0.0


class CausalImpactAnalyzer:
    """
    Main analyzer for causal impact analysis.

    Implements multiple causal inference strategies and provides
    interpretable results for both technical and non-technical users.
    """

    def __init__(self, model=None):
        self.model = model  # Reference to EmailClassifierModel
        self.causal_graph = CausalGraph()
        self.intervention_history: List[InterventionResult] = []
        self._initialize_default_structure()

    def _initialize_default_structure(self):
        """Initialize default causal structure for email classification"""
        # Define known causal relationships
        default_edges = [
            CausalEdge("raw_email_email_subject", "email_embeddings_average_embedding", 0.9, 0.95),
            CausalEdge("raw_email_email_body", "email_embeddings_average_embedding", 0.9, 0.95),
            CausalEdge("email_embeddings_average_embedding", "prediction", 0.85, 0.9),
            CausalEdge("spam_has_spam_words", "prediction", 0.6, 0.8),
            CausalEdge("word_length_average_word_length", "prediction", 0.4, 0.7),
            CausalEdge("non_text_non_text_char_count", "prediction", 0.3, 0.6),
        ]

        for edge in default_edges:
            self.causal_graph.add_edge(edge)

    def analyze_feature_impact(
        self,
        features: Dict[str, Any],
        prediction: str,
        intervention_type: InterventionType = InterventionType.COUNTERFACTUAL
    ) -> Dict[str, float]:
        """
        Calculate causal impact of each feature on prediction.

        Uses counterfactual reasoning: what would happen if feature X
        had a different value?

        Args:
            features: Current feature values
            prediction: Current prediction
            intervention_type: Type of causal intervention

        Returns:
            Dict mapping feature names to impact scores (0-1)
        """
        if self.model is None:
            return self._heuristic_impact_scores(features)

        impacts = {}
        baseline_scores = self.model.get_topic_scores(features)
        baseline_confidence = baseline_scores[prediction]

        for feature_name, feature_value in features.items():
            # Skip raw text features (too large for intervention)
            if "email_subject" in feature_name or "email_body" in feature_name:
                continue

            # Perform intervention: set feature to neutral value
            modified_features = features.copy()
            neutral_value = self._get_neutral_value(feature_name, feature_value)
            modified_features[feature_name] = neutral_value

            # Measure outcome change
            counterfactual_scores = self.model.get_topic_scores(modified_features)
            counterfactual_confidence = counterfactual_scores[prediction]

            # Impact = change in predicted topic's confidence
            impact = abs(baseline_confidence - counterfactual_confidence)
            impacts[feature_name] = impact

            # Record intervention
            result = InterventionResult(
                intervention_type=intervention_type,
                target_variable=feature_name,
                original_value=feature_value,
                intervened_value=neutral_value,
                outcome_change=impact,
                confidence=0.8,  # Confidence in intervention
                affected_variables=[prediction]
            )
            self.intervention_history.append(result)

        return impacts

    def _heuristic_impact_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Fallback heuristic when model is unavailable"""
        impacts = {}
        for feature_name in features.keys():
            # Use causal graph structure if available
            strength = self.causal_graph.get_edge_strength(feature_name, "prediction")
            impacts[feature_name] = abs(strength) if strength else 0.3
        return impacts

    def _get_neutral_value(self, feature_name: str, current_value: Any) -> Any:
        """Get neutral/reference value for intervention"""
        if isinstance(current_value, (int, float)):
            return 0.0
        elif isinstance(current_value, bool):
            return False
        elif isinstance(current_value, str):
            return ""
        else:
            return None

    def find_minimal_intervention(
        self,
        features: Dict[str, Any],
        target_prediction: str,
        max_features: int = 3
    ) -> List[Tuple[str, Any, float]]:
        """
        Find minimal set of feature changes to achieve target prediction.

        Useful for SMEs to understand: "What's the smallest change needed?"

        Args:
            features: Current features
            target_prediction: Desired outcome
            max_features: Maximum features to modify

        Returns:
            List of (feature_name, new_value, expected_impact)
        """
        if self.model is None:
            return []

        current_pred = self.model.predict(features)
        if current_pred == target_prediction:
            return []  # Already at target

        # Try single feature interventions first
        interventions = []
        for feature_name, feature_value in features.items():
            if "email_subject" in feature_name or "email_body" in feature_name:
                continue

            # Try multiple intervention values
            for delta in [10, 50, 100, 200, -10, -50, -100]:
                if not isinstance(feature_value, (int, float)):
                    continue

                modified = features.copy()
                modified[feature_name] = feature_value + delta

                if self.model.predict(modified) == target_prediction:
                    impact = abs(delta / (feature_value + 1))  # Normalize
                    interventions.append((feature_name, feature_value + delta, impact))
                    break

        # Sort by minimal impact (smallest change)
        interventions.sort(key=lambda x: x[2])
        return interventions[:max_features]

    def explain_prediction(
        self,
        features: Dict[str, Any],
        prediction: str
    ) -> Dict[str, Any]:
        """
        Generate SME-friendly explanation of prediction.

        Returns plain-language explanation with actionable insights.
        """
        impacts = self.analyze_feature_impact(features, prediction)

        # Rank features by impact
        ranked_features = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
        top_features = ranked_features[:5]

        # Categorize by severity
        severity_buckets = defaultdict(list)
        for feature_name, impact in impacts.items():
            edge = CausalEdge(feature_name, prediction, impact, 0.8)
            severity = edge.get_severity()
            severity_buckets[severity.value].append({
                "feature": self._humanize_feature_name(feature_name),
                "impact": round(impact, 3),
                "value": features.get(feature_name)
            })

        return {
            "prediction": prediction,
            "confidence": "High" if max(impacts.values()) > 0.6 else "Medium",
            "top_drivers": [
                {
                    "feature": self._humanize_feature_name(f),
                    "impact_score": round(i, 3),
                    "interpretation": self._interpret_impact(f, i)
                }
                for f, i in top_features
            ],
            "severity_analysis": dict(severity_buckets),
            "recommendation": self._generate_recommendation(impacts, prediction)
        }

    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert technical feature names to readable labels"""
        name_map = {
            "email_embeddings_average_embedding": "Email Content Similarity",
            "spam_has_spam_words": "Spam Indicator",
            "word_length_average_word_length": "Writing Complexity",
            "non_text_non_text_char_count": "Special Characters",
            "raw_email_email_subject": "Subject Line",
            "raw_email_email_body": "Email Body"
        }
        return name_map.get(feature_name, feature_name.replace("_", " ").title())

    def _interpret_impact(self, feature_name: str, impact: float) -> str:
        """Generate plain-language interpretation"""
        if impact >= 0.7:
            return "Critical driver - major influence on classification"
        elif impact >= 0.5:
            return "Strong driver - significant influence"
        elif impact >= 0.3:
            return "Moderate driver - noticeable influence"
        elif impact >= 0.1:
            return "Minor driver - small influence"
        else:
            return "Minimal driver - negligible influence"

    def _generate_recommendation(self, impacts: Dict[str, float], prediction: str) -> str:
        """Generate actionable recommendation for SMEs"""
        max_impact_feature = max(impacts.items(), key=lambda x: x[1])[0]
        human_name = self._humanize_feature_name(max_impact_feature)

        return (
            f"Classification as '{prediction}' is primarily driven by {human_name}. "
            f"To change the outcome, focus on modifying this feature first."
        )

    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of all interventions performed"""
        if not self.intervention_history:
            return {"total_interventions": 0, "average_impact": 0}

        total = len(self.intervention_history)
        avg_impact = sum(i.outcome_change for i in self.intervention_history) / total

        by_type = defaultdict(int)
        for intervention in self.intervention_history:
            by_type[intervention.intervention_type.value] += 1

        return {
            "total_interventions": total,
            "average_impact": round(avg_impact, 3),
            "by_type": dict(by_type),
            "recent_interventions": [
                {
                    "variable": i.target_variable,
                    "impact": round(i.outcome_change, 3),
                    "timestamp": i.timestamp
                }
                for i in self.intervention_history[-10:]
            ]
        }