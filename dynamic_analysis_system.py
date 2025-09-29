#!/usr/bin/env python3
"""
Dynamic Analysis System for Email Classification
Allows composable analysis at multiple pivot points
"""

import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import networkx as nx

# API Configuration
BASE_URL = "http://localhost:8000"

@dataclass
class AnalysisPivot:
    """Represents a pivot point for analysis"""
    name: str
    dimensions: List[str]
    metrics: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregations: List[str] = field(default_factory=list)

@dataclass
class TraversalPath:
    """Represents a path through the analysis space"""
    nodes: List[str]
    edges: List[tuple]
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicEmailAnalyzer:
    """Dynamic, composable analysis system"""

    def __init__(self):
        self.api_url = BASE_URL
        self.analysis_cache = {}
        self.pivot_points = self._initialize_pivot_points()
        self.analysis_graph = nx.DiGraph()
        self._build_analysis_graph()

    def _initialize_pivot_points(self) -> Dict[str, AnalysisPivot]:
        """Define all possible pivot points for analysis"""
        return {
            'feature_space': AnalysisPivot(
                name='Feature Space Analysis',
                dimensions=['generator', 'feature_type', 'value_range'],
                metrics=['importance', 'correlation', 'variance'],
                aggregations=['mean', 'std', 'percentile']
            ),
            'classification_space': AnalysisPivot(
                name='Classification Analysis',
                dimensions=['topic', 'confidence', 'model_type'],
                metrics=['accuracy', 'precision', 'recall', 'f1'],
                aggregations=['weighted_avg', 'macro_avg', 'micro_avg']
            ),
            'temporal_space': AnalysisPivot(
                name='Temporal Analysis',
                dimensions=['hour', 'day', 'week', 'month'],
                metrics=['volume', 'accuracy', 'response_time'],
                aggregations=['sum', 'avg', 'trend']
            ),
            'user_space': AnalysisPivot(
                name='User Behavior Analysis',
                dimensions=['sender_domain', 'recipient_count', 'thread_depth'],
                metrics=['engagement', 'classification_accuracy', 'feedback_rate'],
                aggregations=['cohort', 'segment']
            ),
            'error_space': AnalysisPivot(
                name='Error Analysis',
                dimensions=['misclassification_type', 'confidence_bucket', 'feature_failure'],
                metrics=['error_rate', 'impact_score', 'recovery_time'],
                aggregations=['root_cause', 'pattern']
            )
        }

    def _build_analysis_graph(self):
        """Build a graph of analysis relationships"""
        # Define nodes (analysis points)
        nodes = [
            'email_input', 'feature_extraction', 'classification',
            'confidence_scoring', 'topic_assignment', 'error_detection',
            'feedback_loop', 'model_update', 'performance_metrics'
        ]

        # Define edges (traversal paths)
        edges = [
            ('email_input', 'feature_extraction'),
            ('feature_extraction', 'classification'),
            ('classification', 'confidence_scoring'),
            ('confidence_scoring', 'topic_assignment'),
            ('topic_assignment', 'error_detection'),
            ('error_detection', 'feedback_loop'),
            ('feedback_loop', 'model_update'),
            ('model_update', 'performance_metrics'),
            ('performance_metrics', 'feature_extraction'),  # Cycle for improvement
        ]

        self.analysis_graph.add_nodes_from(nodes)
        self.analysis_graph.add_edges_from(edges)

    def compose_analysis(self, pivots: List[str], operation: str = 'intersect') -> Dict[str, Any]:
        """
        Compose multiple analysis pivots dynamically

        Args:
            pivots: List of pivot point names
            operation: How to combine ('intersect', 'union', 'chain')
        """
        results = {}

        if operation == 'intersect':
            # Find common dimensions across pivots
            common_dims = set(self.pivot_points[pivots[0]].dimensions)
            for pivot in pivots[1:]:
                common_dims &= set(self.pivot_points[pivot].dimensions)

            results['common_dimensions'] = list(common_dims)
            results['analysis'] = self._analyze_intersection(pivots, common_dims)

        elif operation == 'union':
            # Combine all dimensions
            all_dims = set()
            for pivot in pivots:
                all_dims |= set(self.pivot_points[pivot].dimensions)

            results['all_dimensions'] = list(all_dims)
            results['analysis'] = self._analyze_union(pivots, all_dims)

        elif operation == 'chain':
            # Sequential analysis through pivots
            results['chain'] = []
            data = None
            for pivot in pivots:
                data = self._analyze_pivot(pivot, input_data=data)
                results['chain'].append({
                    'pivot': pivot,
                    'output': data
                })

        return results

    def traverse_analysis(self, start_node: str, end_node: str) -> TraversalPath:
        """
        Find and execute analysis path between two points
        """
        try:
            path = nx.shortest_path(self.analysis_graph, start_node, end_node)
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]

            # Execute analysis along path
            path_data = self._execute_traversal(path)

            return TraversalPath(
                nodes=path,
                edges=edges,
                metadata={
                    'length': len(path),
                    'data': path_data,
                    'timestamp': datetime.now().isoformat()
                }
            )
        except nx.NetworkXNoPath:
            return TraversalPath(nodes=[], edges=[], metadata={'error': 'No path found'})

    def _analyze_pivot(self, pivot_name: str, input_data=None) -> Dict[str, Any]:
        """Analyze data at a specific pivot point"""
        pivot = self.pivot_points[pivot_name]

        # Fetch or use provided data
        if input_data is None:
            data = self._fetch_data_for_pivot(pivot_name)
        else:
            data = input_data

        analysis = {
            'pivot': pivot_name,
            'timestamp': datetime.now().isoformat(),
            'dimensions': {},
            'metrics': {}
        }

        # Analyze each dimension
        for dim in pivot.dimensions:
            analysis['dimensions'][dim] = self._analyze_dimension(data, dim)

        # Calculate metrics
        for metric in pivot.metrics:
            analysis['metrics'][metric] = self._calculate_metric(data, metric)

        # Apply aggregations
        if pivot.aggregations:
            analysis['aggregations'] = {}
            for agg in pivot.aggregations:
                analysis['aggregations'][agg] = self._apply_aggregation(data, agg)

        return analysis

    def _fetch_data_for_pivot(self, pivot_name: str) -> Dict[str, Any]:
        """Fetch relevant data for a pivot point"""
        if pivot_name == 'feature_space':
            # Get feature information
            response = requests.get(f"{self.api_url}/features")
            return response.json() if response.ok else {}

        elif pivot_name == 'classification_space':
            # Get classification results
            test_emails = self._generate_test_emails()
            results = []
            for email in test_emails:
                response = requests.post(
                    f"{self.api_url}/emails/classify",
                    json=email
                )
                if response.ok:
                    results.append(response.json())
            return {'classifications': results}

        elif pivot_name == 'temporal_space':
            # Get stored emails with timestamps
            response = requests.get(f"{self.api_url}/emails")
            return response.json() if response.ok else {}

        return {}

    def _analyze_dimension(self, data: Dict, dimension: str) -> Dict[str, Any]:
        """Analyze a specific dimension of the data"""
        if dimension == 'generator':
            # Analyze feature generators
            if 'available_generators' in data:
                return {
                    'count': len(data['available_generators']),
                    'types': [g['name'] for g in data['available_generators']],
                    'feature_counts': {
                        g['name']: len(g['features'])
                        for g in data['available_generators']
                    }
                }

        elif dimension == 'topic':
            # Analyze topics
            if 'classifications' in data:
                topics = [c['predicted_topic'] for c in data['classifications']]
                return {
                    'unique': len(set(topics)),
                    'distribution': pd.Series(topics).value_counts().to_dict()
                }

        elif dimension == 'confidence':
            # Analyze confidence scores
            if 'classifications' in data:
                confidences = []
                for c in data['classifications']:
                    if 'topic_scores' in c:
                        max_score = max(c['topic_scores'].values())
                        confidences.append(max_score)

                if confidences:
                    return {
                        'mean': np.mean(confidences),
                        'std': np.std(confidences),
                        'min': min(confidences),
                        'max': max(confidences),
                        'quartiles': np.percentile(confidences, [25, 50, 75]).tolist()
                    }

        return {}

    def _calculate_metric(self, data: Dict, metric: str) -> float:
        """Calculate a specific metric"""
        if metric == 'accuracy' and 'classifications' in data:
            # Simplified accuracy calculation
            correct = sum(1 for c in data['classifications']
                         if self._is_correct_classification(c))
            total = len(data['classifications'])
            return correct / total if total > 0 else 0

        elif metric == 'variance' and 'available_generators' in data:
            # Feature count variance
            counts = [len(g['features']) for g in data['available_generators']]
            return np.var(counts) if counts else 0

        return 0.0

    def _apply_aggregation(self, data: Dict, aggregation: str) -> Any:
        """Apply an aggregation function"""
        if aggregation == 'mean' and 'classifications' in data:
            scores = []
            for c in data['classifications']:
                if 'topic_scores' in c:
                    scores.extend(c['topic_scores'].values())
            return np.mean(scores) if scores else 0

        return None

    def _is_correct_classification(self, classification: Dict) -> bool:
        """Check if classification is correct (simplified)"""
        # In real system, would compare with ground truth
        return classification.get('predicted_topic') != 'unknown'

    def slice_and_dice(self,
                       data_source: str,
                       dimensions: List[str],
                       filters: Optional[Dict] = None,
                       aggregation: str = 'count') -> pd.DataFrame:
        """
        Slice and dice data across multiple dimensions
        """
        # Fetch data
        data = self._fetch_data_for_pivot(data_source)

        # Convert to DataFrame for easy manipulation
        if 'classifications' in data:
            df = pd.DataFrame(data['classifications'])
        elif 'emails' in data:
            df = pd.DataFrame(data['emails'])
        else:
            df = pd.DataFrame()

        if df.empty:
            return df

        # Apply filters
        if filters:
            for col, value in filters.items():
                if col in df.columns:
                    df = df[df[col] == value]

        # Group by dimensions and aggregate
        if dimensions and all(dim in df.columns for dim in dimensions):
            if aggregation == 'count':
                result = df.groupby(dimensions).size().reset_index(name='count')
            elif aggregation == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                result = df.groupby(dimensions)[numeric_cols].mean().reset_index()
            else:
                result = df.groupby(dimensions).first().reset_index()
        else:
            result = df

        return result

    def _generate_test_emails(self) -> List[Dict[str, str]]:
        """Generate test emails for analysis"""
        return [
            {"subject": "Meeting tomorrow", "body": "Let's discuss the project"},
            {"subject": "50% off sale", "body": "Limited time offer"},
            {"subject": "Your invoice", "body": "Payment due in 30 days"},
            {"subject": "Newsletter", "body": "Weekly updates"},
            {"subject": "Support ticket", "body": "Issue with your account"}
        ]

    def _analyze_intersection(self, pivots: List[str], dimensions: set) -> Dict:
        """Analyze intersection of pivot points"""
        results = {}
        for dim in dimensions:
            dim_results = []
            for pivot in pivots:
                data = self._fetch_data_for_pivot(pivot)
                dim_results.append(self._analyze_dimension(data, dim))
            results[dim] = dim_results
        return results

    def _analyze_union(self, pivots: List[str], dimensions: set) -> Dict:
        """Analyze union of pivot points"""
        results = {}
        for pivot in pivots:
            data = self._fetch_data_for_pivot(pivot)
            for dim in self.pivot_points[pivot].dimensions:
                if dim in dimensions:
                    results[f"{pivot}_{dim}"] = self._analyze_dimension(data, dim)
        return results

    def _execute_traversal(self, path: List[str]) -> List[Dict]:
        """Execute analysis along a path"""
        results = []
        for node in path:
            if node == 'email_input':
                data = self._generate_test_emails()
            elif node == 'feature_extraction':
                data = self._extract_features_for_emails(
                    results[-1] if results else self._generate_test_emails()
                )
            elif node == 'classification':
                data = self._classify_emails(results[-1] if results else None)
            else:
                data = {'node': node, 'placeholder': True}

            results.append(data)
        return results

    def _extract_features_for_emails(self, emails: List[Dict]) -> List[Dict]:
        """Extract features for a list of emails"""
        features = []
        for email in emails:
            response = requests.post(
                f"{self.api_url}/emails/classify",
                json=email
            )
            if response.ok:
                result = response.json()
                features.append({
                    'email': email,
                    'features': result.get('features', {})
                })
        return features

    def _classify_emails(self, email_data: List[Dict]) -> List[Dict]:
        """Classify emails"""
        classifications = []
        emails = email_data if isinstance(email_data[0], dict) and 'subject' in email_data[0] \
                else [d.get('email', {}) for d in email_data]

        for email in emails:
            if email:
                response = requests.post(
                    f"{self.api_url}/emails/classify",
                    json=email
                )
                if response.ok:
                    classifications.append(response.json())
        return classifications

    def visualize_analysis(self, analysis_type: str = 'pivot'):
        """Create interactive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Pivot Point Network
        ax = axes[0, 0]
        pos = nx.spring_layout(self.analysis_graph)
        nx.draw(self.analysis_graph, pos, ax=ax,
                with_labels=True, node_color='lightblue',
                node_size=2000, font_size=8,
                edge_color='gray', arrows=True)
        ax.set_title("Analysis Traversal Graph")

        # 2. Feature Importance
        ax = axes[0, 1]
        response = requests.get(f"{self.api_url}/features")
        if response.ok:
            data = response.json()
            generators = [g['name'] for g in data['available_generators']]
            feature_counts = [len(g['features']) for g in data['available_generators']]
            ax.bar(generators, feature_counts, color='green', alpha=0.7)
            ax.set_xlabel("Generator")
            ax.set_ylabel("Feature Count")
            ax.set_title("Feature Generator Distribution")
            ax.tick_params(axis='x', rotation=45)

        # 3. Classification Confidence Distribution
        ax = axes[1, 0]
        test_results = []
        for email in self._generate_test_emails():
            response = requests.post(f"{self.api_url}/emails/classify", json=email)
            if response.ok:
                result = response.json()
                if 'topic_scores' in result:
                    max_score = max(result['topic_scores'].values())
                    test_results.append(max_score)

        if test_results:
            ax.hist(test_results, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Classification Confidence Distribution")

        # 4. Topic Distribution
        ax = axes[1, 1]
        topics = []
        for email in self._generate_test_emails():
            response = requests.post(f"{self.api_url}/emails/classify", json=email)
            if response.ok:
                topics.append(response.json()['predicted_topic'])

        if topics:
            topic_counts = pd.Series(topics).value_counts()
            ax.pie(topic_counts.values, labels=topic_counts.index,
                  autopct='%1.1f%%', startangle=90)
            ax.set_title("Topic Distribution")

        plt.tight_layout()
        plt.savefig('dynamic_analysis.png', dpi=300, bbox_inches='tight')
        return fig


class InteractiveAnalysisCLI:
    """Interactive CLI for dynamic analysis"""

    def __init__(self):
        self.analyzer = DynamicEmailAnalyzer()

    def run(self):
        """Run interactive analysis session"""
        print("\n" + "="*60)
        print("Dynamic Email Classification Analysis System")
        print("="*60)

        while True:
            print("\n1. Analyze Pivot Point")
            print("2. Compose Multiple Pivots")
            print("3. Traverse Analysis Path")
            print("4. Slice and Dice Data")
            print("5. Visualize Analysis")
            print("6. Run Complete Analysis")
            print("0. Exit")

            choice = input("\nSelect option: ")

            if choice == '1':
                self.analyze_pivot_interactive()
            elif choice == '2':
                self.compose_pivots_interactive()
            elif choice == '3':
                self.traverse_path_interactive()
            elif choice == '4':
                self.slice_dice_interactive()
            elif choice == '5':
                self.visualize_interactive()
            elif choice == '6':
                self.run_complete_analysis()
            elif choice == '0':
                break

    def analyze_pivot_interactive(self):
        """Interactive pivot analysis"""
        print("\nAvailable Pivot Points:")
        for i, pivot in enumerate(self.analyzer.pivot_points.keys(), 1):
            print(f"{i}. {pivot}")

        pivot_names = list(self.analyzer.pivot_points.keys())
        choice = int(input("Select pivot (number): ")) - 1

        if 0 <= choice < len(pivot_names):
            pivot = pivot_names[choice]
            result = self.analyzer._analyze_pivot(pivot)
            print(f"\nAnalysis for {pivot}:")
            print(json.dumps(result, indent=2, default=str))

    def compose_pivots_interactive(self):
        """Interactive pivot composition"""
        pivot_names = list(self.analyzer.pivot_points.keys())
        selected = []

        print("\nSelect pivots to compose (enter 0 to finish):")
        for i, pivot in enumerate(pivot_names, 1):
            print(f"{i}. {pivot}")

        while True:
            choice = int(input("Add pivot: "))
            if choice == 0:
                break
            if 1 <= choice <= len(pivot_names):
                selected.append(pivot_names[choice - 1])

        if len(selected) >= 2:
            operation = input("Operation (intersect/union/chain): ")
            result = self.analyzer.compose_analysis(selected, operation)
            print(f"\nComposed Analysis:")
            print(json.dumps(result, indent=2, default=str))

    def traverse_path_interactive(self):
        """Interactive path traversal"""
        nodes = list(self.analyzer.analysis_graph.nodes())

        print("\nAvailable Nodes:")
        for i, node in enumerate(nodes, 1):
            print(f"{i}. {node}")

        start_idx = int(input("Start node (number): ")) - 1
        end_idx = int(input("End node (number): ")) - 1

        if 0 <= start_idx < len(nodes) and 0 <= end_idx < len(nodes):
            path = self.analyzer.traverse_analysis(nodes[start_idx], nodes[end_idx])
            print(f"\nTraversal Path:")
            print(f"Nodes: {' -> '.join(path.nodes)}")
            print(f"Length: {path.metadata.get('length', 0)}")

    def slice_dice_interactive(self):
        """Interactive slice and dice"""
        dimensions = ['predicted_topic', 'confidence', 'feature_count']

        print("\nAvailable Dimensions:")
        for dim in dimensions:
            print(f"- {dim}")

        selected_dims = input("Enter dimensions (comma-separated): ").split(',')
        selected_dims = [d.strip() for d in selected_dims]

        aggregation = input("Aggregation (count/mean/sum): ")

        result = self.analyzer.slice_and_dice(
            'classification_space',
            selected_dims,
            aggregation=aggregation
        )

        print("\nSlice and Dice Result:")
        print(result.to_string() if not result.empty else "No data")

    def visualize_interactive(self):
        """Create and save visualizations"""
        print("\nGenerating visualizations...")
        fig = self.analyzer.visualize_analysis()
        print("Visualizations saved to 'dynamic_analysis.png'")
        plt.show()

    def run_complete_analysis(self):
        """Run complete analysis workflow"""
        print("\n" + "="*60)
        print("Running Complete Dynamic Analysis")
        print("="*60)

        # 1. Individual Pivot Analysis
        print("\n1. Analyzing Individual Pivots:")
        for pivot in self.analyzer.pivot_points.keys():
            result = self.analyzer._analyze_pivot(pivot)
            print(f"\n  {pivot}:")
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    print(f"    {metric}: {value}")

        # 2. Composed Analysis
        print("\n2. Composed Analysis (Feature + Classification):")
        composed = self.analyzer.compose_analysis(
            ['feature_space', 'classification_space'],
            'intersect'
        )
        print(f"  Common dimensions: {composed.get('common_dimensions', [])}")

        # 3. Path Traversal
        print("\n3. Analysis Path Traversal:")
        path = self.analyzer.traverse_analysis('email_input', 'performance_metrics')
        print(f"  Path: {' -> '.join(path.nodes)}")

        # 4. Visualization
        print("\n4. Generating Visualizations...")
        self.analyzer.visualize_analysis()
        print("  Saved to 'dynamic_analysis.png'")

        print("\n" + "="*60)
        print("Analysis Complete!")


def main():
    """Main entry point"""
    print("Initializing Dynamic Analysis System...")

    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if not response.ok:
            print("Error: API server not responding. Please start the server first.")
            print("Run: uvicorn app.main:app --reload")
            return
    except:
        print("Error: Cannot connect to API server at", BASE_URL)
        print("Please start the server: uvicorn app.main:app --reload")
        return

    # Run interactive CLI
    cli = InteractiveAnalysisCLI()
    cli.run()


if __name__ == "__main__":
    main()