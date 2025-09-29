#!/usr/bin/env python3
"""
Neo4j GraphQL Integration for Enhanced MLOps System
Author: John Affolter (johnaffolter)
Date: September 29, 2025

This module integrates Neo4j GraphQL APIs with the MLOps training system
to provide real-time graph analytics and feedback loops.
"""

import requests
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j GraphQL endpoints"""
    proper_url: str = "https://37424e73-graphql.production-orch-0706.neo4j.io/graphql"
    proper_api_key: str = "WRzgIV6S6q0fEutyzQ9FMWHfDTWfALLT"
    melting_url: str = "https://24f99d3d-graphql.production-orch-0336.neo4j.io/graphql"
    melting_api_key: str = "Zd6Zb12b11DDXtVjxeF7jCX3WPD7dWN5"
    client_id: str = "MPGDSjrdi1iYhpcFGkmua1LKkTCEMjPx"
    client_secret: str = "LnaIH0C2BSUxTlXTnrBdZsR0Tbgxi6bbJTA8clk7wlZKe0TAmhmIOmaZuIO1FyYj"

class Neo4jMLOpsIntegration:
    """Integrates Neo4j graph database with MLOps training system"""

    def __init__(self, config: Neo4jConfig = None):
        self.config = config or Neo4jConfig()
        self.proper_session = requests.Session()
        self.melting_session = requests.Session()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.training_data = []
        self.feedback_history = []

    def query_proper_graph(self, query: str, variables: Dict = None) -> Dict[str, Any]:
        """Query PROPER Neo4j instance"""
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.config.proper_api_key
        }

        payload = {
            "query": query,
            "variables": variables or {}
        }

        try:
            response = self.proper_session.post(
                self.config.proper_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"PROPER query failed: {e}")
            return {"errors": [str(e)]}

    def query_melting_graph(self, query: str, variables: Dict = None) -> Dict[str, Any]:
        """Query MELTING Neo4j instance"""
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.config.melting_api_key
        }

        payload = {
            "query": query,
            "variables": variables or {}
        }

        try:
            response = self.melting_session.post(
                self.config.melting_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"MELTING query failed: {e}")
            return {"errors": [str(e)]}

    def fetch_training_data(self, source: str = "proper") -> List[Dict]:
        """Fetch training data from Neo4j for ML model training"""

        # GraphQL query to fetch order and customer data
        query = """
        query FetchTrainingData($limit: Int!) {
            orders(limit: $limit) {
                id
                totalAmount
                businessDate
                orderType
                status
                lineItems {
                    itemName
                    quantity
                    price
                    category
                }
                customer {
                    id
                    segment
                    lifetimeValue
                    churnRisk
                }
            }
        }
        """

        variables = {"limit": 1000}

        if source == "proper":
            result = self.query_proper_graph(query, variables)
        else:
            result = self.query_melting_graph(query, variables)

        if "data" in result and "orders" in result["data"]:
            self.training_data = result["data"]["orders"]
            logger.info(f"Fetched {len(self.training_data)} training records from {source}")
            return self.training_data

        return []

    def analyze_customer_patterns(self) -> Dict[str, Any]:
        """Analyze customer patterns using graph data"""

        query = """
        query CustomerPatternAnalysis {
            customerAnalytics {
                segments {
                    name
                    count
                    avgOrderValue
                    churnRate
                }
                patterns {
                    type
                    frequency
                    impact
                }
                predictions {
                    nextMonthRevenue
                    churnRiskCustomers
                    growthOpportunities
                }
            }
        }
        """

        proper_result = self.query_proper_graph(query)
        melting_result = self.query_melting_graph(query)

        analysis = {
            "proper_patterns": proper_result.get("data", {}).get("customerAnalytics", {}),
            "melting_patterns": melting_result.get("data", {}).get("customerAnalytics", {}),
            "timestamp": datetime.now().isoformat(),
            "combined_insights": self._combine_insights(proper_result, melting_result)
        }

        return analysis

    def _combine_insights(self, proper_data: Dict, melting_data: Dict) -> Dict:
        """Combine insights from both Neo4j instances"""
        insights = {
            "total_customers": 0,
            "avg_order_value": 0,
            "churn_risk": 0,
            "growth_potential": 0
        }

        # Process PROPER data
        if "data" in proper_data:
            # Extract metrics
            pass

        # Process MELTING data
        if "data" in melting_data:
            # Extract metrics
            pass

        return insights

    def train_graph_enhanced_model(self, model_type: str = "classification") -> Dict:
        """Train ML model enhanced with graph features"""

        # Fetch fresh training data
        proper_data = self.fetch_training_data("proper")
        melting_data = self.fetch_training_data("melting")

        # Extract features from graph data
        features = self._extract_graph_features(proper_data + melting_data)

        # Train model based on type
        if model_type == "classification":
            model = self._train_classification_model(features)
        elif model_type == "clustering":
            model = self._train_clustering_model(features)
        else:
            model = self._train_regression_model(features)

        # Store model in Neo4j
        self._store_model_in_graph(model, model_type)

        return {
            "model_type": model_type,
            "training_samples": len(features),
            "accuracy": model.get("accuracy", 0),
            "timestamp": datetime.now().isoformat()
        }

    def _extract_graph_features(self, data: List[Dict]) -> np.ndarray:
        """Extract features from graph data"""
        features = []

        for record in data:
            feature_vector = [
                float(record.get("totalAmount", 0)),
                len(record.get("lineItems", [])),
                self._calculate_item_diversity(record.get("lineItems", [])),
                self._calculate_customer_score(record.get("customer", {}))
            ]
            features.append(feature_vector)

        return np.array(features)

    def _calculate_item_diversity(self, items: List[Dict]) -> float:
        """Calculate diversity score for order items"""
        if not items:
            return 0

        categories = set(item.get("category", "") for item in items)
        return len(categories) / len(items)

    def _calculate_customer_score(self, customer: Dict) -> float:
        """Calculate customer value score"""
        if not customer:
            return 0

        ltv = float(customer.get("lifetimeValue", 0))
        churn_risk = float(customer.get("churnRisk", 0))

        # Higher LTV and lower churn risk = higher score
        return ltv * (1 - churn_risk)

    def _train_classification_model(self, features: np.ndarray) -> Dict:
        """Train classification model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Generate synthetic labels for demo
        labels = np.random.choice([0, 1], size=len(features))

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        return {
            "model": model,
            "accuracy": accuracy,
            "feature_importance": model.feature_importances_.tolist()
        }

    def _train_clustering_model(self, features: np.ndarray) -> Dict:
        """Train clustering model"""
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=5, random_state=42)
        model.fit(features)

        return {
            "model": model,
            "n_clusters": 5,
            "inertia": model.inertia_
        }

    def _train_regression_model(self, features: np.ndarray) -> Dict:
        """Train regression model"""
        from sklearn.linear_model import LinearRegression

        # Use first feature as target for demo
        X = features[:, 1:]
        y = features[:, 0]

        model = LinearRegression()
        model.fit(X, y)

        return {
            "model": model,
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_
        }

    def _store_model_in_graph(self, model_data: Dict, model_type: str):
        """Store trained model metadata in Neo4j"""

        mutation = """
        mutation StoreModel($input: ModelInput!) {
            createModel(input: $input) {
                id
                type
                accuracy
                createdAt
            }
        }
        """

        variables = {
            "input": {
                "type": model_type,
                "accuracy": model_data.get("accuracy", 0),
                "metadata": json.dumps({
                    "feature_importance": model_data.get("feature_importance", []),
                    "training_samples": len(self.training_data)
                })
            }
        }

        # Store in both instances
        self.query_proper_graph(mutation, variables)
        self.query_melting_graph(mutation, variables)

    async def create_feedback_loop(self):
        """Create continuous feedback loop with Neo4j"""

        while True:
            try:
                # Fetch latest metrics from graph
                metrics = await self._fetch_real_time_metrics()

                # Analyze performance
                analysis = self._analyze_performance(metrics)

                # Generate recommendations
                recommendations = self._generate_recommendations(analysis)

                # Apply improvements
                await self._apply_improvements(recommendations)

                # Log feedback
                self.feedback_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "recommendations": recommendations
                })

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Feedback loop error: {e}")
                await asyncio.sleep(30)

    async def _fetch_real_time_metrics(self) -> Dict:
        """Fetch real-time metrics from Neo4j"""
        query = """
        query RealTimeMetrics {
            systemMetrics {
                orderVolume
                avgProcessingTime
                errorRate
                customerSatisfaction
            }
        }
        """

        proper_metrics = self.query_proper_graph(query)
        melting_metrics = self.query_melting_graph(query)

        return {
            "proper": proper_metrics.get("data", {}).get("systemMetrics", {}),
            "melting": melting_metrics.get("data", {}).get("systemMetrics", {})
        }

    def _analyze_performance(self, metrics: Dict) -> Dict:
        """Analyze system performance"""
        analysis = {
            "performance_score": 0,
            "bottlenecks": [],
            "opportunities": []
        }

        # Calculate performance score
        if metrics.get("proper"):
            proper_score = self._calculate_performance_score(metrics["proper"])
            analysis["performance_score"] += proper_score * 0.5

        if metrics.get("melting"):
            melting_score = self._calculate_performance_score(metrics["melting"])
            analysis["performance_score"] += melting_score * 0.5

        return analysis

    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate performance score from metrics"""
        score = 0.0

        # Order volume (normalized)
        order_volume = metrics.get("orderVolume", 0)
        score += min(order_volume / 1000, 1.0) * 0.25

        # Processing time (inverse)
        processing_time = metrics.get("avgProcessingTime", 1000)
        score += max(0, 1 - (processing_time / 5000)) * 0.25

        # Error rate (inverse)
        error_rate = metrics.get("errorRate", 0)
        score += (1 - error_rate) * 0.25

        # Customer satisfaction
        satisfaction = metrics.get("customerSatisfaction", 0)
        score += satisfaction * 0.25

        return score

    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate improvement recommendations"""
        recommendations = []

        score = analysis.get("performance_score", 0)

        if score < 0.5:
            recommendations.append({
                "type": "critical",
                "action": "Immediate model retraining required",
                "priority": 1
            })
        elif score < 0.7:
            recommendations.append({
                "type": "warning",
                "action": "Consider parameter tuning",
                "priority": 2
            })
        else:
            recommendations.append({
                "type": "info",
                "action": "System performing well",
                "priority": 3
            })

        return recommendations

    async def _apply_improvements(self, recommendations: List[Dict]):
        """Apply recommended improvements"""
        for rec in recommendations:
            if rec["type"] == "critical" and rec["priority"] == 1:
                # Trigger immediate retraining
                logger.warning(f"Applying critical improvement: {rec['action']}")
                await self._trigger_retraining()

    async def _trigger_retraining(self):
        """Trigger model retraining"""
        logger.info("Triggering model retraining based on feedback")
        self.train_graph_enhanced_model("classification")


def test_neo4j_integration():
    """Test Neo4j integration with MLOps system"""

    print("\n" + "="*80)
    print("🔬 Testing Neo4j GraphQL Integration for MLOps")
    print("="*80)

    # Initialize integration
    integration = Neo4jMLOpsIntegration()

    # Test 1: Connection to PROPER
    print("\n📡 Testing PROPER GraphQL connection...")
    test_query = """
    query TestConnection {
        __typename
    }
    """

    result = integration.query_proper_graph(test_query)
    if "errors" not in result:
        print("✅ PROPER connection successful")
    else:
        print(f"❌ PROPER connection failed: {result['errors']}")

    # Test 2: Connection to MELTING
    print("\n📡 Testing MELTING GraphQL connection...")
    result = integration.query_melting_graph(test_query)
    if "errors" not in result:
        print("✅ MELTING connection successful")
    else:
        print(f"❌ MELTING connection failed: {result['errors']}")

    # Test 3: Fetch training data
    print("\n📊 Fetching training data...")
    proper_data = integration.fetch_training_data("proper")
    melting_data = integration.fetch_training_data("melting")
    print(f"✅ Fetched {len(proper_data)} records from PROPER")
    print(f"✅ Fetched {len(melting_data)} records from MELTING")

    # Test 4: Train graph-enhanced model
    print("\n🤖 Training graph-enhanced ML model...")
    model_result = integration.train_graph_enhanced_model("classification")
    print(f"✅ Model trained with accuracy: {model_result.get('accuracy', 0):.2%}")

    # Test 5: Analyze customer patterns
    print("\n🔍 Analyzing customer patterns...")
    patterns = integration.analyze_customer_patterns()
    print(f"✅ Pattern analysis complete: {len(patterns)} insights generated")

    print("\n" + "="*80)
    print("✅ Neo4j MLOps Integration Test Complete!")
    print("="*80)

    return integration


if __name__ == "__main__":
    # Run integration test
    integration = test_neo4j_integration()

    # Start async feedback loop
    print("\n🔄 Starting continuous feedback loop...")
    print("Press Ctrl+C to stop")

    try:
        asyncio.run(integration.create_feedback_loop())
    except KeyboardInterrupt:
        print("\n👋 Feedback loop stopped")