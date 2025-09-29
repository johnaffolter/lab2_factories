"""
MLOps Neo4j Knowledge Graph Integration
Connects all email classification components into a unified graph:
- Emails and their classifications
- ML Models and their predictions
- Feature Generators and extracted features
- Training experiments and results
- Design patterns (Factory, Strategy, Dataclass)
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json
import hashlib

load_dotenv()


class MLOpsKnowledgeGraph:
    """Unified knowledge graph for MLOps email classification system"""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = None
        if self.uri and self.username and self.password:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self._initialize_mlops_schema()

    def _initialize_mlops_schema(self):
        """Create comprehensive schema for MLOps knowledge graph"""
        with self.driver.session(database=self.database) as session:
            constraints = [
                # Core entities
                "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (m:MLModel) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT generator_name IF NOT EXISTS FOR (g:FeatureGenerator) REQUIRE g.name IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT experiment_id IF NOT EXISTS FOR (ex:Experiment) REQUIRE ex.id IS UNIQUE",
                "CREATE CONSTRAINT pattern_name IF NOT EXISTS FOR (p:DesignPattern) REQUIRE p.name IS UNIQUE",
            ]

            indexes = [
                "CREATE INDEX email_timestamp IF NOT EXISTS FOR (e:Email) ON (e.timestamp)",
                "CREATE INDEX model_type IF NOT EXISTS FOR (m:MLModel) ON (m.model_type)",
                "CREATE INDEX experiment_timestamp IF NOT EXISTS FOR (ex:Experiment) ON (ex.timestamp)",
                "CREATE INDEX prediction_confidence IF NOT EXISTS FOR ()-[r:PREDICTED_AS]->() ON (r.confidence)",
            ]

            for query in constraints + indexes:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"Schema setup note: {str(e)}")

    def store_email_with_classification(self, email_data: Dict, classification_result: Dict) -> str:
        """
        Store email with its classification results and create full knowledge graph

        Args:
            email_data: {subject, body, sender, timestamp}
            classification_result: {predicted_topic, confidence_scores, features, model_type}

        Returns:
            email_id
        """
        if not self.driver:
            return None

        with self.driver.session(database=self.database) as session:
            email_id = self._generate_id(f"{email_data.get('subject')}_{email_data.get('body')}")

            # 1. Create Email node
            email_query = """
            MERGE (e:Email {id: $email_id})
            SET e.subject = $subject,
                e.body = $body,
                e.sender = $sender,
                e.timestamp = datetime($timestamp),
                e.word_count = $word_count,
                e.char_count = $char_count
            RETURN e.id as email_id
            """

            session.run(
                email_query,
                email_id=email_id,
                subject=email_data.get("subject", ""),
                body=email_data.get("body", ""),
                sender=email_data.get("sender", "unknown"),
                timestamp=email_data.get("timestamp", datetime.now().isoformat()),
                word_count=len(email_data.get("body", "").split()),
                char_count=len(email_data.get("body", ""))
            )

            # 2. Create Topic node and classification relationship
            predicted_topic = classification_result.get("predicted_topic") or classification_result.get("predicted_label")
            if not predicted_topic:
                raise ValueError("Classification result must contain 'predicted_topic' or 'predicted_label'")

            confidence_scores = classification_result.get("confidence_scores", classification_result.get("all_scores", {}))
            confidence = confidence_scores.get(predicted_topic, classification_result.get("confidence", 0))

            topic_query = """
            MATCH (e:Email {id: $email_id})
            MERGE (t:Topic {name: $topic_name})
            ON CREATE SET t.created_at = datetime()
            MERGE (e)-[r:CLASSIFIED_AS]->(t)
            SET r.confidence = $confidence,
                r.timestamp = datetime(),
                r.all_scores = $all_scores
            """

            session.run(
                topic_query,
                email_id=email_id,
                topic_name=predicted_topic,
                confidence=float(confidence) if confidence else 0.0,
                all_scores=json.dumps(confidence_scores)
            )

            # 3. Create ML Model node and prediction relationship
            model_type = classification_result.get("model_type", "EmailClassifierModel")
            model_id = f"model_{model_type}_{datetime.now().strftime('%Y%m%d')}"

            model_query = """
            MATCH (e:Email {id: $email_id})
            MERGE (m:MLModel {id: $model_id})
            ON CREATE SET
                m.model_type = $model_type,
                m.created_at = datetime(),
                m.version = $version
            MERGE (m)-[r:PREDICTED]->(e)
            SET r.confidence = $confidence,
                r.timestamp = datetime(),
                r.processing_time_ms = $processing_time
            """

            session.run(
                model_query,
                email_id=email_id,
                model_id=model_id,
                model_type=model_type,
                version="1.0.0",
                confidence=confidence,
                processing_time=classification_result.get("processing_time_ms", 0)
            )

            # 4. Create Feature nodes and extraction relationships
            features = classification_result.get("features", {})
            self._store_features(session, email_id, features)

            return email_id

    def _store_features(self, session, email_id: str, features: Dict):
        """Store extracted features and link to feature generators"""

        # Group features by generator type
        feature_groups = self._group_features_by_generator(features)

        for generator_name, feature_dict in feature_groups.items():
            # Create FeatureGenerator node
            generator_query = """
            MATCH (e:Email {id: $email_id})
            MERGE (g:FeatureGenerator {name: $generator_name})
            ON CREATE SET
                g.category = $category,
                g.version = $version
            MERGE (g)-[r:EXTRACTED_FROM]->(e)
            SET r.features = $features,
                r.timestamp = datetime()
            """

            session.run(
                generator_query,
                email_id=email_id,
                generator_name=generator_name,
                category=self._get_generator_category(generator_name),
                version="1.0.0",
                features=json.dumps(feature_dict)
            )

    def _group_features_by_generator(self, features: Dict) -> Dict[str, Dict]:
        """Group features by their generator name (prefix)"""
        groups = {}

        for feature_name, feature_value in features.items():
            # Feature naming convention: generator_name_feature_name
            if "_" in feature_name:
                generator_name = feature_name.split("_")[0]
                if generator_name not in groups:
                    groups[generator_name] = {}
                groups[generator_name][feature_name] = feature_value

        return groups

    def _get_generator_category(self, generator_name: str) -> str:
        """Get category for feature generator"""
        categories = {
            "spam": "content_analysis",
            "word": "linguistic_analysis",
            "email": "ml_features",
            "raw": "raw_features",
            "non": "content_analysis"
        }
        for key, category in categories.items():
            if key in generator_name.lower():
                return category
        return "general"

    def store_training_experiment(self, experiment_data: Dict) -> str:
        """
        Store ML training experiment in knowledge graph

        Args:
            experiment_data: {
                experiment_id, models, dataset_size, accuracy,
                s3_path, timestamp, hyperparameters
            }
        """
        if not self.driver:
            return None

        with self.driver.session(database=self.database) as session:
            experiment_id = experiment_data.get("experiment_id", f"exp_{datetime.now().timestamp()}")

            query = """
            MERGE (ex:Experiment {id: $experiment_id})
            SET ex.timestamp = datetime($timestamp),
                ex.dataset_size = $dataset_size,
                ex.accuracy = $accuracy,
                ex.s3_path = $s3_path,
                ex.hyperparameters = $hyperparameters,
                ex.status = $status
            """

            session.run(
                query,
                experiment_id=experiment_id,
                timestamp=experiment_data.get("timestamp", datetime.now().isoformat()),
                dataset_size=experiment_data.get("dataset_size", 0),
                accuracy=experiment_data.get("accuracy", 0.0),
                s3_path=experiment_data.get("s3_path", ""),
                hyperparameters=json.dumps(experiment_data.get("hyperparameters", {})),
                status="completed"
            )

            # Link experiment to models trained
            for model_name in experiment_data.get("models", []):
                model_link_query = """
                MATCH (ex:Experiment {id: $experiment_id})
                MERGE (m:MLModel {id: $model_id})
                ON CREATE SET m.model_type = $model_name
                MERGE (ex)-[:TRAINED]->(m)
                """
                session.run(
                    model_link_query,
                    experiment_id=experiment_id,
                    model_id=f"model_{model_name}_{experiment_id}",
                    model_name=model_name
                )

            return experiment_id

    def store_design_patterns(self):
        """Store design patterns used in the system"""
        if not self.driver:
            return

        with self.driver.session(database=self.database) as session:
            patterns = [
                {
                    "name": "Factory Pattern",
                    "component": "FeatureGeneratorFactory",
                    "purpose": "Dynamic feature generator creation",
                    "benefits": ["Decoupling", "Extensibility", "Single Responsibility"]
                },
                {
                    "name": "Strategy Pattern",
                    "component": "BaseFeatureGenerator",
                    "purpose": "Interchangeable feature extraction algorithms",
                    "benefits": ["Flexibility", "Open/Closed Principle", "Runtime Selection"]
                },
                {
                    "name": "Dataclass Pattern",
                    "component": "Email",
                    "purpose": "Structured email data representation",
                    "benefits": ["Type Safety", "Immutability", "Validation"]
                },
                {
                    "name": "Registry Pattern",
                    "component": "GENERATORS Registry",
                    "purpose": "Centralized generator registration",
                    "benefits": ["Discoverability", "Maintainability", "Configuration"]
                }
            ]

            for pattern in patterns:
                query = """
                MERGE (p:DesignPattern {name: $name})
                SET p.component = $component,
                    p.purpose = $purpose,
                    p.benefits = $benefits,
                    p.updated_at = datetime()
                """

                session.run(
                    query,
                    name=pattern["name"],
                    component=pattern["component"],
                    purpose=pattern["purpose"],
                    benefits=json.dumps(pattern["benefits"])
                )

            # Connect patterns to components
            self._connect_patterns_to_components(session)

    def _connect_patterns_to_components(self, session):
        """Create relationships between design patterns and system components"""
        connections = [
            ("Factory Pattern", "FeatureGenerator", "IMPLEMENTS"),
            ("Strategy Pattern", "FeatureGenerator", "IMPLEMENTS"),
            ("Factory Pattern", "MLModel", "CREATES"),
        ]

        for pattern_name, component_type, relationship in connections:
            query = """
            MATCH (p:DesignPattern {name: $pattern_name})
            MATCH (g:FeatureGenerator)
            WHERE g.name IS NOT NULL
            MERGE (p)-[r:USED_BY]->(g)
            """
            try:
                session.run(query, pattern_name=pattern_name)
            except:
                pass

    def get_mlops_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data for MLOps system"""
        if not self.driver:
            return {}

        with self.driver.session(database=self.database) as session:
            # Overall statistics
            stats_query = """
            MATCH (e:Email)
            WITH count(e) as total_emails
            MATCH (m:MLModel)
            WITH total_emails, count(m) as total_models
            MATCH (t:Topic)
            WITH total_emails, total_models, count(t) as total_topics
            MATCH (g:FeatureGenerator)
            WITH total_emails, total_models, total_topics, count(g) as total_generators
            MATCH (ex:Experiment)
            RETURN total_emails, total_models, total_topics, total_generators, count(ex) as total_experiments
            """

            stats = session.run(stats_query).single()

            # Topic distribution
            topic_query = """
            MATCH (t:Topic)<-[:CLASSIFIED_AS]-(e:Email)
            RETURN t.name as topic, count(e) as count
            ORDER BY count DESC
            """
            topics = session.run(topic_query).data()

            # Model performance
            model_query = """
            MATCH (m:MLModel)-[r:PREDICTED]->(e:Email)
            RETURN m.model_type as model,
                   count(e) as predictions,
                   avg(r.confidence) as avg_confidence
            ORDER BY predictions DESC
            """
            models = session.run(model_query).data()

            # Recent classifications
            recent_query = """
            MATCH (e:Email)-[r:CLASSIFIED_AS]->(t:Topic)
            RETURN e.subject as subject,
                   t.name as topic,
                   r.confidence as confidence,
                   e.timestamp as timestamp
            ORDER BY e.timestamp DESC
            LIMIT 10
            """
            recent = session.run(recent_query).data()

            return {
                "statistics": {
                    "total_emails": stats["total_emails"] if stats else 0,
                    "total_models": stats["total_models"] if stats else 0,
                    "total_topics": stats["total_topics"] if stats else 0,
                    "total_generators": stats["total_generators"] if stats else 0,
                    "total_experiments": stats["total_experiments"] if stats else 0
                },
                "topic_distribution": topics,
                "model_performance": models,
                "recent_classifications": recent
            }

    def get_email_knowledge_graph(self, email_id: str) -> Dict:
        """Get complete knowledge graph for a specific email"""
        if not self.driver:
            return {}

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Email {id: $email_id})
            OPTIONAL MATCH (e)-[r1:CLASSIFIED_AS]->(t:Topic)
            OPTIONAL MATCH (m:MLModel)-[r2:PREDICTED]->(e)
            OPTIONAL MATCH (g:FeatureGenerator)-[r3:EXTRACTED_FROM]->(e)
            RETURN e,
                   collect(DISTINCT {topic: t.name, confidence: r1.confidence}) as topics,
                   collect(DISTINCT {model: m.model_type, confidence: r2.confidence}) as models,
                   collect(DISTINCT {generator: g.name, features: r3.features}) as generators
            """

            result = session.run(query, email_id=email_id).single()

            if result:
                return {
                    "email": dict(result["e"]),
                    "topics": result["topics"],
                    "models": result["models"],
                    "generators": result["generators"]
                }

            return {}

    def query_similar_emails(self, email_id: str, limit: int = 5) -> List[Dict]:
        """Find similar emails based on classification and features"""
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            # Find emails classified to same topics
            query = """
            MATCH (e1:Email {id: $email_id})-[:CLASSIFIED_AS]->(t:Topic)
            MATCH (e2:Email)-[r:CLASSIFIED_AS]->(t)
            WHERE e1.id <> e2.id
            RETURN DISTINCT e2.id as email_id,
                   e2.subject as subject,
                   t.name as topic,
                   r.confidence as confidence
            ORDER BY r.confidence DESC
            LIMIT $limit
            """

            results = session.run(query, email_id=email_id, limit=limit).data()
            return results

    def store_ground_truth(self, email_id: str, true_label: str, annotator: str = "human") -> bool:
        """
        Store ground truth label for an email

        Args:
            email_id: Email identifier
            true_label: True classification label
            annotator: Who provided the label (human, system, etc.)
        """
        if not self.driver:
            return False

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Email {id: $email_id})
            MERGE (t:Topic {name: $true_label})
            MERGE (e)-[r:HAS_GROUND_TRUTH]->(t)
            SET r.annotator = $annotator,
                r.timestamp = datetime(),
                r.verified = true
            RETURN e.id as email_id
            """

            result = session.run(
                query,
                email_id=email_id,
                true_label=true_label,
                annotator=annotator
            )

            return result.single() is not None

    def generate_training_examples(self, count: int = 100) -> List[Dict]:
        """
        Generate training examples from emails with ground truth

        Returns:
            List of {email_id, subject, body, true_label, features}
        """
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e:Email)-[r:HAS_GROUND_TRUTH]->(t:Topic)
            OPTIONAL MATCH (g:FeatureGenerator)-[rf:EXTRACTED_FROM]->(e)
            RETURN e.id as email_id,
                   e.subject as subject,
                   e.body as body,
                   t.name as true_label,
                   collect({generator: g.name, features: rf.features}) as features
            LIMIT $count
            """

            results = session.run(query, count=count).data()

            training_examples = []
            for record in results:
                # Parse features JSON
                all_features = {}
                for feat_group in record["features"]:
                    if feat_group["features"]:
                        features_dict = json.loads(feat_group["features"])
                        all_features.update(features_dict)

                training_examples.append({
                    "email_id": record["email_id"],
                    "subject": record["subject"],
                    "body": record["body"],
                    "true_label": record["true_label"],
                    "features": all_features
                })

            return training_examples

    def store_model_version(self, model_data: Dict) -> str:
        """
        Store a versioned ML model in the knowledge graph

        Args:
            model_data: {
                model_id, model_type, version, accuracy, f1_score,
                training_date, hyperparameters, s3_path, training_size
            }
        """
        if not self.driver:
            return None

        with self.driver.session(database=self.database) as session:
            model_id = model_data.get("model_id", f"model_{datetime.now().timestamp()}")

            query = """
            MERGE (m:MLModel {id: $model_id})
            SET m.model_type = $model_type,
                m.version = $version,
                m.accuracy = $accuracy,
                m.f1_score = $f1_score,
                m.training_date = datetime($training_date),
                m.hyperparameters = $hyperparameters,
                m.s3_path = $s3_path,
                m.training_size = $training_size,
                m.status = $status
            """

            session.run(
                query,
                model_id=model_id,
                model_type=model_data.get("model_type", "EmailClassifier"),
                version=model_data.get("version", "1.0.0"),
                accuracy=model_data.get("accuracy", 0.0),
                f1_score=model_data.get("f1_score", 0.0),
                training_date=model_data.get("training_date", datetime.now().isoformat()),
                hyperparameters=json.dumps(model_data.get("hyperparameters", {})),
                s3_path=model_data.get("s3_path", ""),
                training_size=model_data.get("training_size", 0),
                status="active"
            )

            return model_id

    def store_training_flow(self, flow_data: Dict) -> str:
        """
        Store complete training flow/pipeline in knowledge graph

        Args:
            flow_data: {
                flow_id, name, steps, input_data, output_models,
                execution_time, status, airflow_dag_id, s3_artifacts
            }
        """
        if not self.driver:
            return None

        with self.driver.session(database=self.database) as session:
            flow_id = flow_data.get("flow_id", f"flow_{datetime.now().timestamp()}")

            # Create training flow node
            flow_query = """
            MERGE (f:TrainingFlow {id: $flow_id})
            SET f.name = $name,
                f.steps = $steps,
                f.execution_time_sec = $execution_time,
                f.status = $status,
                f.airflow_dag_id = $airflow_dag_id,
                f.s3_artifacts = $s3_artifacts,
                f.created_at = datetime($created_at)
            """

            session.run(
                flow_query,
                flow_id=flow_id,
                name=flow_data.get("name", "MLOps Training Pipeline"),
                steps=json.dumps(flow_data.get("steps", [])),
                execution_time=flow_data.get("execution_time", 0),
                status=flow_data.get("status", "completed"),
                airflow_dag_id=flow_data.get("airflow_dag_id", ""),
                s3_artifacts=json.dumps(flow_data.get("s3_artifacts", {})),
                created_at=datetime.now().isoformat()
            )

            # Link flow to input dataset
            if flow_data.get("input_data"):
                input_query = """
                MATCH (f:TrainingFlow {id: $flow_id})
                MERGE (d:Dataset {id: $dataset_id})
                SET d.size = $size,
                    d.source = $source
                MERGE (f)-[:USED_DATA]->(d)
                """

                session.run(
                    input_query,
                    flow_id=flow_id,
                    dataset_id=flow_data["input_data"].get("id", "dataset_unknown"),
                    size=flow_data["input_data"].get("size", 0),
                    source=flow_data["input_data"].get("source", "unknown")
                )

            # Link flow to output models
            for model_id in flow_data.get("output_models", []):
                model_query = """
                MATCH (f:TrainingFlow {id: $flow_id})
                MERGE (m:MLModel {id: $model_id})
                MERGE (f)-[:PRODUCED]->(m)
                """
                session.run(model_query, flow_id=flow_id, model_id=model_id)

            return flow_id

    def get_model_lineage(self, model_id: str) -> Dict:
        """
        Get complete lineage for a model (training flow, dataset, performance)

        Returns:
            {model_info, training_flow, dataset_used, predictions_made, performance_metrics}
        """
        if not self.driver:
            return {}

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (m:MLModel {id: $model_id})
            OPTIONAL MATCH (f:TrainingFlow)-[:PRODUCED]->(m)
            OPTIONAL MATCH (f)-[:USED_DATA]->(d:Dataset)
            OPTIONAL MATCH (m)-[p:PREDICTED]->(e:Email)
            RETURN m as model,
                   f as training_flow,
                   d as dataset,
                   count(e) as total_predictions,
                   avg(p.confidence) as avg_confidence
            """

            result = session.run(query, model_id=model_id).single()

            if result:
                return {
                    "model_info": dict(result["model"]) if result["model"] else {},
                    "training_flow": dict(result["training_flow"]) if result["training_flow"] else {},
                    "dataset": dict(result["dataset"]) if result["dataset"] else {},
                    "total_predictions": result["total_predictions"],
                    "avg_confidence": result["avg_confidence"]
                }

            return {}

    def compare_models(self, model_ids: List[str]) -> Dict:
        """
        Compare multiple models side by side

        Returns:
            Comparison of accuracy, predictions, confidence for each model
        """
        if not self.driver:
            return {}

        with self.driver.session(database=self.database) as session:
            comparisons = []

            for model_id in model_ids:
                query = """
                MATCH (m:MLModel {id: $model_id})
                OPTIONAL MATCH (m)-[p:PREDICTED]->(e:Email)
                RETURN m.id as model_id,
                       m.model_type as model_type,
                       m.version as version,
                       m.accuracy as accuracy,
                       m.f1_score as f1_score,
                       count(e) as predictions,
                       avg(p.confidence) as avg_confidence
                """

                result = session.run(query, model_id=model_id).single()
                if result:
                    comparisons.append(dict(result))

            return {
                "models": comparisons,
                "comparison_timestamp": datetime.now().isoformat()
            }

    def export_training_dataset(self, topic: Optional[str] = None, min_confidence: float = 0.8) -> List[Dict]:
        """
        Export curated training dataset from knowledge graph

        Args:
            topic: Filter by specific topic (None = all topics)
            min_confidence: Minimum confidence for predictions to include

        Returns:
            List of training examples with features and labels
        """
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            if topic:
                query = """
                MATCH (e:Email)-[r:HAS_GROUND_TRUTH]->(t:Topic {name: $topic})
                OPTIONAL MATCH (g:FeatureGenerator)-[rf:EXTRACTED_FROM]->(e)
                RETURN e.id as id,
                       e.subject as subject,
                       e.body as body,
                       t.name as label,
                       collect({generator: g.name, features: rf.features}) as features
                """
                results = session.run(query, topic=topic).data()
            else:
                query = """
                MATCH (e:Email)-[r:HAS_GROUND_TRUTH]->(t:Topic)
                OPTIONAL MATCH (g:FeatureGenerator)-[rf:EXTRACTED_FROM]->(e)
                RETURN e.id as id,
                       e.subject as subject,
                       e.body as body,
                       t.name as label,
                       collect({generator: g.name, features: rf.features}) as features
                """
                results = session.run(query).data()

            # Format for ML training
            dataset = []
            for record in results:
                all_features = {}
                for feat_group in record["features"]:
                    if feat_group["features"]:
                        features_dict = json.loads(feat_group["features"])
                        all_features.update(features_dict)

                dataset.append({
                    "id": record["id"],
                    "text": f"{record['subject']} {record['body']}",
                    "subject": record["subject"],
                    "body": record["body"],
                    "label": record["label"],
                    "features": all_features
                })

            return dataset

    def get_mlops_system_overview(self) -> Dict:
        """
        Get complete overview of MLOps system state

        Returns:
            Comprehensive system metrics and status
        """
        if not self.driver:
            return {}

        with self.driver.session(database=self.database) as session:
            overview_query = """
            CALL () {
                MATCH (e:Email) RETURN count(e) as total_emails
            }
            CALL () {
                MATCH (e:Email)-[:HAS_GROUND_TRUTH]->(:Topic) RETURN count(e) as labeled_emails
            }
            CALL () {
                MATCH (m:MLModel) RETURN count(m) as total_models
            }
            CALL () {
                MATCH (f:TrainingFlow) RETURN count(f) as total_flows
            }
            CALL () {
                MATCH (t:Topic) RETURN count(t) as total_topics
            }
            CALL () {
                MATCH (g:FeatureGenerator) RETURN count(g) as total_generators
            }
            CALL () {
                MATCH (ex:Experiment) RETURN count(ex) as total_experiments
            }
            RETURN total_emails, labeled_emails, total_models, total_flows,
                   total_topics, total_generators, total_experiments
            """

            result = session.run(overview_query).single()

            return {
                "system_status": "operational",
                "emails": {
                    "total": result["total_emails"],
                    "labeled": result["labeled_emails"],
                    "unlabeled": result["total_emails"] - result["labeled_emails"]
                },
                "models": {
                    "total": result["total_models"],
                    "active": result["total_models"]  # Can be refined
                },
                "training": {
                    "total_flows": result["total_flows"],
                    "total_experiments": result["total_experiments"]
                },
                "infrastructure": {
                    "topics": result["total_topics"],
                    "feature_generators": result["total_generators"]
                },
                "timestamp": datetime.now().isoformat()
            }

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content"""
        return hashlib.md5(content.encode()).hexdigest()

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()


# Singleton instance
_kg_instance = None

def get_knowledge_graph() -> MLOpsKnowledgeGraph:
    """Get singleton instance of knowledge graph"""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = MLOpsKnowledgeGraph()
    return _kg_instance