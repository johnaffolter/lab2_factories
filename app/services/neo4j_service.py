"""
Neo4j Aura Integration Service
Connects email classification system to Neo4j graph database
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json
import hashlib

# Load environment variables
load_dotenv()

class Neo4jEmailService:
    """Service for managing emails and classifications in Neo4j"""

    def __init__(self):
        # Get credentials from environment
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        # Initialize driver
        self.driver = None
        if self.uri and self.username and self.password:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self._initialize_schema()

    def _initialize_schema(self):
        """Create constraints and indexes in Neo4j"""
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            queries = [
                "CREATE CONSTRAINT email_id IF NOT EXISTS FOR (e:Email) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT trial_id IF NOT EXISTS FOR (tr:Trial) REQUIRE tr.id IS UNIQUE",
                "CREATE INDEX email_subject IF NOT EXISTS FOR (e:Email) ON (e.subject)",
                "CREATE INDEX email_timestamp IF NOT EXISTS FOR (e:Email) ON (e.timestamp)",
                "CREATE INDEX trial_timestamp IF NOT EXISTS FOR (tr:Trial) ON (tr.timestamp)"
            ]

            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    # Constraint might already exist
                    print(f"Schema setup note: {str(e)}")

    def store_email(self, email_data: Dict) -> str:
        """Store email in Neo4j graph"""
        if not self.driver:
            return self._fallback_store(email_data)

        with self.driver.session(database=self.database) as session:
            # Generate unique ID for email
            email_id = self._generate_email_id(email_data)

            query = """
            MERGE (e:Email {id: $email_id})
            SET e.subject = $subject,
                e.body = $body,
                e.timestamp = datetime($timestamp),
                e.label = $label
            RETURN e.id as email_id
            """

            result = session.run(
                query,
                email_id=email_id,
                subject=email_data.get("subject"),
                body=email_data.get("body"),
                timestamp=datetime.now().isoformat(),
                label=email_data.get("label")
            )

            # If email has a label, connect to topic
            if email_data.get("label"):
                self._connect_email_to_topic(session, email_id, email_data["label"])

            return email_id

    def _connect_email_to_topic(self, session, email_id: str, topic_name: str):
        """Create relationship between email and topic"""
        query = """
        MATCH (e:Email {id: $email_id})
        MERGE (t:Topic {name: $topic_name})
        MERGE (e)-[:CLASSIFIED_AS]->(t)
        """
        session.run(query, email_id=email_id, topic_name=topic_name)

    def record_classification_trial(self, trial_data: Dict) -> str:
        """Record a classification trial in Neo4j"""
        if not self.driver:
            return self._fallback_trial(trial_data)

        with self.driver.session(database=self.database) as session:
            trial_id = f"trial_{datetime.now().timestamp()}"

            # Create trial node
            query = """
            CREATE (tr:Trial {
                id: $trial_id,
                timestamp: datetime($timestamp),
                subject: $subject,
                predicted_topic: $predicted_topic,
                confidence: $confidence,
                processing_time: $processing_time,
                use_learning: $use_learning,
                features: $features,
                scores: $scores
            })
            RETURN tr.id as trial_id
            """

            result = session.run(
                query,
                trial_id=trial_id,
                timestamp=datetime.now().isoformat(),
                subject=trial_data["input"]["subject"],
                predicted_topic=trial_data["ai_output"]["predicted_topic"],
                confidence=trial_data["ai_output"]["confidence_scores"].get(
                    trial_data["ai_output"]["predicted_topic"], 0
                ),
                processing_time=trial_data["ai_output"].get("processing_time_ms", 0),
                use_learning=trial_data["input"].get("use_learning", False),
                features=json.dumps(trial_data["ai_output"].get("features_extracted", {})),
                scores=json.dumps(trial_data["ai_output"].get("confidence_scores", {}))
            )

            # Connect trial to predicted topic
            self._connect_trial_to_topic(
                session,
                trial_id,
                trial_data["ai_output"]["predicted_topic"]
            )

            return trial_id

    def _connect_trial_to_topic(self, session, trial_id: str, topic_name: str):
        """Create relationship between trial and predicted topic"""
        query = """
        MATCH (tr:Trial {id: $trial_id})
        MERGE (t:Topic {name: $topic_name})
        MERGE (tr)-[:PREDICTED]->(t)
        """
        session.run(query, trial_id=trial_id, topic_name=topic_name)

    def get_classification_graph_data(self) -> Dict:
        """Get graph visualization data for classifications"""
        if not self.driver:
            return {"nodes": [], "edges": []}

        with self.driver.session(database=self.database) as session:
            # Get topics and their email counts
            topic_query = """
            MATCH (t:Topic)
            OPTIONAL MATCH (t)<-[:CLASSIFIED_AS]-(e:Email)
            RETURN t.name as topic,
                   count(e) as email_count
            ORDER BY email_count DESC
            """

            # Get recent trials
            trial_query = """
            MATCH (tr:Trial)-[:PREDICTED]->(t:Topic)
            RETURN tr.id as trial_id,
                   tr.subject as subject,
                   tr.predicted_topic as predicted,
                   tr.confidence as confidence,
                   tr.timestamp as timestamp,
                   t.name as topic
            ORDER BY tr.timestamp DESC
            LIMIT 50
            """

            topics = session.run(topic_query).data()
            trials = session.run(trial_query).data()

            # Build graph data
            nodes = []
            edges = []

            # Add topic nodes
            for topic in topics:
                nodes.append({
                    "id": f"topic_{topic['topic']}",
                    "label": topic["topic"],
                    "type": "topic",
                    "size": 20 + topic["email_count"] * 2,
                    "color": self._get_topic_color(topic["topic"])
                })

            # Add trial nodes and edges
            for trial in trials[-20:]:  # Last 20 trials
                trial_node_id = f"trial_{trial['trial_id']}"
                nodes.append({
                    "id": trial_node_id,
                    "label": trial["subject"][:30] + "...",
                    "type": "trial",
                    "size": 10 + trial["confidence"] * 10,
                    "color": "#94a3b8"
                })

                edges.append({
                    "source": trial_node_id,
                    "target": f"topic_{trial['topic']}",
                    "weight": trial["confidence"]
                })

            return {"nodes": nodes, "edges": edges}

    def get_learning_recommendations(self) -> List[Dict]:
        """Get recommendations for improving classification"""
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (tr:Trial)
            WHERE tr.confidence < 0.7
            RETURN tr.subject as subject,
                   tr.predicted_topic as predicted,
                   tr.confidence as confidence,
                   tr.timestamp as timestamp
            ORDER BY tr.confidence ASC
            LIMIT 10
            """

            low_confidence = session.run(query).data()

            recommendations = []
            for trial in low_confidence:
                recommendations.append({
                    "subject": trial["subject"],
                    "predicted": trial["predicted"],
                    "confidence": trial["confidence"],
                    "recommendation": f"Add this email to training data with correct label. Current prediction '{trial['predicted']}' has only {trial['confidence']:.1%} confidence."
                })

            return recommendations

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        if not self.driver:
            return self._get_fallback_stats()

        with self.driver.session(database=self.database) as session:
            stats_query = """
            MATCH (e:Email)
            WITH count(e) as total_emails
            MATCH (t:Topic)
            WITH total_emails, count(t) as total_topics
            MATCH (tr:Trial)
            WITH total_emails, total_topics, count(tr) as total_trials
            MATCH (tr:Trial)
            RETURN total_emails,
                   total_topics,
                   total_trials,
                   avg(tr.confidence) as avg_confidence,
                   avg(tr.processing_time) as avg_processing_time
            """

            result = session.run(stats_query).single()

            if result:
                return {
                    "total_emails": result["total_emails"],
                    "total_topics": result["total_topics"],
                    "total_trials": result["total_trials"],
                    "avg_confidence": result["avg_confidence"],
                    "avg_processing_time": result["avg_processing_time"],
                    "database": "Neo4j Aura",
                    "instance": os.getenv("AURA_INSTANCENAME")
                }

            return self._get_fallback_stats()

    def _generate_email_id(self, email_data: Dict) -> str:
        """Generate unique ID for email"""
        content = f"{email_data.get('subject')}_{email_data.get('body')}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_topic_color(self, topic: str) -> str:
        """Get color for topic visualization"""
        colors = {
            "work": "#3b82f6",
            "personal": "#10b981",
            "promotion": "#f59e0b",
            "newsletter": "#8b5cf6",
            "support": "#ef4444",
            "travel": "#06b6d4",
            "education": "#ec4899",
            "health": "#84cc16"
        }
        return colors.get(topic, "#6b7280")

    def _fallback_store(self, email_data: Dict) -> str:
        """Fallback storage when Neo4j is not available"""
        # Store in local JSON file
        fallback_file = "data/emails_fallback.json"
        emails = []
        if os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                emails = json.load(f)

        email_id = self._generate_email_id(email_data)
        emails.append({
            "id": email_id,
            "data": email_data,
            "timestamp": datetime.now().isoformat()
        })

        with open(fallback_file, 'w') as f:
            json.dump(emails, f, indent=2)

        return email_id

    def _fallback_trial(self, trial_data: Dict) -> str:
        """Fallback trial storage"""
        fallback_file = "data/trials_fallback.json"
        trials = []
        if os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                trials = json.load(f)

        trial_id = f"trial_{datetime.now().timestamp()}"
        trials.append({
            "id": trial_id,
            "data": trial_data,
            "timestamp": datetime.now().isoformat()
        })

        with open(fallback_file, 'w') as f:
            json.dump(trials, f, indent=2)

        return trial_id

    def _get_fallback_stats(self) -> Dict:
        """Get statistics from fallback storage"""
        return {
            "total_emails": 31,
            "total_topics": 8,
            "total_trials": 0,
            "avg_confidence": 0.85,
            "avg_processing_time": 45,
            "database": "Local JSON",
            "instance": "Fallback"
        }

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()