"""
Advanced Document Classification and Graph Storage Service
Supports multiple document types and Neo4j visualization
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from neo4j import GraphDatabase
import hashlib
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class DocumentType(Enum):
    """Supported document types"""
    EMAIL = "email"
    REPORT = "report"
    SCREENSHOT = "screenshot"
    CODE = "code"
    DOCUMENTATION = "documentation"
    MEETING_NOTES = "meeting_notes"
    RESEARCH_PAPER = "research_paper"
    INVOICE = "invoice"
    CONTRACT = "contract"
    PRESENTATION = "presentation"

class DocumentGraphService:
    """Service for managing documents in Neo4j knowledge graph"""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "neo4j+s://e0753253.databases.neo4j.io")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        if self.password:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self._initialize_graph_schema()
        else:
            print("⚠️ Neo4j credentials not found, using local fallback")
            self.driver = None

    def _initialize_graph_schema(self):
        """Create graph schema for document management"""
        with self.driver.session(database=self.database) as session:
            queries = [
                # Document node constraints
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE INDEX doc_type IF NOT EXISTS FOR (d:Document) ON (d.type)",
                "CREATE INDEX doc_timestamp IF NOT EXISTS FOR (d:Document) ON (d.timestamp)",

                # Classification constraints
                "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Classification) REQUIRE c.id IS UNIQUE",

                # Feature constraints
                "CREATE CONSTRAINT feature_name IF NOT EXISTS FOR (f:Feature) REQUIRE f.name IS UNIQUE",

                # User constraints
                "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",

                # Project constraints
                "CREATE CONSTRAINT project_name IF NOT EXISTS FOR (p:Project) REQUIRE p.name IS UNIQUE"
            ]

            for query in queries:
                try:
                    session.run(query)
                except Exception as e:
                    print(f"Schema note: {str(e)[:50]}")

    def store_document(self, document_data: Dict[str, Any]) -> str:
        """Store document in graph with rich metadata"""
        if not self.driver:
            return self._fallback_store_document(document_data)

        with self.driver.session(database=self.database) as session:
            doc_id = self._generate_document_id(document_data)

            # Determine document type
            doc_type = document_data.get("type", DocumentType.EMAIL.value)

            query = """
            MERGE (d:Document {id: $doc_id})
            SET d.type = $type,
                d.title = $title,
                d.content = $content,
                d.timestamp = datetime($timestamp),
                d.metadata = $metadata,
                d.file_path = $file_path,
                d.size = $size,
                d.format = $format,
                d.language = $language,
                d.author = $author,
                d.project = $project

            // Create or connect to Topic
            WITH d
            UNWIND $topics as topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (d)-[:ABOUT]->(t)

            // Create or connect to Project
            WITH d
            MERGE (p:Project {name: $project})
            MERGE (d)-[:BELONGS_TO]->(p)

            // Create or connect to Author
            WITH d
            MERGE (u:User {id: $author})
            MERGE (u)-[:CREATED]->(d)

            RETURN d.id as doc_id
            """

            # Extract metadata
            metadata = document_data.get("metadata", {})

            result = session.run(
                query,
                doc_id=doc_id,
                type=doc_type,
                title=document_data.get("title", "Untitled"),
                content=document_data.get("content", ""),
                timestamp=datetime.now().isoformat(),
                metadata=json.dumps(metadata),
                file_path=document_data.get("file_path", ""),
                size=document_data.get("size", 0),
                format=document_data.get("format", "text"),
                language=document_data.get("language", "en"),
                author=document_data.get("author", "system"),
                project=document_data.get("project", "email_classification"),
                topics=document_data.get("topics", [])
            )

            return doc_id

    def classify_and_store(self, document: Dict, classification_result: Dict) -> str:
        """Classify document and store results in graph"""
        if not self.driver:
            return self._fallback_classify(document, classification_result)

        with self.driver.session(database=self.database) as session:
            # Store the document first
            doc_id = self.store_document(document)

            # Store classification
            class_id = f"class_{datetime.now().timestamp()}"

            query = """
            MATCH (d:Document {id: $doc_id})
            CREATE (c:Classification {
                id: $class_id,
                timestamp: datetime($timestamp),
                predicted_topic: $predicted_topic,
                confidence: $confidence,
                algorithm: $algorithm,
                features: $features,
                scores: $scores
            })
            MERGE (d)-[:CLASSIFIED_AS]->(c)

            // Connect to predicted topic
            MERGE (t:Topic {name: $predicted_topic})
            MERGE (c)-[:PREDICTS]->(t)

            // Store features as separate nodes
            WITH c
            UNWIND $feature_list as feature
            MERGE (f:Feature {name: feature.name})
            SET f.value = feature.value,
                f.type = feature.type
            CREATE (c)-[:USES_FEATURE]->(f)

            RETURN c.id as class_id
            """

            # Prepare features for storage
            features = classification_result.get("features", {})
            feature_list = [
                {"name": k, "value": str(v), "type": type(v).__name__}
                for k, v in features.items()
            ]

            result = session.run(
                query,
                doc_id=doc_id,
                class_id=class_id,
                timestamp=datetime.now().isoformat(),
                predicted_topic=classification_result.get("predicted_topic"),
                confidence=classification_result.get("confidence", 0),
                algorithm=classification_result.get("algorithm", "cosine_similarity"),
                features=json.dumps(features),
                scores=json.dumps(classification_result.get("topic_scores", {})),
                feature_list=feature_list
            )

            return class_id

    def get_document_graph(self, limit: int = 50) -> Dict:
        """Get document relationship graph for visualization"""
        if not self.driver:
            return {"nodes": [], "edges": []}

        with self.driver.session(database=self.database) as session:
            query = """
            // Get documents with their relationships
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:ABOUT]->(t:Topic)
            OPTIONAL MATCH (d)-[:CLASSIFIED_AS]->(c:Classification)
            OPTIONAL MATCH (d)-[:BELONGS_TO]->(p:Project)
            OPTIONAL MATCH (u:User)-[:CREATED]->(d)

            WITH d, collect(DISTINCT t) as topics,
                 collect(DISTINCT c) as classifications,
                 p, u
            LIMIT $limit

            RETURN d.id as doc_id,
                   d.type as doc_type,
                   d.title as title,
                   d.timestamp as timestamp,
                   [t IN topics | t.name] as topics,
                   [c IN classifications | c.predicted_topic] as predictions,
                   p.name as project,
                   u.id as author
            ORDER BY d.timestamp DESC
            """

            results = session.run(query, limit=limit).data()

            # Build graph structure
            nodes = []
            edges = []
            seen_nodes = set()

            for record in results:
                # Add document node
                doc_node_id = f"doc_{record['doc_id']}"
                if doc_node_id not in seen_nodes:
                    nodes.append({
                        "id": doc_node_id,
                        "label": record['title'][:30],
                        "type": "document",
                        "doc_type": record['doc_type'],
                        "color": self._get_doc_color(record['doc_type'])
                    })
                    seen_nodes.add(doc_node_id)

                # Add topic nodes and edges
                for topic in record['topics'] or []:
                    topic_node_id = f"topic_{topic}"
                    if topic_node_id not in seen_nodes:
                        nodes.append({
                            "id": topic_node_id,
                            "label": topic,
                            "type": "topic",
                            "color": "#10b981"
                        })
                        seen_nodes.add(topic_node_id)

                    edges.append({
                        "source": doc_node_id,
                        "target": topic_node_id,
                        "type": "about"
                    })

                # Add project node
                if record['project']:
                    project_node_id = f"project_{record['project']}"
                    if project_node_id not in seen_nodes:
                        nodes.append({
                            "id": project_node_id,
                            "label": record['project'],
                            "type": "project",
                            "color": "#8b5cf6"
                        })
                        seen_nodes.add(project_node_id)

                    edges.append({
                        "source": doc_node_id,
                        "target": project_node_id,
                        "type": "belongs_to"
                    })

            return {"nodes": nodes, "edges": edges}

    def search_similar_documents(self, document: Dict, limit: int = 5) -> List[Dict]:
        """Find similar documents using graph patterns"""
        if not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            # Extract key features
            topics = document.get("topics", [])
            doc_type = document.get("type", "email")

            query = """
            // Find documents with similar characteristics
            MATCH (d:Document)
            WHERE d.type = $doc_type

            OPTIONAL MATCH (d)-[:ABOUT]->(t:Topic)
            WHERE t.name IN $topics

            WITH d, count(DISTINCT t) as shared_topics
            WHERE shared_topics > 0

            RETURN d.id as id,
                   d.title as title,
                   d.type as type,
                   d.content as content,
                   shared_topics,
                   d.timestamp as timestamp
            ORDER BY shared_topics DESC, d.timestamp DESC
            LIMIT $limit
            """

            results = session.run(
                query,
                doc_type=doc_type,
                topics=topics,
                limit=limit
            ).data()

            return results

    def get_document_insights(self) -> Dict:
        """Get insights about document collection"""
        if not self.driver:
            return self._get_fallback_insights()

        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (d:Document)
            WITH count(d) as total_docs,
                 collect(DISTINCT d.type) as doc_types

            MATCH (t:Topic)<-[:ABOUT]-(d:Document)
            WITH total_docs, doc_types, t, count(d) as docs_per_topic
            ORDER BY docs_per_topic DESC
            LIMIT 10

            WITH total_docs, doc_types,
                 collect({topic: t.name, count: docs_per_topic}) as top_topics

            MATCH (c:Classification)
            WITH total_docs, doc_types, top_topics,
                 avg(c.confidence) as avg_confidence,
                 count(c) as total_classifications

            MATCH (p:Project)<-[:BELONGS_TO]-(d:Document)
            WITH total_docs, doc_types, top_topics,
                 avg_confidence, total_classifications,
                 count(DISTINCT p) as project_count

            RETURN total_docs, doc_types, top_topics,
                   avg_confidence, total_classifications,
                   project_count
            """

            result = session.run(query).single()

            if result:
                return {
                    "total_documents": result["total_docs"],
                    "document_types": result["doc_types"],
                    "top_topics": result["top_topics"],
                    "average_confidence": result["avg_confidence"],
                    "total_classifications": result["total_classifications"],
                    "project_count": result["project_count"]
                }

            return self._get_fallback_insights()

    def _generate_document_id(self, document: Dict) -> str:
        """Generate unique document ID"""
        content = f"{document.get('title', '')}_{document.get('content', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_doc_color(self, doc_type: str) -> str:
        """Get color for document type visualization"""
        colors = {
            "email": "#3b82f6",
            "report": "#ef4444",
            "screenshot": "#f59e0b",
            "code": "#10b981",
            "documentation": "#6366f1",
            "meeting_notes": "#ec4899",
            "research_paper": "#8b5cf6",
            "invoice": "#14b8a6",
            "contract": "#f97316",
            "presentation": "#84cc16"
        }
        return colors.get(doc_type, "#6b7280")

    def _fallback_store_document(self, document: Dict) -> str:
        """Fallback document storage"""
        doc_id = self._generate_document_id(document)
        # Store locally
        fallback_file = "data/documents_fallback.json"
        documents = []

        if os.path.exists(fallback_file):
            with open(fallback_file, 'r') as f:
                documents = json.load(f)

        documents.append({
            "id": doc_id,
            "document": document,
            "timestamp": datetime.now().isoformat()
        })

        with open(fallback_file, 'w') as f:
            json.dump(documents, f, indent=2)

        return doc_id

    def _fallback_classify(self, document: Dict, classification: Dict) -> str:
        """Fallback classification storage"""
        class_id = f"class_{datetime.now().timestamp()}"
        return class_id

    def _get_fallback_insights(self) -> Dict:
        """Get fallback insights"""
        return {
            "total_documents": 0,
            "document_types": [],
            "top_topics": [],
            "average_confidence": 0,
            "total_classifications": 0,
            "project_count": 0
        }

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()