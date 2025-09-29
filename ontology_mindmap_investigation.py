#!/usr/bin/env python3
"""
Ontology Mind Mapping and Investigation System
Comprehensive knowledge graph construction, automated reasoning, and investigation framework
"""

import json
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import re
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

class InvestigationLevel(Enum):
    """Investigation depth levels"""
    SURFACE = "surface"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"

class NodeType(Enum):
    """Ontology node types"""
    CONCEPT = "concept"
    ENTITY = "entity"
    PROCESS = "process"
    PROPERTY = "property"
    RELATIONSHIP = "relationship"
    CONSTRAINT = "constraint"
    RULE = "rule"
    PATTERN = "pattern"
    EXAMPLE = "example"
    METRIC = "metric"

class RelationshipType(Enum):
    """Relationship types in the ontology"""
    IS_A = "is_a"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"
    IMPLEMENTS = "implements"
    GENERATES = "generates"
    PROCESSES = "processes"
    CONTAINS = "contains"
    SUPPORTS = "supports"
    VALIDATES = "validates"
    TRANSFORMS = "transforms"
    FLOWS_TO = "flows_to"

@dataclass
class OntologyNode:
    """Node in the ontology graph"""
    id: str
    name: str
    type: NodeType
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class OntologyRelationship:
    """Relationship between ontology nodes"""
    id: str
    source_id: str
    target_id: str
    type: RelationshipType
    strength: float
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class InvestigationPath:
    """Investigation reasoning path"""
    path_id: str
    nodes: List[str]
    relationships: List[str]
    reasoning: str
    confidence: float
    evidence_strength: float
    insights: List[str]

@dataclass
class MindMapVisualization:
    """Mind map visualization configuration"""
    layout: str
    color_scheme: str
    node_sizes: Dict[str, float]
    edge_weights: Dict[str, float]
    clusters: Dict[str, List[str]]
    annotations: Dict[str, str]

class OntologyMindMapper:
    """Advanced ontology mind mapping system"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, OntologyNode] = {}
        self.relationships: Dict[str, OntologyRelationship] = {}
        self.investigation_paths: List[InvestigationPath] = []
        self.domain_ontologies = self._initialize_domain_ontologies()

    def _initialize_domain_ontologies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive domain ontologies"""
        return {
            "email_classification": {
                "core_concepts": {
                    "factory_pattern": {
                        "description": "Creational design pattern for object instantiation",
                        "components": ["abstract_factory", "concrete_factory", "product", "client"],
                        "benefits": ["flexibility", "extensibility", "testability", "decoupling"],
                        "implementation": ["generator_registry", "feature_extraction", "classification"]
                    },
                    "feature_engineering": {
                        "description": "Process of selecting and transforming variables for ML models",
                        "types": ["numerical", "categorical", "text", "temporal", "composite"],
                        "techniques": ["extraction", "transformation", "selection", "scaling", "encoding"]
                    },
                    "classification_pipeline": {
                        "description": "End-to-end process for email categorization",
                        "stages": ["ingestion", "preprocessing", "feature_extraction", "classification", "validation"],
                        "components": ["data_loader", "preprocessor", "feature_generator", "classifier", "evaluator"]
                    }
                },
                "data_flow": {
                    "ingestion": ["airbyte", "api_sources", "file_uploads", "streaming"],
                    "processing": ["validation", "cleaning", "transformation", "enrichment"],
                    "storage": ["postgres", "neo4j", "s3", "snowflake"],
                    "analysis": ["ml_models", "rules_engine", "pattern_matching", "anomaly_detection"]
                },
                "quality_metrics": {
                    "accuracy": "Percentage of correctly classified emails",
                    "precision": "True positives / (True positives + False positives)",
                    "recall": "True positives / (True positives + False negatives)",
                    "f1_score": "Harmonic mean of precision and recall",
                    "confidence": "Model certainty in predictions"
                }
            },

            "business_domains": {
                "finance": {
                    "entities": ["invoice", "receipt", "contract", "financial_statement", "budget"],
                    "processes": ["accounting", "budgeting", "forecasting", "auditing", "compliance"],
                    "metrics": ["revenue", "profit", "roi", "cash_flow", "debt_ratio"],
                    "systems": ["erp", "accounting_software", "banking_systems", "payment_processors"]
                },
                "marketing": {
                    "entities": ["campaign", "lead", "customer", "content", "channel"],
                    "processes": ["segmentation", "targeting", "positioning", "execution", "measurement"],
                    "metrics": ["ctr", "conversion_rate", "cac", "ltv", "roas"],
                    "systems": ["crm", "marketing_automation", "analytics_platforms", "social_media"]
                },
                "operations": {
                    "entities": ["process", "resource", "workflow", "system", "output"],
                    "processes": ["planning", "execution", "monitoring", "optimization", "control"],
                    "metrics": ["efficiency", "productivity", "quality", "cost", "time"],
                    "systems": ["erp", "workflow_management", "monitoring_tools", "automation"]
                }
            },

            "data_architecture": {
                "ingestion_patterns": {
                    "batch": {"frequency": "scheduled", "volume": "high", "latency": "hours"},
                    "streaming": {"frequency": "continuous", "volume": "medium", "latency": "seconds"},
                    "real_time": {"frequency": "event_driven", "volume": "low", "latency": "milliseconds"}
                },
                "storage_patterns": {
                    "data_lake": {"s3", "azure_data_lake", "google_cloud_storage"},
                    "data_warehouse": {"snowflake", "redshift", "bigquery"},
                    "operational_db": {"postgres", "mysql", "mongodb"},
                    "graph_db": {"neo4j", "amazon_neptune", "azure_cosmos_db"}
                },
                "processing_patterns": {
                    "etl": {"extract", "transform", "load"},
                    "elt": {"extract", "load", "transform"},
                    "streaming": {"kafka", "kinesis", "pubsub"},
                    "batch": {"airflow", "dagster", "prefect"}
                }
            },

            "ml_operations": {
                "model_lifecycle": {
                    "development": ["research", "experimentation", "training", "validation"],
                    "deployment": ["packaging", "serving", "monitoring", "scaling"],
                    "maintenance": ["retraining", "updating", "debugging", "optimization"]
                },
                "feature_engineering": {
                    "text_features": ["tfidf", "word_embeddings", "ngrams", "sentiment"],
                    "numerical_features": ["scaling", "binning", "polynomial", "interaction"],
                    "temporal_features": ["seasonality", "trends", "lags", "windows"]
                },
                "evaluation_metrics": {
                    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
                    "regression": ["mse", "mae", "r2", "mape"],
                    "clustering": ["silhouette", "davies_bouldin", "calinski_harabasz"]
                }
            }
        }

    def build_comprehensive_ontology(self, domain: str = "email_classification") -> None:
        """Build comprehensive ontology for specified domain"""

        print(f"🧠 Building comprehensive ontology for domain: {domain}")

        if domain not in self.domain_ontologies:
            raise ValueError(f"Domain {domain} not supported")

        domain_data = self.domain_ontologies[domain]

        # Add core concept nodes
        self._add_concept_nodes(domain_data)

        # Add relationship networks
        self._add_relationship_networks(domain_data)

        # Add process flows
        self._add_process_flows(domain_data)

        # Add quality and validation nodes
        self._add_quality_nodes(domain_data)

        print(f"✓ Ontology built: {len(self.nodes)} nodes, {len(self.relationships)} relationships")

    def _add_concept_nodes(self, domain_data: Dict[str, Any]) -> None:
        """Add concept nodes to the ontology"""

        for section_name, section_data in domain_data.items():
            # Create section node
            section_node = OntologyNode(
                id=f"section_{section_name}",
                name=section_name.replace("_", " ").title(),
                type=NodeType.CONCEPT,
                description=f"Domain section: {section_name}",
                tags=["domain_section"]
            )
            self._add_node(section_node)

            if isinstance(section_data, dict):
                for concept_name, concept_data in section_data.items():
                    concept_node = OntologyNode(
                        id=f"concept_{concept_name}",
                        name=concept_name.replace("_", " ").title(),
                        type=NodeType.CONCEPT,
                        description=self._extract_description(concept_data),
                        properties=self._extract_properties(concept_data),
                        tags=["core_concept", section_name]
                    )
                    self._add_node(concept_node)

                    # Link to section
                    self._add_relationship(
                        section_node.id, concept_node.id,
                        RelationshipType.CONTAINS, 1.0,
                        f"Section {section_name} contains concept {concept_name}"
                    )

                    # Add sub-concepts if present
                    if isinstance(concept_data, dict):
                        self._add_sub_concepts(concept_node, concept_data)

    def _add_sub_concepts(self, parent_node: OntologyNode, concept_data: Dict[str, Any]) -> None:
        """Add sub-concepts recursively"""

        for key, value in concept_data.items():
            if key in ["description"]:
                continue

            if isinstance(value, list):
                for item in value:
                    sub_concept_node = OntologyNode(
                        id=f"subconcept_{parent_node.id}_{item}",
                        name=str(item).replace("_", " ").title(),
                        type=NodeType.ENTITY,
                        description=f"{key.replace('_', ' ').title()}: {item}",
                        tags=["sub_concept", key]
                    )
                    self._add_node(sub_concept_node)

                    self._add_relationship(
                        parent_node.id, sub_concept_node.id,
                        RelationshipType.CONTAINS, 0.8,
                        f"{parent_node.name} contains {item}"
                    )

            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_concept_node = OntologyNode(
                        id=f"subconcept_{parent_node.id}_{sub_key}",
                        name=sub_key.replace("_", " ").title(),
                        type=NodeType.PROPERTY,
                        description=str(sub_value),
                        tags=["property", key]
                    )
                    self._add_node(sub_concept_node)

                    self._add_relationship(
                        parent_node.id, sub_concept_node.id,
                        RelationshipType.CONTAINS, 0.7,
                        f"{parent_node.name} has property {sub_key}"
                    )

    def _add_relationship_networks(self, domain_data: Dict[str, Any]) -> None:
        """Add complex relationship networks"""

        # Factory pattern relationships
        if "core_concepts" in domain_data and "factory_pattern" in domain_data["core_concepts"]:
            self._add_factory_pattern_relationships()

        # Data flow relationships
        if "data_flow" in domain_data:
            self._add_data_flow_relationships(domain_data["data_flow"])

        # Cross-domain relationships
        self._add_cross_domain_relationships()

    def _add_factory_pattern_relationships(self) -> None:
        """Add factory pattern specific relationships"""

        factory_concepts = [
            "abstract_factory", "concrete_factory", "product", "client",
            "generator_registry", "feature_extraction", "classification"
        ]

        # Create dependency network
        dependencies = [
            ("client", "abstract_factory", "depends_on"),
            ("concrete_factory", "abstract_factory", "implements"),
            ("concrete_factory", "product", "generates"),
            ("generator_registry", "concrete_factory", "contains"),
            ("feature_extraction", "generator_registry", "uses"),
            ("classification", "feature_extraction", "depends_on")
        ]

        for source, target, rel_type in dependencies:
            source_id = f"subconcept_concept_factory_pattern_{source}"
            target_id = f"subconcept_concept_factory_pattern_{target}"

            if source_id in self.nodes and target_id in self.nodes:
                self._add_relationship(
                    source_id, target_id,
                    RelationshipType(rel_type), 0.9,
                    f"Factory pattern: {source} {rel_type} {target}"
                )

    def _add_data_flow_relationships(self, data_flow: Dict[str, Any]) -> None:
        """Add data flow relationships"""

        flow_stages = ["ingestion", "processing", "storage", "analysis"]

        for i in range(len(flow_stages) - 1):
            current_stage = flow_stages[i]
            next_stage = flow_stages[i + 1]

            current_id = f"subconcept_section_data_flow_{current_stage}"
            next_id = f"subconcept_section_data_flow_{next_stage}"

            if current_id in self.nodes and next_id in self.nodes:
                self._add_relationship(
                    current_id, next_id,
                    RelationshipType.FLOWS_TO, 1.0,
                    f"Data flows from {current_stage} to {next_stage}"
                )

    def _add_process_flows(self, domain_data: Dict[str, Any]) -> None:
        """Add process flow nodes and relationships"""

        # ML pipeline process
        ml_processes = [
            "data_ingestion", "data_validation", "feature_engineering",
            "model_training", "model_evaluation", "model_deployment",
            "monitoring", "retraining"
        ]

        previous_process = None
        for process in ml_processes:
            process_node = OntologyNode(
                id=f"process_{process}",
                name=process.replace("_", " ").title(),
                type=NodeType.PROCESS,
                description=f"ML pipeline process: {process}",
                tags=["ml_process", "pipeline"]
            )
            self._add_node(process_node)

            if previous_process:
                self._add_relationship(
                    previous_process.id, process_node.id,
                    RelationshipType.FLOWS_TO, 0.9,
                    f"Process flow: {previous_process.name} to {process_node.name}"
                )

            previous_process = process_node

    def _add_quality_nodes(self, domain_data: Dict[str, Any]) -> None:
        """Add quality and validation nodes"""

        if "quality_metrics" in domain_data:
            quality_metrics = domain_data["quality_metrics"]

            for metric_name, metric_description in quality_metrics.items():
                metric_node = OntologyNode(
                    id=f"metric_{metric_name}",
                    name=metric_name.replace("_", " ").title(),
                    type=NodeType.METRIC,
                    description=metric_description,
                    tags=["quality_metric", "evaluation"]
                )
                self._add_node(metric_node)

                # Link metrics to classification process
                classification_process = "process_model_evaluation"
                if classification_process in self.nodes:
                    self._add_relationship(
                        classification_process, metric_node.id,
                        RelationshipType.GENERATES, 0.8,
                        f"Model evaluation generates {metric_name}"
                    )

    def _add_cross_domain_relationships(self) -> None:
        """Add relationships across different domains"""

        # Link feature engineering to factory pattern
        feature_eng_id = "concept_feature_engineering"
        factory_pattern_id = "concept_factory_pattern"

        if feature_eng_id in self.nodes and factory_pattern_id in self.nodes:
            self._add_relationship(
                factory_pattern_id, feature_eng_id,
                RelationshipType.IMPLEMENTS, 0.8,
                "Factory pattern implements feature engineering"
            )

    def investigate_concept(self, concept_id: str, level: InvestigationLevel = InvestigationLevel.DETAILED) -> Dict[str, Any]:
        """Perform deep investigation of a concept"""

        if concept_id not in self.nodes:
            return {"error": f"Concept {concept_id} not found"}

        print(f"🔍 Investigating concept: {self.nodes[concept_id].name}")
        print(f"Investigation level: {level.value}")

        investigation = {
            "concept": self.nodes[concept_id],
            "direct_relationships": self._get_direct_relationships(concept_id),
            "investigation_paths": [],
            "insights": [],
            "recommendations": []
        }

        if level in [InvestigationLevel.DETAILED, InvestigationLevel.COMPREHENSIVE, InvestigationLevel.EXHAUSTIVE]:
            investigation["extended_network"] = self._get_extended_network(concept_id, depth=2)
            investigation["influence_analysis"] = self._analyze_influence(concept_id)

        if level in [InvestigationLevel.COMPREHENSIVE, InvestigationLevel.EXHAUSTIVE]:
            investigation["pattern_analysis"] = self._analyze_patterns(concept_id)
            investigation["dependency_chains"] = self._find_dependency_chains(concept_id)

        if level == InvestigationLevel.EXHAUSTIVE:
            investigation["reasoning_paths"] = self._generate_reasoning_paths(concept_id)
            investigation["cross_domain_analysis"] = self._analyze_cross_domain_connections(concept_id)

        # Generate insights
        investigation["insights"] = self._generate_insights(investigation)
        investigation["recommendations"] = self._generate_recommendations(investigation)

        return investigation

    def _get_direct_relationships(self, node_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get direct relationships for a node"""

        incoming = []
        outgoing = []

        for rel_id, relationship in self.relationships.items():
            if relationship.target_id == node_id:
                incoming.append({
                    "source": self.nodes[relationship.source_id].name,
                    "type": relationship.type.value,
                    "strength": relationship.strength,
                    "description": relationship.description
                })
            elif relationship.source_id == node_id:
                outgoing.append({
                    "target": self.nodes[relationship.target_id].name,
                    "type": relationship.type.value,
                    "strength": relationship.strength,
                    "description": relationship.description
                })

        return {"incoming": incoming, "outgoing": outgoing}

    def _get_extended_network(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get extended network around a node"""

        visited = set()
        network = {"nodes": [], "relationships": []}

        def explore(current_id: str, current_depth: int):
            if current_depth > depth or current_id in visited:
                return

            visited.add(current_id)
            network["nodes"].append(self.nodes[current_id])

            for rel_id, relationship in self.relationships.items():
                if relationship.source_id == current_id:
                    network["relationships"].append(relationship)
                    explore(relationship.target_id, current_depth + 1)
                elif relationship.target_id == current_id:
                    network["relationships"].append(relationship)
                    explore(relationship.source_id, current_depth + 1)

        explore(node_id, 0)
        return network

    def _analyze_influence(self, node_id: str) -> Dict[str, Any]:
        """Analyze influence patterns of a node"""

        # Calculate influence metrics
        incoming_strength = sum(
            rel.strength for rel in self.relationships.values()
            if rel.target_id == node_id
        )

        outgoing_strength = sum(
            rel.strength for rel in self.relationships.values()
            if rel.source_id == node_id
        )

        # Count relationship types
        rel_types = Counter()
        for rel in self.relationships.values():
            if rel.source_id == node_id or rel.target_id == node_id:
                rel_types[rel.type.value] += 1

        return {
            "influence_score": outgoing_strength - incoming_strength,
            "dependency_score": incoming_strength,
            "impact_score": outgoing_strength,
            "relationship_diversity": len(rel_types),
            "most_common_relationships": rel_types.most_common(3)
        }

    def _analyze_patterns(self, node_id: str) -> Dict[str, Any]:
        """Analyze patterns around a node"""

        patterns = {
            "hub_pattern": False,
            "bridge_pattern": False,
            "leaf_pattern": False,
            "cluster_center": False
        }

        # Get node's connections
        connections = len([
            rel for rel in self.relationships.values()
            if rel.source_id == node_id or rel.target_id == node_id
        ])

        # Hub pattern: many connections
        if connections > 5:
            patterns["hub_pattern"] = True

        # Leaf pattern: few connections
        if connections <= 2:
            patterns["leaf_pattern"] = True

        # Bridge pattern: connects different clusters
        # (Simplified heuristic)
        neighbor_tags = set()
        for rel in self.relationships.values():
            if rel.source_id == node_id:
                neighbor_tags.update(self.nodes[rel.target_id].tags)
            elif rel.target_id == node_id:
                neighbor_tags.update(self.nodes[rel.source_id].tags)

        if len(neighbor_tags) > 3:
            patterns["bridge_pattern"] = True

        return patterns

    def _find_dependency_chains(self, node_id: str) -> List[List[str]]:
        """Find dependency chains involving the node"""

        chains = []

        def trace_chain(current_id: str, path: List[str], direction: str = "forward"):
            if len(path) > 10:  # Prevent infinite loops
                return

            found_dependency = False

            for rel in self.relationships.values():
                if rel.type == RelationshipType.DEPENDS_ON:
                    if direction == "forward" and rel.source_id == current_id:
                        new_path = path + [rel.target_id]
                        if len(new_path) > 2:  # Minimum chain length
                            chains.append(new_path.copy())
                        trace_chain(rel.target_id, new_path, direction)
                        found_dependency = True
                    elif direction == "backward" and rel.target_id == current_id:
                        new_path = [rel.source_id] + path
                        if len(new_path) > 2:
                            chains.append(new_path.copy())
                        trace_chain(rel.source_id, new_path, direction)
                        found_dependency = True

        # Trace forward and backward
        trace_chain(node_id, [node_id], "forward")
        trace_chain(node_id, [node_id], "backward")

        return chains[:10]  # Return top 10 chains

    def _generate_reasoning_paths(self, node_id: str) -> List[InvestigationPath]:
        """Generate reasoning paths for investigation"""

        paths = []

        # Find all paths to important nodes
        important_types = [NodeType.PROCESS, NodeType.METRIC, NodeType.PATTERN]

        for target_node_id, target_node in self.nodes.items():
            if target_node.type in important_types and target_node_id != node_id:
                try:
                    path = nx.shortest_path(self.graph, node_id, target_node_id)
                    if len(path) > 1 and len(path) <= 5:  # Reasonable path length
                        path_obj = InvestigationPath(
                            path_id=str(uuid.uuid4()),
                            nodes=path,
                            relationships=self._get_path_relationships(path),
                            reasoning=self._generate_path_reasoning(path),
                            confidence=self._calculate_path_confidence(path),
                            evidence_strength=0.8,  # Simplified
                            insights=self._generate_path_insights(path)
                        )
                        paths.append(path_obj)
                except nx.NetworkXNoPath:
                    continue

        return sorted(paths, key=lambda p: p.confidence, reverse=True)[:5]

    def _get_path_relationships(self, path: List[str]) -> List[str]:
        """Get relationships in a path"""

        path_rels = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            for rel in self.relationships.values():
                if rel.source_id == source_id and rel.target_id == target_id:
                    path_rels.append(rel.id)
                    break

        return path_rels

    def _generate_path_reasoning(self, path: List[str]) -> str:
        """Generate reasoning for a path"""

        if len(path) < 2:
            return "No reasoning path"

        start_name = self.nodes[path[0]].name
        end_name = self.nodes[path[-1]].name

        reasoning_parts = [f"Starting from {start_name}"]

        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            # Find relationship
            rel_type = "connects to"
            for rel in self.relationships.values():
                if rel.source_id == source_id and rel.target_id == target_id:
                    rel_type = rel.type.value.replace("_", " ")
                    break

            reasoning_parts.append(f"{rel_type} {self.nodes[target_id].name}")

        reasoning_parts.append(f"leading to {end_name}")

        return " → ".join(reasoning_parts)

    def _calculate_path_confidence(self, path: List[str]) -> float:
        """Calculate confidence for a reasoning path"""

        if len(path) < 2:
            return 0.0

        confidences = []
        for node_id in path:
            confidences.append(self.nodes[node_id].confidence)

        # Relationship strengths
        rel_strengths = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            for rel in self.relationships.values():
                if rel.source_id == source_id and rel.target_id == target_id:
                    rel_strengths.append(rel.strength)
                    break

        # Combined confidence
        avg_node_conf = np.mean(confidences) if confidences else 0.5
        avg_rel_strength = np.mean(rel_strengths) if rel_strengths else 0.5

        return (avg_node_conf + avg_rel_strength) / 2

    def _generate_path_insights(self, path: List[str]) -> List[str]:
        """Generate insights for a path"""

        insights = []

        if len(path) > 3:
            insights.append("Complex dependency chain indicates high system coupling")

        # Check for patterns
        node_types = [self.nodes[node_id].type for node_id in path]
        if NodeType.PROCESS in node_types and NodeType.METRIC in node_types:
            insights.append("Path connects processes to measurable outcomes")

        if NodeType.CONCEPT in node_types and NodeType.ENTITY in node_types:
            insights.append("Path bridges abstract concepts with concrete implementations")

        return insights

    def _analyze_cross_domain_connections(self, node_id: str) -> Dict[str, Any]:
        """Analyze cross-domain connections"""

        node = self.nodes[node_id]
        connected_domains = set()

        # Find connected nodes and their domains
        for rel in self.relationships.values():
            other_node_id = None
            if rel.source_id == node_id:
                other_node_id = rel.target_id
            elif rel.target_id == node_id:
                other_node_id = rel.source_id

            if other_node_id:
                other_node = self.nodes[other_node_id]
                connected_domains.update(other_node.tags)

        # Remove current node's domains
        connected_domains -= set(node.tags)

        return {
            "connected_domains": list(connected_domains),
            "cross_domain_strength": len(connected_domains),
            "integration_potential": len(connected_domains) > 2
        }

    def _generate_insights(self, investigation: Dict[str, Any]) -> List[str]:
        """Generate investigation insights"""

        insights = []
        concept = investigation["concept"]

        # Relationship insights
        direct_rels = investigation["direct_relationships"]
        incoming_count = len(direct_rels["incoming"])
        outgoing_count = len(direct_rels["outgoing"])

        if incoming_count > outgoing_count:
            insights.append(f"{concept.name} is primarily a dependency for other concepts")
        elif outgoing_count > incoming_count:
            insights.append(f"{concept.name} influences many other concepts")
        else:
            insights.append(f"{concept.name} has balanced dependencies and influences")

        # Network insights
        if "extended_network" in investigation:
            network_size = len(investigation["extended_network"]["nodes"])
            if network_size > 10:
                insights.append(f"{concept.name} is part of a large interconnected system")

        # Influence insights
        if "influence_analysis" in investigation:
            influence = investigation["influence_analysis"]
            if influence["influence_score"] > 1.0:
                insights.append(f"{concept.name} has high system influence")

        # Pattern insights
        if "pattern_analysis" in investigation:
            patterns = investigation["pattern_analysis"]
            active_patterns = [p for p, active in patterns.items() if active]
            if active_patterns:
                insights.append(f"{concept.name} exhibits {', '.join(active_patterns)}")

        return insights

    def _generate_recommendations(self, investigation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on investigation"""

        recommendations = []
        concept = investigation["concept"]

        # Based on influence analysis
        if "influence_analysis" in investigation:
            influence = investigation["influence_analysis"]

            if influence["dependency_score"] > 2.0:
                recommendations.append(f"Monitor {concept.name} closely as many concepts depend on it")

            if influence["relationship_diversity"] < 2:
                recommendations.append(f"Consider expanding the relationship types for {concept.name}")

        # Based on patterns
        if "pattern_analysis" in investigation:
            patterns = investigation["pattern_analysis"]

            if patterns.get("leaf_pattern"):
                recommendations.append(f"Consider integrating {concept.name} more deeply into the system")

            if patterns.get("bridge_pattern"):
                recommendations.append(f"Leverage {concept.name} as an integration point between domains")

        # Based on reasoning paths
        if "reasoning_paths" in investigation:
            paths = investigation["reasoning_paths"]
            if len(paths) < 3:
                recommendations.append(f"Develop more connections from {concept.name} to key system outcomes")

        return recommendations

    def _add_node(self, node: OntologyNode) -> None:
        """Add node to the ontology"""
        self.nodes[node.id] = node
        self.graph.add_node(node.id, **node.__dict__)

    def _add_relationship(self, source_id: str, target_id: str, rel_type: RelationshipType,
                         strength: float, description: str) -> None:
        """Add relationship to the ontology"""

        rel_id = f"{source_id}_{rel_type.value}_{target_id}"
        relationship = OntologyRelationship(
            id=rel_id,
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            strength=strength,
            description=description
        )

        self.relationships[rel_id] = relationship
        self.graph.add_edge(source_id, target_id, **relationship.__dict__)

    def _extract_description(self, data: Any) -> str:
        """Extract description from data"""
        if isinstance(data, dict) and "description" in data:
            return data["description"]
        elif isinstance(data, str):
            return data
        else:
            return f"Automatically generated for {str(data)[:50]}"

    def _extract_properties(self, data: Any) -> Dict[str, Any]:
        """Extract properties from data"""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k != "description"}
        else:
            return {"value": data}

    def generate_mind_map_visualization(self, focus_node_id: str = None) -> MindMapVisualization:
        """Generate interactive mind map visualization"""

        # Determine layout based on graph size
        if len(self.nodes) < 20:
            layout = "spring"
        elif len(self.nodes) < 50:
            layout = "circular"
        else:
            layout = "hierarchical"

        # Calculate node sizes based on connections
        node_sizes = {}
        for node_id in self.nodes:
            connections = len([
                rel for rel in self.relationships.values()
                if rel.source_id == node_id or rel.target_id == node_id
            ])
            node_sizes[node_id] = 20 + (connections * 5)

        # Calculate edge weights
        edge_weights = {}
        for rel_id, rel in self.relationships.items():
            edge_weights[rel_id] = rel.strength

        # Create clusters based on node types
        clusters = defaultdict(list)
        for node_id, node in self.nodes.items():
            clusters[node.type.value].append(node_id)

        # Generate annotations for important nodes
        annotations = {}
        for node_id, node in self.nodes.items():
            if len(node.description) > 0:
                annotations[node_id] = node.description[:100] + "..."

        return MindMapVisualization(
            layout=layout,
            color_scheme="viridis",
            node_sizes=node_sizes,
            edge_weights=edge_weights,
            clusters=dict(clusters),
            annotations=annotations
        )

    def create_interactive_plotly_mindmap(self, focus_node_id: str = None) -> go.Figure:
        """Create interactive Plotly mind map"""

        viz_config = self.generate_mind_map_visualization(focus_node_id)

        # Create networkx layout
        if viz_config.layout == "spring":
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif viz_config.layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.hierarchical_layout(self.graph) if hasattr(nx, 'hierarchical_layout') else nx.spring_layout(self.graph)

        # Extract coordinates
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []

        color_map = {
            NodeType.CONCEPT: 'blue',
            NodeType.ENTITY: 'green',
            NodeType.PROCESS: 'red',
            NodeType.PROPERTY: 'orange',
            NodeType.METRIC: 'purple'
        }

        for node_id in self.graph.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node = self.nodes[node_id]
            node_info.append(f"{node.name}<br>Type: {node.type.value}<br>Description: {node.description[:100]}")
            node_colors.append(color_map.get(node.type, 'gray'))
            node_sizes.append(viz_config.node_sizes.get(node_id, 20))

        # Create edges
        edge_x = []
        edge_y = []

        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[self.nodes[node_id].name for node_id in self.graph.nodes()],
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Ontology Mind Map",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Ontology visualization showing concepts, relationships, and patterns",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        return fig

    def export_investigation_report(self, concept_id: str, output_dir: str = "investigation_output") -> str:
        """Export comprehensive investigation report"""

        Path(output_dir).mkdir(exist_ok=True)

        # Perform comprehensive investigation
        investigation = self.investigate_concept(concept_id, InvestigationLevel.COMPREHENSIVE)

        # Create report
        report = {
            "investigation_metadata": {
                "concept_id": concept_id,
                "concept_name": investigation["concept"].name,
                "investigation_timestamp": datetime.now().isoformat(),
                "investigation_level": InvestigationLevel.COMPREHENSIVE.value,
                "ontology_size": {
                    "nodes": len(self.nodes),
                    "relationships": len(self.relationships)
                }
            },
            "concept_analysis": {
                "description": investigation["concept"].description,
                "type": investigation["concept"].type.value,
                "properties": investigation["concept"].properties,
                "tags": investigation["concept"].tags
            },
            "relationship_analysis": investigation["direct_relationships"],
            "network_analysis": {
                "extended_network_size": len(investigation.get("extended_network", {}).get("nodes", [])),
                "influence_metrics": investigation.get("influence_analysis", {}),
                "pattern_analysis": investigation.get("pattern_analysis", {})
            },
            "insights": investigation["insights"],
            "recommendations": investigation["recommendations"],
            "reasoning_paths": [
                {
                    "path": [self.nodes[node_id].name for node_id in path.nodes],
                    "reasoning": path.reasoning,
                    "confidence": path.confidence,
                    "insights": path.insights
                }
                for path in investigation.get("reasoning_paths", [])
            ]
        }

        # Save report
        report_file = Path(output_dir) / f"investigation_report_{concept_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create visualization
        fig = self.create_interactive_plotly_mindmap(concept_id)
        viz_file = Path(output_dir) / f"mindmap_{concept_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(viz_file)

        print(f"📊 Investigation report exported:")
        print(f"   Report: {report_file}")
        print(f"   Visualization: {viz_file}")

        return str(report_file)

def main():
    """Demonstrate the ontology mind mapping system"""
    print("🧠 ONTOLOGY MIND MAPPING AND INVESTIGATION SYSTEM")
    print("Comprehensive knowledge graph construction and automated reasoning")
    print("=" * 80)

    # Initialize mind mapper
    mapper = OntologyMindMapper()

    # Build comprehensive ontology
    mapper.build_comprehensive_ontology("email_classification")

    print(f"\n📈 Ontology Statistics:")
    print(f"   Nodes: {len(mapper.nodes)}")
    print(f"   Relationships: {len(mapper.relationships)}")
    print(f"   Node types: {len(set(node.type for node in mapper.nodes.values()))}")
    print(f"   Relationship types: {len(set(rel.type for rel in mapper.relationships.values()))}")

    # Investigate key concepts
    key_concepts = [
        "concept_factory_pattern",
        "concept_feature_engineering",
        "concept_classification_pipeline"
    ]

    print(f"\n🔍 INVESTIGATING KEY CONCEPTS:")
    print("-" * 60)

    for concept_id in key_concepts:
        if concept_id in mapper.nodes:
            print(f"\n🎯 Investigating: {mapper.nodes[concept_id].name}")

            investigation = mapper.investigate_concept(concept_id, InvestigationLevel.DETAILED)

            print(f"   Direct relationships: {len(investigation['direct_relationships']['incoming']) + len(investigation['direct_relationships']['outgoing'])}")

            if investigation["insights"]:
                print(f"   Key insights:")
                for insight in investigation["insights"][:3]:
                    print(f"     • {insight}")

            if investigation["recommendations"]:
                print(f"   Recommendations:")
                for rec in investigation["recommendations"][:2]:
                    print(f"     • {rec}")

    # Generate mind map visualization
    print(f"\n🎨 GENERATING MIND MAP VISUALIZATION:")
    print("-" * 60)

    try:
        fig = mapper.create_interactive_plotly_mindmap()
        output_file = "ontology_mindmap.html"
        fig.write_html(output_file)
        print(f"✓ Interactive mind map saved: {output_file}")
    except Exception as e:
        print(f"⚠️  Visualization generation skipped: {e}")

    # Export investigation reports
    print(f"\n📋 EXPORTING INVESTIGATION REPORTS:")
    print("-" * 60)

    for concept_id in key_concepts[:2]:  # Export first 2 for demo
        if concept_id in mapper.nodes:
            try:
                report_file = mapper.export_investigation_report(concept_id)
                print(f"✓ Report exported: {report_file}")
            except Exception as e:
                print(f"⚠️  Report export failed for {concept_id}: {e}")

    # Demonstrate advanced investigation
    print(f"\n🔬 ADVANCED INVESTIGATION DEMONSTRATION:")
    print("-" * 60)

    factory_concept = "concept_factory_pattern"
    if factory_concept in mapper.nodes:
        investigation = mapper.investigate_concept(factory_concept, InvestigationLevel.COMPREHENSIVE)

        print(f"Concept: {investigation['concept'].name}")
        print(f"Network size: {len(investigation.get('extended_network', {}).get('nodes', []))}")

        if "reasoning_paths" in investigation:
            print(f"Reasoning paths found: {len(investigation['reasoning_paths'])}")
            for i, path in enumerate(investigation['reasoning_paths'][:3]):
                print(f"  Path {i+1}: {path.reasoning}")
                print(f"           Confidence: {path.confidence:.2f}")

    print(f"\n" + "=" * 80)
    print("🎯 ONTOLOGY INVESTIGATION COMPLETE")
    print(f"✓ Comprehensive knowledge graph built and analyzed")
    print(f"✓ Multi-level investigation framework demonstrated")
    print(f"✓ Automated reasoning and insight generation")
    print(f"✓ Interactive visualization and reporting")

    return mapper

if __name__ == "__main__":
    main()