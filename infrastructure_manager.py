#!/usr/bin/env python3
"""
Infrastructure Manager with Factory Pattern
Configures and manages AWS (S3, ECR, EC2), Neo4j Knowledge Graphs, and system integrations
"""

import os
import json
import boto3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from neo4j import GraphDatabase
import requests

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BASE_API_URL = "http://localhost:8000"

@dataclass
class InfrastructureConfig:
    """Configuration for infrastructure component"""
    name: str
    type: str
    region: str
    tags: Dict[str, str]
    parameters: Dict[str, Any]
    status: str = "not_created"
    created_at: Optional[str] = None
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None

class InfrastructureProvider(ABC):
    """Abstract base for infrastructure providers (Factory Pattern)"""

    @abstractmethod
    def create_resource(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """Create infrastructure resource"""
        pass

    @abstractmethod
    def get_resource_status(self, resource_id: str) -> str:
        """Get resource status"""
        pass

    @abstractmethod
    def delete_resource(self, resource_id: str) -> bool:
        """Delete resource"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass

class S3Provider(InfrastructureProvider):
    """AWS S3 bucket management"""

    def __init__(self):
        try:
            self.s3_client = boto3.client('s3', region_name=AWS_REGION)
            self.available = True
        except Exception as e:
            print(f"AWS credentials not configured: {e}")
            self.available = False

    def create_resource(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """Create S3 bucket"""

        if not self.available:
            return {
                "status": "simulated",
                "bucket_name": config.parameters.get("bucket_name"),
                "message": "AWS not configured - simulated creation"
            }

        bucket_name = config.parameters.get("bucket_name")

        try:
            # Create bucket
            if AWS_REGION == "us-east-1":
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                )

            # Add tags
            if config.tags:
                self.s3_client.put_bucket_tagging(
                    Bucket=bucket_name,
                    Tagging={'TagSet': [{'Key': k, 'Value': v} for k, v in config.tags.items()]}
                )

            # Enable versioning if requested
            if config.parameters.get("versioning", False):
                self.s3_client.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )

            return {
                "status": "created",
                "bucket_name": bucket_name,
                "region": AWS_REGION,
                "endpoint": f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com"
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_resource_status(self, resource_id: str) -> str:
        """Check if bucket exists"""
        if not self.available:
            return "simulated"

        try:
            self.s3_client.head_bucket(Bucket=resource_id)
            return "active"
        except:
            return "not_found"

    def delete_resource(self, resource_id: str) -> bool:
        """Delete S3 bucket"""
        if not self.available:
            return True

        try:
            # Empty bucket first
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=resource_id):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    self.s3_client.delete_objects(Bucket=resource_id, Delete={'Objects': objects})

            # Delete bucket
            self.s3_client.delete_bucket(Bucket=resource_id)
            return True
        except Exception as e:
            print(f"Error deleting bucket: {e}")
            return False

    def get_provider_name(self) -> str:
        return "AWS S3"

class EC2Provider(InfrastructureProvider):
    """AWS EC2 instance management"""

    def __init__(self):
        try:
            self.ec2_client = boto3.client('ec2', region_name=AWS_REGION)
            self.available = True
        except Exception as e:
            print(f"AWS credentials not configured: {e}")
            self.available = False

    def create_resource(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """Create EC2 instance"""

        if not self.available:
            return {
                "status": "simulated",
                "instance_id": "i-simulated-12345",
                "message": "AWS not configured - simulated creation"
            }

        try:
            response = self.ec2_client.run_instances(
                ImageId=config.parameters.get("ami_id", "ami-0c55b159cbfafe1f0"),
                InstanceType=config.parameters.get("instance_type", "t2.micro"),
                MinCount=1,
                MaxCount=1,
                KeyName=config.parameters.get("key_name"),
                SecurityGroupIds=config.parameters.get("security_groups", []),
                SubnetId=config.parameters.get("subnet_id"),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': k, 'Value': v} for k, v in config.tags.items()]
                }],
                UserData=config.parameters.get("user_data", "")
            )

            instance_id = response['Instances'][0]['InstanceId']

            return {
                "status": "creating",
                "instance_id": instance_id,
                "region": AWS_REGION
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_resource_status(self, resource_id: str) -> str:
        """Get EC2 instance status"""
        if not self.available:
            return "simulated"

        try:
            response = self.ec2_client.describe_instances(InstanceIds=[resource_id])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            return state
        except:
            return "not_found"

    def delete_resource(self, resource_id: str) -> bool:
        """Terminate EC2 instance"""
        if not self.available:
            return True

        try:
            self.ec2_client.terminate_instances(InstanceIds=[resource_id])
            return True
        except Exception as e:
            print(f"Error terminating instance: {e}")
            return False

    def get_provider_name(self) -> str:
        return "AWS EC2"

class ECRProvider(InfrastructureProvider):
    """AWS ECR repository management"""

    def __init__(self):
        try:
            self.ecr_client = boto3.client('ecr', region_name=AWS_REGION)
            self.available = True
        except Exception as e:
            print(f"AWS credentials not configured: {e}")
            self.available = False

    def create_resource(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """Create ECR repository"""

        if not self.available:
            return {
                "status": "simulated",
                "repository_name": config.parameters.get("repository_name"),
                "message": "AWS not configured - simulated creation"
            }

        repository_name = config.parameters.get("repository_name")

        try:
            response = self.ecr_client.create_repository(
                repositoryName=repository_name,
                imageScanningConfiguration={
                    'scanOnPush': config.parameters.get("scan_on_push", True)
                },
                imageTagMutability=config.parameters.get("tag_mutability", "MUTABLE"),
                tags=[{'Key': k, 'Value': v} for k, v in config.tags.items()]
            )

            repository_uri = response['repository']['repositoryUri']

            return {
                "status": "created",
                "repository_name": repository_name,
                "repository_uri": repository_uri,
                "registry_id": response['repository']['registryId']
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_resource_status(self, resource_id: str) -> str:
        """Check if repository exists"""
        if not self.available:
            return "simulated"

        try:
            self.ecr_client.describe_repositories(repositoryNames=[resource_id])
            return "active"
        except:
            return "not_found"

    def delete_resource(self, resource_id: str) -> bool:
        """Delete ECR repository"""
        if not self.available:
            return True

        try:
            self.ecr_client.delete_repository(
                repositoryName=resource_id,
                force=True
            )
            return True
        except Exception as e:
            print(f"Error deleting repository: {e}")
            return False

    def get_provider_name(self) -> str:
        return "AWS ECR"

class Neo4jProvider(InfrastructureProvider):
    """Neo4j Knowledge Graph management"""

    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            self.available = True
        except Exception as e:
            print(f"Neo4j not available: {e}")
            self.available = False

    def create_resource(self, config: InfrastructureConfig) -> Dict[str, Any]:
        """Create Neo4j database/graph"""

        if not self.available:
            return {
                "status": "simulated",
                "graph_name": config.parameters.get("graph_name"),
                "message": "Neo4j not configured - simulated creation"
            }

        graph_name = config.parameters.get("graph_name", "homework_graph")

        try:
            with self.driver.session() as session:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT email_id IF NOT EXISTS
                    FOR (e:Email) REQUIRE e.id IS UNIQUE
                """)

                session.run("""
                    CREATE CONSTRAINT topic_name IF NOT EXISTS
                    FOR (t:Topic) REQUIRE t.name IS UNIQUE
                """)

                # Create indexes
                session.run("""
                    CREATE INDEX email_subject IF NOT EXISTS
                    FOR (e:Email) ON (e.subject)
                """)

                session.run("""
                    CREATE INDEX topic_confidence IF NOT EXISTS
                    FOR ()-[r:CLASSIFIED_AS]->() ON (r.confidence)
                """)

            return {
                "status": "created",
                "graph_name": graph_name,
                "uri": NEO4J_URI,
                "constraints_created": 2,
                "indexes_created": 2
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_resource_status(self, resource_id: str) -> str:
        """Check Neo4j connection"""
        if not self.available:
            return "simulated"

        try:
            self.driver.verify_connectivity()
            return "active"
        except:
            return "disconnected"

    def delete_resource(self, resource_id: str) -> bool:
        """Clear graph data"""
        if not self.available:
            return True

        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            return True
        except Exception as e:
            print(f"Error clearing graph: {e}")
            return False

    def get_provider_name(self) -> str:
        return "Neo4j Knowledge Graph"

    def load_classification_data(self) -> Dict[str, Any]:
        """Load classification data from API into graph"""

        if not self.available:
            return {"status": "simulated", "message": "Neo4j not available"}

        try:
            # Get topics from API
            topics_response = requests.get(f"{BASE_API_URL}/topics")
            topics = topics_response.json().get("topics", [])

            # Get emails from file
            with open('data/emails.json', 'r') as f:
                emails = json.load(f)

            with self.driver.session() as session:
                # Create topic nodes
                for topic in topics:
                    session.run("""
                        MERGE (t:Topic {name: $name})
                        SET t.created_at = datetime()
                    """, name=topic)

                # Create email nodes and relationships
                for email in emails:
                    # Create email node
                    session.run("""
                        MERGE (e:Email {id: $id})
                        SET e.subject = $subject,
                            e.body = $body,
                            e.ground_truth = $ground_truth,
                            e.created_at = datetime()
                    """,
                        id=email['id'],
                        subject=email['subject'],
                        body=email['body'],
                        ground_truth=email.get('ground_truth')
                    )

                    # Classify email and create relationships
                    classify_response = requests.post(
                        f"{BASE_API_URL}/emails/classify",
                        json={
                            "subject": email['subject'],
                            "body": email['body'],
                            "use_email_similarity": False
                        }
                    )

                    if classify_response.status_code == 200:
                        result = classify_response.json()
                        predicted_topic = result.get('predicted_topic')
                        topic_scores = result.get('topic_scores', {})

                        # Create classification relationship
                        if predicted_topic:
                            confidence = topic_scores.get(predicted_topic, 0.0)
                            session.run("""
                                MATCH (e:Email {id: $email_id})
                                MATCH (t:Topic {name: $topic})
                                MERGE (e)-[r:CLASSIFIED_AS]->(t)
                                SET r.confidence = $confidence,
                                    r.classified_at = datetime()
                            """,
                                email_id=email['id'],
                                topic=predicted_topic,
                                confidence=confidence
                            )

                        # Create similarity relationships for top topics
                        top_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        for topic, score in top_topics:
                            if score > 0.5:  # Only if reasonably similar
                                session.run("""
                                    MATCH (e:Email {id: $email_id})
                                    MATCH (t:Topic {name: $topic})
                                    MERGE (e)-[r:SIMILAR_TO]->(t)
                                    SET r.similarity = $score
                                """,
                                    email_id=email['id'],
                                    topic=topic,
                                    score=score
                                )

            # Get statistics
            with self.driver.session() as session:
                topic_count = session.run("MATCH (t:Topic) RETURN count(t) as count").single()['count']
                email_count = session.run("MATCH (e:Email) RETURN count(e) as count").single()['count']
                relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']

            return {
                "status": "loaded",
                "topics": topic_count,
                "emails": email_count,
                "relationships": relationship_count
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def search_graph(self, query_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""

        if not self.available:
            return [{"status": "simulated", "message": "Neo4j not available"}]

        queries = {
            "find_email_by_topic": """
                MATCH (e:Email)-[r:CLASSIFIED_AS]->(t:Topic {name: $topic})
                RETURN e.id, e.subject, e.body, r.confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
            """,
            "find_similar_emails": """
                MATCH (e1:Email {id: $email_id})-[:CLASSIFIED_AS]->(t:Topic)<-[:CLASSIFIED_AS]-(e2:Email)
                WHERE e1 <> e2
                RETURN e2.id, e2.subject, e2.body, t.name as shared_topic
                LIMIT $limit
            """,
            "topic_statistics": """
                MATCH (t:Topic)<-[r:CLASSIFIED_AS]-(e:Email)
                WITH t.name as topic, count(e) as email_count, avg(r.confidence) as avg_confidence
                RETURN topic, email_count, avg_confidence
                ORDER BY email_count DESC
            """,
            "high_confidence_classifications": """
                MATCH (e:Email)-[r:CLASSIFIED_AS]->(t:Topic)
                WHERE r.confidence > $threshold
                RETURN e.subject, t.name, r.confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
            """,
            "email_relationships": """
                MATCH (e:Email {id: $email_id})-[r]->(t:Topic)
                RETURN type(r) as relationship, t.name as topic,
                       CASE type(r)
                           WHEN 'CLASSIFIED_AS' THEN r.confidence
                           WHEN 'SIMILAR_TO' THEN r.similarity
                           ELSE null
                       END as score
                ORDER BY score DESC
            """
        }

        if query_type not in queries:
            return [{"error": f"Unknown query type: {query_type}"}]

        try:
            with self.driver.session() as session:
                result = session.run(queries[query_type], **parameters)
                return [dict(record) for record in result]

        except Exception as e:
            return [{"error": str(e)}]

class InfrastructureFactory:
    """Factory for creating infrastructure providers"""

    @staticmethod
    def create_provider(provider_type: str) -> InfrastructureProvider:
        """Create infrastructure provider"""

        providers = {
            "s3": S3Provider,
            "ec2": EC2Provider,
            "ecr": ECRProvider,
            "neo4j": Neo4jProvider,
        }

        if provider_type not in providers:
            raise ValueError(f"Unknown provider type: {provider_type}")

        return providers[provider_type]()

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available providers"""
        return ["s3", "ec2", "ecr", "neo4j"]

class InfrastructureOrchestrator:
    """Orchestrate infrastructure across multiple providers"""

    def __init__(self):
        self.configurations = []
        self.resources = {}

    def add_preset_configs(self):
        """Add preset infrastructure configurations"""

        presets = [
            InfrastructureConfig(
                name="homework-data-bucket",
                type="s3",
                region=AWS_REGION,
                tags={"Project": "MLOps-Homework", "Environment": "Dev"},
                parameters={
                    "bucket_name": f"mlops-homework-data-{datetime.now().strftime('%Y%m%d')}",
                    "versioning": True
                }
            ),
            InfrastructureConfig(
                name="homework-model-repo",
                type="ecr",
                region=AWS_REGION,
                tags={"Project": "MLOps-Homework", "Type": "Models"},
                parameters={
                    "repository_name": "mlops-homework-models",
                    "scan_on_push": True,
                    "tag_mutability": "MUTABLE"
                }
            ),
            InfrastructureConfig(
                name="homework-inference-instance",
                type="ec2",
                region=AWS_REGION,
                tags={"Project": "MLOps-Homework", "Role": "Inference"},
                parameters={
                    "instance_type": "t2.micro",
                    "ami_id": "ami-0c55b159cbfafe1f0"
                }
            ),
            InfrastructureConfig(
                name="homework-knowledge-graph",
                type="neo4j",
                region="local",
                tags={"Project": "MLOps-Homework", "Type": "Database"},
                parameters={
                    "graph_name": "homework_classification_graph"
                }
            )
        ]

        self.configurations.extend(presets)
        print(f"Added {len(presets)} preset configurations")

    def deploy_infrastructure(self) -> Dict[str, Any]:
        """Deploy all configured infrastructure"""

        print("\n" + "="*80)
        print("INFRASTRUCTURE DEPLOYMENT")
        print("="*80)

        results = {
            "deployed": [],
            "failed": [],
            "simulated": []
        }

        for config in self.configurations:
            print(f"\nDeploying: {config.name} ({config.type})")

            try:
                provider = InfrastructureFactory.create_provider(config.type)
                result = provider.create_resource(config)

                if result.get("status") == "created":
                    config.status = "active"
                    config.created_at = datetime.now().isoformat()
                    config.resource_id = result.get("repository_name") or result.get("bucket_name") or result.get("instance_id")
                    config.endpoint = result.get("endpoint") or result.get("repository_uri")
                    results["deployed"].append({
                        "name": config.name,
                        "type": config.type,
                        "result": result
                    })
                    print(f"  Status: DEPLOYED")

                elif result.get("status") == "simulated":
                    config.status = "simulated"
                    results["simulated"].append({
                        "name": config.name,
                        "type": config.type,
                        "result": result
                    })
                    print(f"  Status: SIMULATED (credentials not configured)")

                else:
                    config.status = "failed"
                    results["failed"].append({
                        "name": config.name,
                        "type": config.type,
                        "error": result.get("error")
                    })
                    print(f"  Status: FAILED - {result.get('error')}")

                self.resources[config.name] = config

            except Exception as e:
                print(f"  Status: ERROR - {e}")
                results["failed"].append({
                    "name": config.name,
                    "type": config.type,
                    "error": str(e)
                })

        print("\n" + "="*80)
        print("DEPLOYMENT SUMMARY")
        print("="*80)
        print(f"Deployed: {len(results['deployed'])}")
        print(f"Simulated: {len(results['simulated'])}")
        print(f"Failed: {len(results['failed'])}")

        return results

    def load_data_to_graph(self) -> Dict[str, Any]:
        """Load classification data into Neo4j"""

        print("\n" + "="*80)
        print("LOADING DATA TO KNOWLEDGE GRAPH")
        print("="*80)

        neo4j_provider = InfrastructureFactory.create_provider("neo4j")
        result = neo4j_provider.load_classification_data()

        if result.get("status") == "loaded":
            print(f"\nSuccessfully loaded:")
            print(f"  Topics: {result['topics']}")
            print(f"  Emails: {result['emails']}")
            print(f"  Relationships: {result['relationships']}")

        return result

    def search_knowledge_graph(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""

        neo4j_provider = InfrastructureFactory.create_provider("neo4j")
        return neo4j_provider.search_graph(query_type, kwargs)

    def save_configuration(self, filename: str = "infrastructure_config.json"):
        """Save infrastructure configuration"""

        data = {
            "deployment_info": {
                "timestamp": datetime.now().isoformat(),
                "total_resources": len(self.resources),
                "active_resources": sum(1 for r in self.resources.values() if r.status == "active")
            },
            "resources": {name: asdict(config) for name, config in self.resources.items()}
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nConfiguration saved to: {filename}")

def main():
    """Main demonstration"""

    print("\n" + "="*80)
    print("INFRASTRUCTURE ORCHESTRATION SYSTEM")
    print("Factory Pattern for Multi-Cloud Management")
    print("="*80)

    # Create orchestrator
    orchestrator = InfrastructureOrchestrator()

    # Add preset configurations
    print("\n[1/4] Adding preset configurations...")
    orchestrator.add_preset_configs()

    # Deploy infrastructure
    print("\n[2/4] Deploying infrastructure...")
    deployment_results = orchestrator.deploy_infrastructure()

    # Load data to graph
    print("\n[3/4] Loading classification data to graph...")
    graph_results = orchestrator.load_data_to_graph()

    # Search examples
    print("\n[4/4] Demonstrating graph search...")

    # Example 1: Find emails by topic
    print("\nQuery 1: Emails classified as 'work'")
    work_emails = orchestrator.search_knowledge_graph(
        "find_email_by_topic",
        topic="work",
        limit=5
    )
    for email in work_emails[:3]:
        if "error" not in email:
            print(f"  - {email.get('e.subject', 'N/A')} (confidence: {email.get('r.confidence', 0):.3f})")

    # Example 2: Topic statistics
    print("\nQuery 2: Topic statistics")
    stats = orchestrator.search_knowledge_graph("topic_statistics", limit=10)
    for stat in stats[:5]:
        if "error" not in stat:
            print(f"  - {stat.get('topic', 'N/A')}: {stat.get('email_count', 0)} emails (avg conf: {stat.get('avg_confidence', 0):.3f})")

    # Example 3: High confidence classifications
    print("\nQuery 3: High confidence classifications (>0.8)")
    high_conf = orchestrator.search_knowledge_graph(
        "high_confidence_classifications",
        threshold=0.8,
        limit=5
    )
    for item in high_conf[:3]:
        if "error" not in item:
            print(f"  - {item.get('e.subject', 'N/A')} -> {item.get('t.name', 'N/A')} ({item.get('r.confidence', 0):.3f})")

    # Save configuration
    orchestrator.save_configuration()

    print("\n" + "="*80)
    print("ORCHESTRATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()