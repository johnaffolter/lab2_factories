#!/usr/bin/env python3

"""
Deployment Orchestrator for MLOps Platform
Manages deployment, scaling, and lifecycle of all services
Supports Docker, Kubernetes, and AWS deployments with real configurations
"""

import os
import sys
import json
import yaml
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
import threading

# Docker SDK
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Kubernetes client
try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# AWS SDK
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    image: str
    ports: Dict[str, int]
    environment: Dict[str, str]
    volumes: List[str] = None
    depends_on: List[str] = None
    replicas: int = 1
    resources: Dict[str, Any] = None
    health_check: Dict[str, Any] = None


@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    service: str
    status: str  # pending, deploying, running, failed, stopped
    platform: str  # docker, kubernetes, aws
    timestamp: datetime
    message: str
    metadata: Dict[str, Any] = None


class DockerDeployment:
    """Manages Docker deployments"""

    def __init__(self):
        if not DOCKER_AVAILABLE:
            self.client = None
            logger.warning("Docker SDK not available")
        else:
            try:
                self.client = docker.from_env()
                logger.info("Connected to Docker daemon")
            except Exception as e:
                self.client = None
                logger.error(f"Failed to connect to Docker: {e}")

        self.network_name = "mlops_network"
        self.containers = {}

    def create_network(self):
        """Create Docker network for services"""
        if not self.client:
            return False

        try:
            networks = self.client.networks.list(names=[self.network_name])
            if not networks:
                network = self.client.networks.create(
                    self.network_name,
                    driver="bridge"
                )
                logger.info(f"Created Docker network: {self.network_name}")
            else:
                logger.info(f"Docker network already exists: {self.network_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create network: {e}")
            return False

    def deploy_service(self, config: ServiceConfig) -> DeploymentStatus:
        """Deploy a service to Docker"""
        if not self.client:
            return DeploymentStatus(
                service=config.name,
                status="failed",
                platform="docker",
                timestamp=datetime.now(),
                message="Docker not available"
            )

        try:
            # Pull image if needed
            logger.info(f"Pulling image {config.image}...")
            self.client.images.pull(config.image)

            # Prepare container configuration
            container_config = {
                "name": f"mlops_{config.name}",
                "image": config.image,
                "environment": config.environment,
                "ports": config.ports,
                "network": self.network_name,
                "detach": True,
                "restart_policy": {"Name": "unless-stopped"}
            }

            if config.volumes:
                container_config["volumes"] = config.volumes

            # Stop existing container if running
            self.stop_service(config.name)

            # Run container
            container = self.client.containers.run(**container_config)
            self.containers[config.name] = container

            logger.info(f"Deployed {config.name} to Docker")
            return DeploymentStatus(
                service=config.name,
                status="running",
                platform="docker",
                timestamp=datetime.now(),
                message=f"Container ID: {container.short_id}",
                metadata={"container_id": container.short_id}
            )

        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return DeploymentStatus(
                service=config.name,
                status="failed",
                platform="docker",
                timestamp=datetime.now(),
                message=str(e)
            )

    def stop_service(self, service_name: str) -> bool:
        """Stop a Docker service"""
        if not self.client:
            return False

        try:
            container_name = f"mlops_{service_name}"
            containers = self.client.containers.list(all=True, filters={"name": container_name})

            for container in containers:
                logger.info(f"Stopping container {container.name}...")
                container.stop()
                container.remove()
                logger.info(f"Removed container {container.name}")

            return True
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            return False

    def get_service_logs(self, service_name: str, lines: int = 100) -> str:
        """Get logs from a Docker service"""
        if not self.client:
            return ""

        try:
            container_name = f"mlops_{service_name}"
            container = self.client.containers.get(container_name)
            logs = container.logs(tail=lines).decode('utf-8')
            return logs
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return ""


class KubernetesDeployment:
    """Manages Kubernetes deployments"""

    def __init__(self):
        if not KUBERNETES_AVAILABLE:
            self.api = None
            logger.warning("Kubernetes client not available")
        else:
            try:
                # Try to load in-cluster config first, then local config
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()

                self.v1 = client.CoreV1Api()
                self.apps_v1 = client.AppsV1Api()
                self.namespace = "mlops"
                logger.info("Connected to Kubernetes cluster")
            except Exception as e:
                self.v1 = None
                self.apps_v1 = None
                logger.error(f"Failed to connect to Kubernetes: {e}")

    def create_namespace(self):
        """Create Kubernetes namespace"""
        if not self.v1:
            return False

        try:
            namespaces = self.v1.list_namespace()
            if not any(ns.metadata.name == self.namespace for ns in namespaces.items):
                namespace = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                self.v1.create_namespace(namespace)
                logger.info(f"Created namespace: {self.namespace}")
            else:
                logger.info(f"Namespace already exists: {self.namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to create namespace: {e}")
            return False

    def deploy_service(self, config: ServiceConfig) -> DeploymentStatus:
        """Deploy a service to Kubernetes"""
        if not self.apps_v1:
            return DeploymentStatus(
                service=config.name,
                status="failed",
                platform="kubernetes",
                timestamp=datetime.now(),
                message="Kubernetes not available"
            )

        try:
            # Create deployment
            deployment = self._create_deployment_spec(config)
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )

            # Create service
            service = self._create_service_spec(config)
            self.v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )

            logger.info(f"Deployed {config.name} to Kubernetes")
            return DeploymentStatus(
                service=config.name,
                status="running",
                platform="kubernetes",
                timestamp=datetime.now(),
                message=f"Deployed to namespace {self.namespace}",
                metadata={"namespace": self.namespace}
            )

        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return DeploymentStatus(
                service=config.name,
                status="failed",
                platform="kubernetes",
                timestamp=datetime.now(),
                message=str(e)
            )

    def _create_deployment_spec(self, config: ServiceConfig):
        """Create Kubernetes deployment specification"""
        # Container spec
        container = client.V1Container(
            name=config.name,
            image=config.image,
            ports=[
                client.V1ContainerPort(container_port=port)
                for port in config.ports.values()
            ],
            env=[
                client.V1EnvVar(name=k, value=v)
                for k, v in config.environment.items()
            ]
        )

        # Add resource limits if specified
        if config.resources:
            container.resources = client.V1ResourceRequirements(
                limits=config.resources.get("limits", {}),
                requests=config.resources.get("requests", {})
            )

        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": config.name}),
            spec=client.V1PodSpec(containers=[container])
        )

        # Deployment spec
        spec = client.V1DeploymentSpec(
            replicas=config.replicas,
            selector=client.V1LabelSelector(
                match_labels={"app": config.name}
            ),
            template=template
        )

        # Deployment
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(name=config.name),
            spec=spec
        )

        return deployment

    def _create_service_spec(self, config: ServiceConfig):
        """Create Kubernetes service specification"""
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(name=config.name),
            spec=client.V1ServiceSpec(
                selector={"app": config.name},
                ports=[
                    client.V1ServicePort(
                        name=name,
                        port=port,
                        target_port=port
                    )
                    for name, port in config.ports.items()
                ],
                type="LoadBalancer"
            )
        )
        return service


class AWSDeployment:
    """Manages AWS deployments"""

    def __init__(self):
        if not AWS_AVAILABLE:
            self.ecs_client = None
            self.ec2_client = None
            logger.warning("AWS SDK not available")
        else:
            try:
                self.ecs_client = boto3.client('ecs', region_name='us-west-2')
                self.ec2_client = boto3.client('ec2', region_name='us-west-2')
                self.ecr_client = boto3.client('ecr', region_name='us-west-2')
                self.cluster_name = "mlops-cluster"
                logger.info("Connected to AWS")
            except Exception as e:
                self.ecs_client = None
                logger.error(f"Failed to connect to AWS: {e}")

    def create_cluster(self):
        """Create ECS cluster"""
        if not self.ecs_client:
            return False

        try:
            clusters = self.ecs_client.list_clusters()
            cluster_arns = clusters.get('clusterArns', [])

            if not any(self.cluster_name in arn for arn in cluster_arns):
                response = self.ecs_client.create_cluster(
                    clusterName=self.cluster_name,
                    capacityProviders=['FARGATE', 'FARGATE_SPOT']
                )
                logger.info(f"Created ECS cluster: {self.cluster_name}")
            else:
                logger.info(f"ECS cluster already exists: {self.cluster_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create cluster: {e}")
            return False

    def deploy_service(self, config: ServiceConfig) -> DeploymentStatus:
        """Deploy a service to AWS ECS"""
        if not self.ecs_client:
            return DeploymentStatus(
                service=config.name,
                status="failed",
                platform="aws",
                timestamp=datetime.now(),
                message="AWS not available"
            )

        try:
            # Register task definition
            task_def = self._create_task_definition(config)
            self.ecs_client.register_task_definition(**task_def)

            # Create or update service
            service_name = f"mlops-{config.name}"
            try:
                # Try to update existing service
                self.ecs_client.update_service(
                    cluster=self.cluster_name,
                    service=service_name,
                    taskDefinition=config.name,
                    desiredCount=config.replicas
                )
                logger.info(f"Updated ECS service: {service_name}")
            except:
                # Create new service
                self.ecs_client.create_service(
                    cluster=self.cluster_name,
                    serviceName=service_name,
                    taskDefinition=config.name,
                    desiredCount=config.replicas,
                    launchType='FARGATE',
                    networkConfiguration={
                        'awsvpcConfiguration': {
                            'subnets': self._get_subnets(),
                            'assignPublicIp': 'ENABLED'
                        }
                    }
                )
                logger.info(f"Created ECS service: {service_name}")

            return DeploymentStatus(
                service=config.name,
                status="running",
                platform="aws",
                timestamp=datetime.now(),
                message=f"Deployed to ECS cluster {self.cluster_name}",
                metadata={"cluster": self.cluster_name}
            )

        except Exception as e:
            logger.error(f"AWS deployment failed: {e}")
            return DeploymentStatus(
                service=config.name,
                status="failed",
                platform="aws",
                timestamp=datetime.now(),
                message=str(e)
            )

    def _create_task_definition(self, config: ServiceConfig):
        """Create ECS task definition"""
        return {
            'family': config.name,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '256',
            'memory': '512',
            'containerDefinitions': [
                {
                    'name': config.name,
                    'image': config.image,
                    'portMappings': [
                        {'containerPort': port, 'protocol': 'tcp'}
                        for port in config.ports.values()
                    ],
                    'environment': [
                        {'name': k, 'value': v}
                        for k, v in config.environment.items()
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f'/ecs/{config.name}',
                            'awslogs-region': 'us-west-2',
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
        }

    def _get_subnets(self) -> List[str]:
        """Get default VPC subnets"""
        try:
            response = self.ec2_client.describe_subnets(
                Filters=[{'Name': 'default-for-az', 'Values': ['true']}]
            )
            return [subnet['SubnetId'] for subnet in response['Subnets']][:2]
        except:
            return []


class DeploymentOrchestrator:
    """Main orchestrator for all deployments"""

    def __init__(self):
        self.docker = DockerDeployment()
        self.kubernetes = KubernetesDeployment()
        self.aws = AWSDeployment()
        self.deployments: List[DeploymentStatus] = []
        self.services = self._load_service_configs()

    def _load_service_configs(self) -> Dict[str, ServiceConfig]:
        """Load service configurations"""
        configs = {
            "monitoring": ServiceConfig(
                name="monitoring",
                image="prom/prometheus:latest",
                ports={"http": 9090},
                environment={"CONFIG": "/etc/prometheus/prometheus.yml"},
                volumes=["/tmp/prometheus:/prometheus"]
            ),
            "neo4j": ServiceConfig(
                name="neo4j",
                image="neo4j:latest",
                ports={"bolt": 7687, "http": 7474},
                environment={
                    "NEO4J_AUTH": "neo4j/password",
                    "NEO4J_ACCEPT_LICENSE_AGREEMENT": "yes"
                },
                volumes=["/tmp/neo4j:/data"]
            ),
            "postgres": ServiceConfig(
                name="postgres",
                image="postgres:14",
                ports={"postgres": 5432},
                environment={
                    "POSTGRES_USER": "admin",
                    "POSTGRES_PASSWORD": "password",
                    "POSTGRES_DB": "mlops"
                },
                volumes=["/tmp/postgres:/var/lib/postgresql/data"]
            ),
            "redis": ServiceConfig(
                name="redis",
                image="redis:alpine",
                ports={"redis": 6379},
                environment={}
            ),
            "airflow": ServiceConfig(
                name="airflow",
                image="apache/airflow:2.5.0",
                ports={"webserver": 8080},
                environment={
                    "AIRFLOW__CORE__EXECUTOR": "LocalExecutor",
                    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": "postgresql+psycopg2://admin:password@postgres/mlops"
                },
                depends_on=["postgres"]
            )
        }
        return configs

    def deploy_all(self, platform: str = "docker", services: List[str] = None) -> List[DeploymentStatus]:
        """Deploy all or specified services to a platform"""
        if services is None:
            services = list(self.services.keys())

        results = []

        # Initialize platform
        if platform == "docker":
            self.docker.create_network()
        elif platform == "kubernetes":
            self.kubernetes.create_namespace()
        elif platform == "aws":
            self.aws.create_cluster()

        # Deploy services
        for service_name in services:
            if service_name not in self.services:
                logger.warning(f"Service {service_name} not found")
                continue

            config = self.services[service_name]

            # Check dependencies
            if config.depends_on:
                for dep in config.depends_on:
                    if dep not in [d.service for d in results if d.status == "running"]:
                        logger.info(f"Deploying dependency {dep} first...")
                        dep_result = self._deploy_service(dep, platform)
                        results.append(dep_result)

            # Deploy service
            result = self._deploy_service(service_name, platform)
            results.append(result)
            self.deployments.append(result)

        return results

    def _deploy_service(self, service_name: str, platform: str) -> DeploymentStatus:
        """Deploy a single service"""
        config = self.services.get(service_name)
        if not config:
            return DeploymentStatus(
                service=service_name,
                status="failed",
                platform=platform,
                timestamp=datetime.now(),
                message="Service configuration not found"
            )

        logger.info(f"Deploying {service_name} to {platform}...")

        if platform == "docker":
            return self.docker.deploy_service(config)
        elif platform == "kubernetes":
            return self.kubernetes.deploy_service(config)
        elif platform == "aws":
            return self.aws.deploy_service(config)
        else:
            return DeploymentStatus(
                service=service_name,
                status="failed",
                platform=platform,
                timestamp=datetime.now(),
                message=f"Unknown platform: {platform}"
            )

    def stop_all(self, platform: str = "docker", services: List[str] = None) -> bool:
        """Stop all or specified services"""
        if services is None:
            services = list(self.services.keys())

        success = True
        for service_name in services:
            if platform == "docker":
                if not self.docker.stop_service(service_name):
                    success = False
            # Add Kubernetes and AWS stop implementations

        return success

    def get_status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return {
            "total_deployments": len(self.deployments),
            "running": len([d for d in self.deployments if d.status == "running"]),
            "failed": len([d for d in self.deployments if d.status == "failed"]),
            "deployments": [asdict(d) for d in self.deployments[-10:]]  # Last 10
        }

    def health_check(self, platform: str = "docker") -> Dict[str, bool]:
        """Check health of deployed services"""
        health = {}

        if platform == "docker" and self.docker.client:
            for service_name in self.services.keys():
                try:
                    container_name = f"mlops_{service_name}"
                    container = self.docker.client.containers.get(container_name)
                    health[service_name] = container.status == "running"
                except:
                    health[service_name] = False

        return health

    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml file"""
        compose = {
            "version": "3.8",
            "services": {},
            "networks": {
                "mlops_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {}
        }

        for name, config in self.services.items():
            service = {
                "image": config.image,
                "container_name": f"mlops_{name}",
                "environment": config.environment,
                "ports": [f"{port}:{port}" for port in config.ports.values()],
                "networks": ["mlops_network"],
                "restart": "unless-stopped"
            }

            if config.volumes:
                service["volumes"] = config.volumes

            if config.depends_on:
                service["depends_on"] = config.depends_on

            compose["services"][name] = service

        return yaml.dump(compose, default_flow_style=False)

    def save_docker_compose(self, filepath: str = "docker-compose.yml"):
        """Save docker-compose.yml file"""
        compose_content = self.generate_docker_compose()
        with open(filepath, 'w') as f:
            f.write(compose_content)
        logger.info(f"Saved docker-compose.yml to {filepath}")


def main():
    """Demo deployment orchestrator"""
    print("🚀 DEPLOYMENT ORCHESTRATOR")
    print("=" * 50)

    orchestrator = DeploymentOrchestrator()

    # Generate docker-compose file
    print("\n📄 Generating docker-compose.yml...")
    orchestrator.save_docker_compose("/tmp/mlops_docker_compose.yml")
    print("   Saved to /tmp/mlops_docker_compose.yml")

    # Deploy to Docker
    print("\n🐳 Deploying services to Docker...")
    results = orchestrator.deploy_all(
        platform="docker",
        services=["redis", "postgres", "neo4j"]
    )

    for result in results:
        icon = "✅" if result.status == "running" else "❌"
        print(f"   {icon} {result.service}: {result.status} - {result.message}")

    # Check health
    print("\n🏥 Health Check:")
    health = orchestrator.health_check(platform="docker")
    for service, healthy in health.items():
        status = "Healthy" if healthy else "Unhealthy"
        print(f"   {service}: {status}")

    # Get status
    status = orchestrator.get_status()
    print(f"\n📊 Deployment Status:")
    print(f"   Total: {status['total_deployments']}")
    print(f"   Running: {status['running']}")
    print(f"   Failed: {status['failed']}")

    print("\n✅ Deployment orchestration complete")


if __name__ == "__main__":
    main()