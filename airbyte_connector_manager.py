#!/usr/bin/env python3

"""
Airbyte Connector Manager
Manages Airbyte connections, sources, destinations, and sync jobs
Provides programmatic configuration of data pipelines
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SourceConfig:
    """Configuration for an Airbyte source"""
    name: str
    source_type: str
    connection_configuration: Dict[str, Any]
    workspace_id: str = None
    source_definition_id: str = None


@dataclass
class DestinationConfig:
    """Configuration for an Airbyte destination"""
    name: str
    destination_type: str
    connection_configuration: Dict[str, Any]
    workspace_id: str = None
    destination_definition_id: str = None


@dataclass
class ConnectionConfig:
    """Configuration for an Airbyte connection"""
    name: str
    source_id: str
    destination_id: str
    sync_frequency: str = "manual"  # manual, hourly, daily, weekly
    namespace_definition: str = "source"
    prefix: str = ""
    status: str = "active"


class AirbyteManager:
    """Manages Airbyte operations via API"""

    def __init__(self, host: str = "localhost", port: int = 8000,
                 username: str = "admin", password: str = "mlops2024"):
        self.base_url = f"http://{host}:{port}/api/v1"
        self.auth = (username, password) if username else None
        self.headers = {"Content-Type": "application/json"}
        self.workspace_id = None
        self._initialize()

    def _initialize(self):
        """Initialize connection and get workspace"""
        try:
            # Check health
            response = requests.get(f"{self.base_url}/health", auth=self.auth)
            if response.status_code == 200:
                logger.info("Connected to Airbyte API")
            else:
                logger.warning(f"Airbyte API returned status {response.status_code}")

            # Get default workspace
            workspaces = self.get_workspaces()
            if workspaces:
                self.workspace_id = workspaces[0]["workspaceId"]
                logger.info(f"Using workspace: {self.workspace_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Airbyte connection: {e}")

    def get_workspaces(self) -> List[Dict]:
        """Get all workspaces"""
        try:
            response = requests.post(
                f"{self.base_url}/workspaces/list",
                json={},
                headers=self.headers,
                auth=self.auth
            )
            if response.status_code == 200:
                return response.json().get("workspaces", [])
        except Exception as e:
            logger.error(f"Failed to get workspaces: {e}")
        return []

    def get_source_definitions(self) -> List[Dict]:
        """Get available source connector definitions"""
        try:
            response = requests.post(
                f"{self.base_url}/source_definitions/list_for_workspace",
                json={"workspaceId": self.workspace_id},
                headers=self.headers,
                auth=self.auth
            )
            if response.status_code == 200:
                return response.json().get("sourceDefinitions", [])
        except Exception as e:
            logger.error(f"Failed to get source definitions: {e}")
        return []

    def get_destination_definitions(self) -> List[Dict]:
        """Get available destination connector definitions"""
        try:
            response = requests.post(
                f"{self.base_url}/destination_definitions/list_for_workspace",
                json={"workspaceId": self.workspace_id},
                headers=self.headers,
                auth=self.auth
            )
            if response.status_code == 200:
                return response.json().get("destinationDefinitions", [])
        except Exception as e:
            logger.error(f"Failed to get destination definitions: {e}")
        return []

    def create_source(self, config: SourceConfig) -> Optional[str]:
        """Create a new source"""
        try:
            # Find source definition ID
            if not config.source_definition_id:
                definitions = self.get_source_definitions()
                for def_ in definitions:
                    if config.source_type.lower() in def_["name"].lower():
                        config.source_definition_id = def_["sourceDefinitionId"]
                        break

            if not config.source_definition_id:
                logger.error(f"Source type {config.source_type} not found")
                return None

            # Create source
            payload = {
                "sourceDefinitionId": config.source_definition_id,
                "connectionConfiguration": config.connection_configuration,
                "workspaceId": config.workspace_id or self.workspace_id,
                "name": config.name
            }

            response = requests.post(
                f"{self.base_url}/sources/create",
                json=payload,
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                source_id = response.json()["sourceId"]
                logger.info(f"Created source {config.name}: {source_id}")
                return source_id
            else:
                logger.error(f"Failed to create source: {response.text}")

        except Exception as e:
            logger.error(f"Error creating source: {e}")
        return None

    def create_destination(self, config: DestinationConfig) -> Optional[str]:
        """Create a new destination"""
        try:
            # Find destination definition ID
            if not config.destination_definition_id:
                definitions = self.get_destination_definitions()
                for def_ in definitions:
                    if config.destination_type.lower() in def_["name"].lower():
                        config.destination_definition_id = def_["destinationDefinitionId"]
                        break

            if not config.destination_definition_id:
                logger.error(f"Destination type {config.destination_type} not found")
                return None

            # Create destination
            payload = {
                "destinationDefinitionId": config.destination_definition_id,
                "connectionConfiguration": config.connection_configuration,
                "workspaceId": config.workspace_id or self.workspace_id,
                "name": config.name
            }

            response = requests.post(
                f"{self.base_url}/destinations/create",
                json=payload,
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                destination_id = response.json()["destinationId"]
                logger.info(f"Created destination {config.name}: {destination_id}")
                return destination_id
            else:
                logger.error(f"Failed to create destination: {response.text}")

        except Exception as e:
            logger.error(f"Error creating destination: {e}")
        return None

    def create_connection(self, config: ConnectionConfig) -> Optional[str]:
        """Create a connection between source and destination"""
        try:
            # Get source schema
            schema_response = requests.post(
                f"{self.base_url}/sources/discover_schema",
                json={"sourceId": config.source_id},
                headers=self.headers,
                auth=self.auth
            )

            if schema_response.status_code != 200:
                logger.error("Failed to discover source schema")
                return None

            catalog = schema_response.json()["catalog"]

            # Configure streams
            configured_catalog = {
                "streams": []
            }

            for stream in catalog.get("streams", []):
                configured_stream = {
                    "stream": stream["stream"],
                    "syncMode": "full_refresh",
                    "destinationSyncMode": "overwrite",
                    "primaryKey": [],
                    "cursorField": [],
                    "selected": True
                }
                configured_catalog["streams"].append(configured_stream)

            # Create connection
            schedule = None
            if config.sync_frequency != "manual":
                schedule = self._create_schedule(config.sync_frequency)

            payload = {
                "name": config.name,
                "sourceId": config.source_id,
                "destinationId": config.destination_id,
                "syncCatalog": configured_catalog,
                "status": config.status,
                "namespaceDefinition": config.namespace_definition,
                "prefix": config.prefix
            }

            if schedule:
                payload["schedule"] = schedule

            response = requests.post(
                f"{self.base_url}/connections/create",
                json=payload,
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                connection_id = response.json()["connectionId"]
                logger.info(f"Created connection {config.name}: {connection_id}")
                return connection_id
            else:
                logger.error(f"Failed to create connection: {response.text}")

        except Exception as e:
            logger.error(f"Error creating connection: {e}")
        return None

    def trigger_sync(self, connection_id: str) -> Optional[str]:
        """Manually trigger a sync job"""
        try:
            response = requests.post(
                f"{self.base_url}/connections/sync",
                json={"connectionId": connection_id},
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                job = response.json()["job"]
                job_id = job["id"]
                logger.info(f"Triggered sync job: {job_id}")
                return job_id
            else:
                logger.error(f"Failed to trigger sync: {response.text}")

        except Exception as e:
            logger.error(f"Error triggering sync: {e}")
        return None

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a sync job"""
        try:
            response = requests.post(
                f"{self.base_url}/jobs/get",
                json={"id": job_id},
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                return response.json()["job"]

        except Exception as e:
            logger.error(f"Error getting job status: {e}")
        return None

    def list_connections(self) -> List[Dict]:
        """List all connections"""
        try:
            response = requests.post(
                f"{self.base_url}/connections/list",
                json={"workspaceId": self.workspace_id},
                headers=self.headers,
                auth=self.auth
            )

            if response.status_code == 200:
                return response.json().get("connections", [])

        except Exception as e:
            logger.error(f"Error listing connections: {e}")
        return []

    def _create_schedule(self, frequency: str) -> Dict:
        """Create schedule configuration"""
        schedules = {
            "hourly": {"units": 1, "timeUnit": "hours"},
            "daily": {"units": 24, "timeUnit": "hours"},
            "weekly": {"units": 168, "timeUnit": "hours"}
        }
        return schedules.get(frequency, {"units": 24, "timeUnit": "hours"})


class AirbyteConnectorFactory:
    """Factory for creating common Airbyte connectors"""

    @staticmethod
    def create_s3_source(bucket: str, access_key: str, secret_key: str,
                        path_pattern: str = "**/*.csv") -> SourceConfig:
        """Create S3 source configuration"""
        return SourceConfig(
            name=f"S3-{bucket}",
            source_type="S3",
            connection_configuration={
                "dataset": bucket,
                "path_pattern": path_pattern,
                "provider": {
                    "bucket": bucket,
                    "aws_access_key_id": access_key,
                    "aws_secret_access_key": secret_key,
                    "region_name": "us-west-2"
                },
                "format": {
                    "filetype": "csv",
                    "delimiter": ",",
                    "quote_char": '"',
                    "escape_char": "\\",
                    "encoding": "utf-8"
                }
            }
        )

    @staticmethod
    def create_postgres_source(host: str, port: int, database: str,
                              username: str, password: str) -> SourceConfig:
        """Create PostgreSQL source configuration"""
        return SourceConfig(
            name=f"PostgreSQL-{database}",
            source_type="Postgres",
            connection_configuration={
                "host": host,
                "port": port,
                "database": database,
                "username": username,
                "password": password,
                "ssl": False,
                "replication_method": {
                    "method": "Standard"
                }
            }
        )

    @staticmethod
    def create_api_source(api_url: str, api_key: str = None) -> SourceConfig:
        """Create REST API source configuration"""
        config = {
            "url_base": api_url,
            "authenticator": {
                "type": "api_key" if api_key else "no_auth"
            }
        }

        if api_key:
            config["authenticator"]["api_key"] = api_key

        return SourceConfig(
            name=f"API-{api_url.split('//')[1].split('/')[0]}",
            source_type="REST API",
            connection_configuration=config
        )

    @staticmethod
    def create_snowflake_destination(account: str, warehouse: str,
                                    database: str, schema: str,
                                    username: str, password: str) -> DestinationConfig:
        """Create Snowflake destination configuration"""
        return DestinationConfig(
            name=f"Snowflake-{database}",
            destination_type="Snowflake",
            connection_configuration={
                "host": f"{account}.snowflakecomputing.com",
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "username": username,
                "password": password,
                "loading_method": {
                    "method": "Internal Staging"
                }
            }
        )

    @staticmethod
    def create_s3_destination(bucket: str, access_key: str, secret_key: str,
                             path_pattern: str = "airbyte/${SOURCE_NAMESPACE}/${STREAM_NAME}/") -> DestinationConfig:
        """Create S3 destination configuration"""
        return DestinationConfig(
            name=f"S3-{bucket}",
            destination_type="S3",
            connection_configuration={
                "s3_bucket_name": bucket,
                "s3_bucket_path": path_pattern,
                "s3_bucket_region": "us-west-2",
                "access_key_id": access_key,
                "secret_access_key": secret_key,
                "format": {
                    "format_type": "CSV",
                    "compression": {
                        "compression_type": "GZIP"
                    }
                }
            }
        )


def setup_example_pipelines():
    """Set up example data pipelines"""
    print("🔄 Setting up Airbyte Data Pipelines")
    print("=" * 40)

    # Initialize manager
    manager = AirbyteManager()

    if not manager.workspace_id:
        print("❌ Could not connect to Airbyte. Is it running?")
        print("   Run ./airbyte_setup.sh to start Airbyte")
        return

    # Example 1: S3 to Snowflake pipeline
    print("\n📦 Creating S3 → Snowflake pipeline...")

    # Create S3 source
    s3_source = AirbyteConnectorFactory.create_s3_source(
        bucket="mlops-data",
        access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        path_pattern="data/*.csv"
    )

    source_id = manager.create_source(s3_source)

    if source_id:
        print(f"   ✅ Created S3 source: {source_id}")

        # Create Snowflake destination
        snowflake_dest = AirbyteConnectorFactory.create_snowflake_destination(
            account=os.getenv("SNOWFLAKE_ACCOUNT", "demo"),
            warehouse="COMPUTE_WH",
            database="MLOPS_DB",
            schema="PUBLIC",
            username=os.getenv("SNOWFLAKE_USER", "demo_user"),
            password=os.getenv("SNOWFLAKE_PASSWORD", "password")
        )

        dest_id = manager.create_destination(snowflake_dest)

        if dest_id:
            print(f"   ✅ Created Snowflake destination: {dest_id}")

            # Create connection
            connection = ConnectionConfig(
                name="S3-to-Snowflake-Sync",
                source_id=source_id,
                destination_id=dest_id,
                sync_frequency="daily"
            )

            conn_id = manager.create_connection(connection)

            if conn_id:
                print(f"   ✅ Created connection: {conn_id}")

                # Trigger initial sync
                job_id = manager.trigger_sync(conn_id)
                if job_id:
                    print(f"   ✅ Started sync job: {job_id}")

    # List all connections
    print("\n📊 Active Connections:")
    connections = manager.list_connections()
    for conn in connections:
        print(f"   • {conn['name']} ({conn['status']})")

    print("\n✅ Airbyte setup complete!")
    print(f"   Access UI at: http://localhost:8000")


def main():
    """Main entry point"""
    setup_example_pipelines()


if __name__ == "__main__":
    main()