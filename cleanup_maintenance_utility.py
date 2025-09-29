#!/usr/bin/env python3

"""
Cleanup and Maintenance Utility for MLOps Platform
Manages temporary files, logs, containers, and system resources
Provides automated cleanup, backup, and maintenance operations
"""

import os
import sys
import json
import shutil
import time
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import tarfile
import zipfile

# Docker support
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# AWS support for S3 backups
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanupUtility:
    """Handles cleanup of various system resources"""

    def __init__(self):
        self.temp_dirs = [
            "/tmp",
            "/var/tmp",
            Path.home() / ".cache",
            Path.cwd() / "__pycache__"
        ]
        self.log_dirs = [
            "/var/log",
            Path.home() / "logs",
            Path.cwd() / "logs"
        ]
        self.cleanup_stats = {
            "files_deleted": 0,
            "space_freed_mb": 0,
            "containers_removed": 0,
            "images_removed": 0,
            "volumes_removed": 0
        }

    def clean_temp_files(self, older_than_days: int = 7) -> Dict[str, Any]:
        """Clean temporary files older than specified days"""
        stats = {"files_deleted": 0, "space_freed": 0}
        cutoff_time = datetime.now() - timedelta(days=older_than_days)

        for temp_dir in self.temp_dirs:
            if not Path(temp_dir).exists():
                continue

            logger.info(f"Cleaning temp files in {temp_dir}")

            for root, dirs, files in os.walk(temp_dir):
                # Skip important directories
                if any(skip in root for skip in ['.git', 'node_modules', 'venv', '.env']):
                    continue

                for file in files:
                    filepath = Path(root) / file

                    try:
                        # Check file age
                        if filepath.stat().st_mtime < cutoff_time.timestamp():
                            # Check if it's safe to delete
                            if self._is_safe_to_delete(filepath):
                                size = filepath.stat().st_size
                                filepath.unlink()
                                stats["files_deleted"] += 1
                                stats["space_freed"] += size
                                self.cleanup_stats["files_deleted"] += 1
                                self.cleanup_stats["space_freed_mb"] += size / (1024 * 1024)
                    except Exception as e:
                        logger.debug(f"Could not delete {filepath}: {e}")

        logger.info(f"Deleted {stats['files_deleted']} files, freed {stats['space_freed'] / (1024*1024):.2f} MB")
        return stats

    def clean_log_files(self, compress_older_than_days: int = 30,
                        delete_older_than_days: int = 90) -> Dict[str, Any]:
        """Clean and compress log files"""
        stats = {"compressed": 0, "deleted": 0, "space_saved": 0}
        compress_cutoff = datetime.now() - timedelta(days=compress_older_than_days)
        delete_cutoff = datetime.now() - timedelta(days=delete_older_than_days)

        for log_dir in self.log_dirs:
            if not Path(log_dir).exists():
                continue

            logger.info(f"Processing log files in {log_dir}")

            for file in Path(log_dir).glob("**/*.log"):
                try:
                    file_time = datetime.fromtimestamp(file.stat().st_mtime)

                    if file_time < delete_cutoff:
                        # Delete very old logs
                        size = file.stat().st_size
                        file.unlink()
                        stats["deleted"] += 1
                        stats["space_saved"] += size
                    elif file_time < compress_cutoff and not file.suffix == ".gz":
                        # Compress moderately old logs
                        self._compress_file(file)
                        stats["compressed"] += 1
                        stats["space_saved"] += file.stat().st_size * 0.7  # Estimate

                except Exception as e:
                    logger.debug(f"Error processing log {file}: {e}")

        logger.info(f"Compressed {stats['compressed']} logs, deleted {stats['deleted']} logs")
        return stats

    def clean_docker_resources(self) -> Dict[str, Any]:
        """Clean Docker resources"""
        if not DOCKER_AVAILABLE:
            logger.warning("Docker not available")
            return {}

        stats = {"containers": 0, "images": 0, "volumes": 0, "networks": 0}

        try:
            client = docker.from_env()

            # Remove stopped containers
            containers = client.containers.list(all=True, filters={"status": "exited"})
            for container in containers:
                try:
                    container.remove()
                    stats["containers"] += 1
                    self.cleanup_stats["containers_removed"] += 1
                except Exception as e:
                    logger.debug(f"Could not remove container {container.name}: {e}")

            # Remove dangling images
            images = client.images.list(filters={"dangling": True})
            for image in images:
                try:
                    client.images.remove(image.id)
                    stats["images"] += 1
                    self.cleanup_stats["images_removed"] += 1
                except Exception as e:
                    logger.debug(f"Could not remove image {image.id}: {e}")

            # Remove unused volumes
            volumes = client.volumes.list(filters={"dangling": True})
            for volume in volumes:
                try:
                    volume.remove()
                    stats["volumes"] += 1
                    self.cleanup_stats["volumes_removed"] += 1
                except Exception as e:
                    logger.debug(f"Could not remove volume {volume.name}: {e}")

            # Remove unused networks (except default ones)
            networks = client.networks.list()
            default_networks = ["bridge", "host", "none"]
            for network in networks:
                if network.name not in default_networks and not network.containers:
                    try:
                        network.remove()
                        stats["networks"] += 1
                    except Exception as e:
                        logger.debug(f"Could not remove network {network.name}: {e}")

            logger.info(f"Docker cleanup: {stats['containers']} containers, "
                       f"{stats['images']} images, {stats['volumes']} volumes removed")

        except Exception as e:
            logger.error(f"Docker cleanup failed: {e}")

        return stats

    def clean_python_cache(self) -> Dict[str, int]:
        """Clean Python cache files"""
        stats = {"pycache_dirs": 0, "pyc_files": 0}

        # Clean __pycache__ directories
        for root, dirs, files in os.walk(Path.cwd()):
            if "__pycache__" in dirs:
                cache_dir = Path(root) / "__pycache__"
                try:
                    shutil.rmtree(cache_dir)
                    stats["pycache_dirs"] += 1
                except Exception as e:
                    logger.debug(f"Could not remove {cache_dir}: {e}")

            # Clean .pyc files
            for file in files:
                if file.endswith('.pyc'):
                    try:
                        (Path(root) / file).unlink()
                        stats["pyc_files"] += 1
                    except Exception as e:
                        logger.debug(f"Could not remove {file}: {e}")

        logger.info(f"Cleaned {stats['pycache_dirs']} __pycache__ dirs and {stats['pyc_files']} .pyc files")
        return stats

    def clean_npm_cache(self) -> bool:
        """Clean npm cache"""
        try:
            result = subprocess.run(["npm", "cache", "clean", "--force"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Cleaned npm cache")
                return True
        except Exception as e:
            logger.debug(f"Could not clean npm cache: {e}")
        return False

    def clean_pip_cache(self) -> bool:
        """Clean pip cache"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "cache", "purge"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Cleaned pip cache")
                return True
        except Exception as e:
            logger.debug(f"Could not clean pip cache: {e}")
        return False

    def _is_safe_to_delete(self, filepath: Path) -> bool:
        """Check if a file is safe to delete"""
        unsafe_patterns = [
            ".env", ".git", "config", "credentials",
            ".pem", ".key", ".crt", ".ssh"
        ]
        return not any(pattern in str(filepath).lower() for pattern in unsafe_patterns)

    def _compress_file(self, filepath: Path):
        """Compress a file using gzip"""
        import gzip

        with open(filepath, 'rb') as f_in:
            with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        filepath.unlink()

    def get_cleanup_summary(self) -> Dict[str, Any]:
        """Get summary of cleanup operations"""
        return self.cleanup_stats


class BackupUtility:
    """Handles backup operations"""

    def __init__(self):
        self.backup_dir = Path.home() / "mlops_backups"
        self.backup_dir.mkdir(exist_ok=True)

        if AWS_AVAILABLE:
            self.s3_client = boto3.client('s3')
            self.s3_bucket = "mlops-backups"
        else:
            self.s3_client = None

    def backup_databases(self) -> Dict[str, str]:
        """Backup all databases"""
        backups = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Backup PostgreSQL
        postgres_backup = self._backup_postgres(timestamp)
        if postgres_backup:
            backups["postgres"] = postgres_backup

        # Backup Neo4j
        neo4j_backup = self._backup_neo4j(timestamp)
        if neo4j_backup:
            backups["neo4j"] = neo4j_backup

        # Backup Redis
        redis_backup = self._backup_redis(timestamp)
        if redis_backup:
            backups["redis"] = redis_backup

        logger.info(f"Created {len(backups)} database backups")
        return backups

    def backup_configs(self) -> str:
        """Backup configuration files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"configs_{timestamp}.tar.gz"

        config_files = [
            "*.yml", "*.yaml", "*.json", "*.env.example",
            "docker-compose.yml", "requirements.txt", "package.json"
        ]

        with tarfile.open(backup_file, "w:gz") as tar:
            for pattern in config_files:
                for file in Path.cwd().glob(pattern):
                    if file.exists():
                        tar.add(file, arcname=file.name)

        logger.info(f"Created config backup: {backup_file}")
        return str(backup_file)

    def backup_logs(self) -> str:
        """Backup log files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"logs_{timestamp}.tar.gz"

        with tarfile.open(backup_file, "w:gz") as tar:
            for log_dir in [Path.cwd() / "logs", Path("/tmp")]:
                if log_dir.exists():
                    for log_file in log_dir.glob("**/*.log"):
                        tar.add(log_file, arcname=f"logs/{log_file.name}")

        logger.info(f"Created log backup: {backup_file}")
        return str(backup_file)

    def upload_to_s3(self, local_file: str, s3_key: str = None) -> bool:
        """Upload backup to S3"""
        if not self.s3_client:
            logger.warning("S3 client not available")
            return False

        if not s3_key:
            s3_key = f"backups/{Path(local_file).name}"

        try:
            self.s3_client.upload_file(local_file, self.s3_bucket, s3_key)
            logger.info(f"Uploaded {local_file} to s3://{self.s3_bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False

    def _backup_postgres(self, timestamp: str) -> Optional[str]:
        """Backup PostgreSQL database"""
        backup_file = self.backup_dir / f"postgres_{timestamp}.sql"

        try:
            result = subprocess.run([
                "pg_dump",
                "-h", "localhost",
                "-U", "admin",
                "-d", "mlops",
                "-f", str(backup_file)
            ], capture_output=True, text=True, env={**os.environ, "PGPASSWORD": "password"})

            if result.returncode == 0:
                logger.info(f"PostgreSQL backup created: {backup_file}")
                return str(backup_file)
        except Exception as e:
            logger.debug(f"PostgreSQL backup failed: {e}")
        return None

    def _backup_neo4j(self, timestamp: str) -> Optional[str]:
        """Backup Neo4j database"""
        backup_file = self.backup_dir / f"neo4j_{timestamp}.dump"

        try:
            # Use neo4j-admin dump command
            result = subprocess.run([
                "neo4j-admin", "dump",
                "--database=neo4j",
                f"--to={backup_file}"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Neo4j backup created: {backup_file}")
                return str(backup_file)
        except Exception as e:
            logger.debug(f"Neo4j backup failed: {e}")
        return None

    def _backup_redis(self, timestamp: str) -> Optional[str]:
        """Backup Redis database"""
        backup_file = self.backup_dir / f"redis_{timestamp}.rdb"

        try:
            # Trigger Redis BGSAVE
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.bgsave()
            time.sleep(2)  # Wait for save to complete

            # Copy dump file
            redis_dump = Path("/var/lib/redis/dump.rdb")
            if redis_dump.exists():
                shutil.copy(redis_dump, backup_file)
                logger.info(f"Redis backup created: {backup_file}")
                return str(backup_file)
        except Exception as e:
            logger.debug(f"Redis backup failed: {e}")
        return None

    def cleanup_old_backups(self, keep_days: int = 30):
        """Remove backups older than specified days"""
        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0

        for backup_file in self.backup_dir.glob("*"):
            if backup_file.stat().st_mtime < cutoff.timestamp():
                backup_file.unlink()
                removed += 1

        logger.info(f"Removed {removed} old backups")


class MaintenanceScheduler:
    """Schedules and runs maintenance tasks"""

    def __init__(self):
        self.cleanup = CleanupUtility()
        self.backup = BackupUtility()
        self.tasks = []

    def run_daily_maintenance(self) -> Dict[str, Any]:
        """Run daily maintenance tasks"""
        logger.info("Starting daily maintenance...")
        results = {}

        # Clean temp files older than 3 days
        results["temp_cleanup"] = self.cleanup.clean_temp_files(older_than_days=3)

        # Clean Python cache
        results["python_cache"] = self.cleanup.clean_python_cache()

        # Docker cleanup
        results["docker"] = self.cleanup.clean_docker_resources()

        # Backup configs
        results["config_backup"] = self.backup.backup_configs()

        logger.info("Daily maintenance completed")
        return results

    def run_weekly_maintenance(self) -> Dict[str, Any]:
        """Run weekly maintenance tasks"""
        logger.info("Starting weekly maintenance...")
        results = {}

        # Deep clean temp files
        results["temp_cleanup"] = self.cleanup.clean_temp_files(older_than_days=7)

        # Compress old logs
        results["log_cleanup"] = self.cleanup.clean_log_files(
            compress_older_than_days=7,
            delete_older_than_days=30
        )

        # Clean package caches
        results["npm_cache"] = self.cleanup.clean_npm_cache()
        results["pip_cache"] = self.cleanup.clean_pip_cache()

        # Backup databases
        results["db_backups"] = self.backup.backup_databases()

        # Backup logs
        results["log_backup"] = self.backup.backup_logs()

        # Upload backups to S3
        if self.backup.s3_client:
            for backup_type, backup_file in results.get("db_backups", {}).items():
                self.backup.upload_to_s3(backup_file)

        logger.info("Weekly maintenance completed")
        return results

    def run_monthly_maintenance(self) -> Dict[str, Any]:
        """Run monthly maintenance tasks"""
        logger.info("Starting monthly maintenance...")
        results = {}

        # Deep clean everything
        results["deep_clean"] = {
            "temp": self.cleanup.clean_temp_files(older_than_days=30),
            "logs": self.cleanup.clean_log_files(
                compress_older_than_days=30,
                delete_older_than_days=90
            ),
            "docker": self.cleanup.clean_docker_resources()
        }

        # Clean old backups
        self.backup.cleanup_old_backups(keep_days=30)

        # System optimization
        results["optimization"] = self._optimize_system()

        logger.info("Monthly maintenance completed")
        return results

    def _optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance"""
        optimizations = {}

        # Clear system caches (Linux)
        if sys.platform == "linux":
            try:
                subprocess.run(["sync"], check=True)
                subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"],
                             shell=True, check=True)
                optimizations["cache_cleared"] = True
            except:
                optimizations["cache_cleared"] = False

        # Defragment databases (if applicable)
        # This would include database-specific optimization commands

        return optimizations

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
            "process_count": len(psutil.pids()),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }


def main():
    """Run maintenance operations"""
    print("🧹 CLEANUP & MAINTENANCE UTILITY")
    print("=" * 50)

    scheduler = MaintenanceScheduler()

    # Show system health
    print("\n📊 System Health:")
    health = scheduler.get_system_health()
    for metric, value in health.items():
        print(f"   {metric}: {value}")

    # Run daily maintenance
    print("\n🔧 Running Daily Maintenance...")
    daily_results = scheduler.run_daily_maintenance()

    # Show cleanup summary
    cleanup_summary = scheduler.cleanup.get_cleanup_summary()
    print("\n📈 Cleanup Summary:")
    print(f"   Files deleted: {cleanup_summary['files_deleted']}")
    print(f"   Space freed: {cleanup_summary['space_freed_mb']:.2f} MB")
    print(f"   Containers removed: {cleanup_summary['containers_removed']}")
    print(f"   Images removed: {cleanup_summary['images_removed']}")
    print(f"   Volumes removed: {cleanup_summary['volumes_removed']}")

    # Export maintenance report
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_health": health,
        "maintenance_results": daily_results,
        "cleanup_summary": cleanup_summary
    }

    report_file = f"/tmp/maintenance_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Report saved to: {report_file}")
    print("\n✅ Maintenance completed successfully")


if __name__ == "__main__":
    main()