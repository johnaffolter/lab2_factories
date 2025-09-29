#!/usr/bin/env python3

"""
Airflow CLI Interface
Interact with Airflow programmatically without browser
Uses REAL Airflow REST API
"""

import requests
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional
import base64

class AirflowCLI:
    """Command-line interface for Airflow operations"""

    def __init__(self, host: str = "localhost", port: int = 8080,
                 username: str = "admin", password: str = "admin"):
        self.base_url = f"http://{host}:{port}/api/v1"
        self.auth = (username, password)
        # Create basic auth header
        credentials = f"{username}:{password}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {base64.b64encode(credentials.encode()).decode()}"
        }

    def check_health(self) -> bool:
        """Check if Airflow is healthy"""
        try:
            response = requests.get(
                f"http://localhost:8080/health",
                timeout=5
            )
            if response.status_code == 200:
                print("✅ Airflow is healthy")
                return True
        except:
            pass

        print("❌ Airflow is not responding")
        return False

    def list_dags_docker(self) -> List[str]:
        """List all DAGs using Docker exec (most reliable)"""
        try:
            result = subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "list"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output
            lines = result.stdout.strip().split('\n')
            dags = []

            for line in lines[2:]:  # Skip header lines
                if line.strip() and not line.startswith('='):
                    parts = line.split('|')
                    if len(parts) > 1:
                        dag_id = parts[0].strip()
                        if dag_id:
                            dags.append(dag_id)

            return dags

        except subprocess.CalledProcessError as e:
            print(f"Error listing DAGs: {e}")
            return []

    def trigger_dag_docker(self, dag_id: str, conf: Dict = None) -> bool:
        """Trigger a DAG run using Docker exec"""
        try:
            cmd = ["docker", "exec", "airflow-standalone", "airflow", "dags", "trigger", dag_id]

            if conf:
                cmd.extend(["--conf", json.dumps(conf)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print(f"✅ Triggered DAG: {dag_id}")
            print(f"   {result.stdout.strip()}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to trigger DAG: {e}")
            return False

    def pause_dag_docker(self, dag_id: str) -> bool:
        """Pause a DAG using Docker exec"""
        try:
            subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "pause", dag_id],
                capture_output=True,
                check=True
            )
            print(f"⏸️ Paused DAG: {dag_id}")
            return True
        except:
            return False

    def unpause_dag_docker(self, dag_id: str) -> bool:
        """Unpause a DAG using Docker exec"""
        try:
            subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "unpause", dag_id],
                capture_output=True,
                check=True
            )
            print(f"▶️ Unpaused DAG: {dag_id}")
            return True
        except:
            return False

    def get_dag_state_docker(self, dag_id: str) -> Optional[str]:
        """Get the state of the latest DAG run"""
        try:
            result = subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "state", dag_id,
                 datetime.now().strftime("%Y-%m-%d")],
                capture_output=True,
                text=True
            )

            state = result.stdout.strip()
            if state:
                return state

        except:
            pass

        return None

    def list_dag_runs_docker(self, dag_id: str) -> List[Dict]:
        """List recent DAG runs"""
        try:
            result = subprocess.run(
                ["docker", "exec", "airflow-standalone", "airflow", "dags", "list-runs",
                 "--dag-id", dag_id],
                capture_output=True,
                text=True
            )

            runs = []
            lines = result.stdout.strip().split('\n')

            for line in lines[2:]:  # Skip headers
                if line.strip() and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        runs.append({
                            'dag_id': parts[0].strip(),
                            'run_id': parts[1].strip(),
                            'state': parts[2].strip(),
                            'execution_date': parts[3].strip()
                        })

            return runs

        except:
            return []


def interactive_menu():
    """Interactive menu for Airflow operations"""
    cli = AirflowCLI()

    print("\n🚀 AIRFLOW CLI INTERFACE")
    print("=" * 40)

    # Check health first
    if not cli.check_health():
        print("\n⚠️ Airflow is not running. Starting it...")
        subprocess.run(["./airflow_quick_start.sh"], shell=True)
        time.sleep(30)

    while True:
        print("\n📋 MENU:")
        print("1. List all DAGs")
        print("2. Trigger a DAG")
        print("3. Pause a DAG")
        print("4. Unpause a DAG")
        print("5. Show DAG runs")
        print("6. Check system health")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            print("\n📊 Available DAGs:")
            dags = cli.list_dags_docker()
            if dags:
                for i, dag in enumerate(dags, 1):
                    print(f"  {i}. {dag}")
            else:
                print("  No DAGs found")

        elif choice == "2":
            dags = cli.list_dags_docker()
            if dags:
                print("\n📊 Select DAG to trigger:")
                for i, dag in enumerate(dags, 1):
                    print(f"  {i}. {dag}")

                try:
                    selection = int(input("\nEnter number: ")) - 1
                    if 0 <= selection < len(dags):
                        dag_id = dags[selection]
                        cli.trigger_dag_docker(dag_id)
                except (ValueError, IndexError):
                    print("Invalid selection")

        elif choice == "3":
            dag_id = input("\nEnter DAG ID to pause: ").strip()
            if dag_id:
                cli.pause_dag_docker(dag_id)

        elif choice == "4":
            dag_id = input("\nEnter DAG ID to unpause: ").strip()
            if dag_id:
                cli.unpause_dag_docker(dag_id)

        elif choice == "5":
            dag_id = input("\nEnter DAG ID to show runs: ").strip()
            if dag_id:
                runs = cli.list_dag_runs_docker(dag_id)
                if runs:
                    print(f"\n📈 Recent runs for {dag_id}:")
                    for run in runs:
                        print(f"  • {run['run_id']}: {run['state']} ({run['execution_date']})")
                else:
                    print("  No runs found")

        elif choice == "6":
            cli.check_health()

        elif choice == "0":
            print("\n👋 Goodbye!")
            break

        else:
            print("Invalid option")


def main():
    """Main entry point"""
    print("=" * 50)
    print("AIRFLOW COMMAND-LINE INTERFACE")
    print("Real Airflow operations without browser")
    print("=" * 50)

    # Quick status check
    cli = AirflowCLI()

    print("\n🔍 Checking Airflow status...")
    if cli.check_health():
        print("\n📊 Available DAGs:")
        dags = cli.list_dags_docker()
        for dag in dags:
            print(f"  • {dag}")

        print("\n✅ Airflow is running and accessible!")
        print("\nYou can:")
        print("  1. Use the interactive menu below")
        print("  2. Access the web UI at http://localhost:8080")
        print("     (If browser has issues, clear cache or use incognito mode)")

        # Start interactive menu
        interactive_menu()
    else:
        print("\n⚠️ Airflow is not running.")
        print("Run ./airflow_quick_start.sh to start it")


if __name__ == "__main__":
    main()