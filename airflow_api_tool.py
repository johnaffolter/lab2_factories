#!/usr/bin/env python3

"""
Airflow API Tool
Complete API access to Airflow without browser
Includes DAG management, monitoring, and execution
"""

import subprocess
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import sys
# from tabulate import tabulate (optional dependency)
import os

class AirflowAPI:
    """Complete Airflow API implementation"""

    def __init__(self, container_name: str = "airflow-standalone"):
        self.container = container_name
        self._check_container()

    def _check_container(self) -> bool:
        """Check if Airflow container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            if self.container not in result.stdout:
                print(f"❌ Container '{self.container}' is not running")
                print("Run ./airflow_fixed_start.sh to start Airflow")
                return False
            return True
        except Exception as e:
            print(f"❌ Error checking container: {e}")
            return False

    def _exec_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Execute command in Airflow container"""
        full_cmd = ["docker", "exec", self.container] + cmd
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr or str(e)

    def list_dags(self, show_paused: bool = True) -> List[Dict]:
        """List all DAGs with details"""
        success, output = self._exec_command(["airflow", "dags", "list"])
        if not success:
            return []

        dags = []
        lines = output.strip().split('\n')

        for line in lines[2:]:  # Skip headers
            if line.strip() and not line.startswith('='):
                parts = line.split('|')
                if len(parts) >= 4:
                    dag_id = parts[0].strip()
                    if dag_id:
                        is_paused = parts[3].strip() == 'True' if len(parts) > 3 else False
                        if show_paused or not is_paused:
                            dags.append({
                                'dag_id': dag_id,
                                'filepath': parts[1].strip() if len(parts) > 1 else '',
                                'schedule': parts[2].strip() if len(parts) > 2 else '',
                                'paused': is_paused
                            })

        return dags

    def trigger_dag(self, dag_id: str, conf: Dict = None, run_id: str = None) -> Tuple[bool, str]:
        """Trigger a DAG run"""
        cmd = ["airflow", "dags", "trigger", dag_id]

        if run_id:
            cmd.extend(["--run-id", run_id])
        else:
            # Generate unique run ID
            run_id = f"api_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cmd.extend(["--run-id", run_id])

        if conf:
            cmd.extend(["--conf", json.dumps(conf)])

        success, output = self._exec_command(cmd)
        if success:
            return True, f"Triggered {dag_id} with run_id: {run_id}"
        return False, output

    def pause_dag(self, dag_id: str) -> bool:
        """Pause a DAG"""
        success, _ = self._exec_command(["airflow", "dags", "pause", dag_id])
        return success

    def unpause_dag(self, dag_id: str) -> bool:
        """Unpause a DAG"""
        success, _ = self._exec_command(["airflow", "dags", "unpause", dag_id])
        return success

    def get_dag_state(self, dag_id: str, execution_date: str = None) -> Optional[str]:
        """Get DAG run state"""
        if not execution_date:
            execution_date = datetime.now().strftime("%Y-%m-%d")

        success, output = self._exec_command(
            ["airflow", "dags", "state", dag_id, execution_date]
        )

        if success:
            return output.strip()
        return None

    def list_dag_runs(self, dag_id: str, limit: int = 10) -> List[Dict]:
        """List DAG runs"""
        cmd = ["airflow", "dags", "list-runs", "--dag-id", dag_id]
        success, output = self._exec_command(cmd)

        if not success:
            return []

        runs = []
        lines = output.strip().split('\n')

        for line in lines[2:limit+2]:  # Skip headers, limit results
            if line.strip() and '|' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    runs.append({
                        'dag_id': parts[0].strip(),
                        'run_id': parts[1].strip(),
                        'state': parts[2].strip(),
                        'execution_date': parts[3].strip(),
                        'start_date': parts[4].strip() if len(parts) > 4 else '',
                        'end_date': parts[5].strip() if len(parts) > 5 else ''
                    })

        return runs

    def list_tasks(self, dag_id: str) -> List[str]:
        """List tasks in a DAG"""
        success, output = self._exec_command(["airflow", "tasks", "list", dag_id])
        if success:
            return [line.strip() for line in output.strip().split('\n') if line.strip()]
        return []

    def test_task(self, dag_id: str, task_id: str, execution_date: str = None) -> Tuple[bool, str]:
        """Test a specific task"""
        if not execution_date:
            execution_date = datetime.now().strftime("%Y-%m-%d")

        cmd = ["airflow", "tasks", "test", dag_id, task_id, execution_date]
        return self._exec_command(cmd)

    def get_task_state(self, dag_id: str, task_id: str, execution_date: str = None) -> Optional[str]:
        """Get task state"""
        if not execution_date:
            execution_date = datetime.now().strftime("%Y-%m-%d")

        cmd = ["airflow", "tasks", "state", dag_id, task_id, execution_date]
        success, output = self._exec_command(cmd)

        if success:
            return output.strip()
        return None

    def clear_dag_runs(self, dag_id: str, start_date: str = None, end_date: str = None) -> bool:
        """Clear DAG runs"""
        cmd = ["airflow", "dags", "clear", dag_id, "-y"]

        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])

        success, _ = self._exec_command(cmd)
        return success

    def get_connections(self) -> List[Dict]:
        """List all connections"""
        success, output = self._exec_command(["airflow", "connections", "list"])
        if not success:
            return []

        connections = []
        lines = output.strip().split('\n')

        for line in lines[2:]:  # Skip headers
            if line.strip() and '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    connections.append({
                        'id': parts[0].strip(),
                        'conn_type': parts[1].strip(),
                        'host': parts[2].strip() if len(parts) > 2 else ''
                    })

        return connections

    def add_connection(self, conn_id: str, conn_type: str, host: str = None,
                       port: int = None, login: str = None, password: str = None) -> bool:
        """Add a new connection"""
        cmd = ["airflow", "connections", "add", conn_id, "--conn-type", conn_type]

        if host:
            cmd.extend(["--host", host])
        if port:
            cmd.extend(["--port", str(port)])
        if login:
            cmd.extend(["--login", login])
        if password:
            cmd.extend(["--password", password])

        success, _ = self._exec_command(cmd)
        return success

    def get_variables(self) -> List[str]:
        """List all variables"""
        success, output = self._exec_command(["airflow", "variables", "list"])
        if success:
            return [line.strip() for line in output.strip().split('\n') if line.strip()]
        return []

    def set_variable(self, key: str, value: str) -> bool:
        """Set a variable"""
        cmd = ["airflow", "variables", "set", key, value]
        success, _ = self._exec_command(cmd)
        return success

    def get_pools(self) -> List[Dict]:
        """List all pools"""
        success, output = self._exec_command(["airflow", "pools", "list"])
        if not success:
            return []

        pools = []
        lines = output.strip().split('\n')

        for line in lines[2:]:  # Skip headers
            if line.strip() and '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    pools.append({
                        'name': parts[0].strip(),
                        'slots': parts[1].strip(),
                        'used': parts[2].strip()
                    })

        return pools


def format_table(data: List[Dict], title: str = None) -> None:
    """Format and print data as table"""
    if not data:
        print("No data to display")
        return

    if title:
        print(f"\n{title}")
        print("=" * len(title))

    # Simple table formatting without tabulate
    if data:
        headers = list(data[0].keys())
        # Print headers
        header_line = " | ".join(f"{h:20}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        # Print rows
        for row in data:
            row_line = " | ".join(f"{str(v)[:20]:20}" for v in row.values())
            print(row_line)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Airflow API Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List DAGs
    list_parser = subparsers.add_parser("list", help="List all DAGs")
    list_parser.add_argument("--active", action="store_true", help="Show only active DAGs")

    # Trigger DAG
    trigger_parser = subparsers.add_parser("trigger", help="Trigger a DAG")
    trigger_parser.add_argument("dag_id", help="DAG ID to trigger")
    trigger_parser.add_argument("--conf", help="JSON configuration", default="{}")

    # Pause/Unpause DAG
    pause_parser = subparsers.add_parser("pause", help="Pause a DAG")
    pause_parser.add_argument("dag_id", help="DAG ID to pause")

    unpause_parser = subparsers.add_parser("unpause", help="Unpause a DAG")
    unpause_parser.add_argument("dag_id", help="DAG ID to unpause")

    # Get runs
    runs_parser = subparsers.add_parser("runs", help="List DAG runs")
    runs_parser.add_argument("dag_id", help="DAG ID")
    runs_parser.add_argument("--limit", type=int, default=10, help="Number of runs to show")

    # Get tasks
    tasks_parser = subparsers.add_parser("tasks", help="List tasks in a DAG")
    tasks_parser.add_argument("dag_id", help="DAG ID")

    # Test task
    test_parser = subparsers.add_parser("test", help="Test a task")
    test_parser.add_argument("dag_id", help="DAG ID")
    test_parser.add_argument("task_id", help="Task ID")

    # Clear runs
    clear_parser = subparsers.add_parser("clear", help="Clear DAG runs")
    clear_parser.add_argument("dag_id", help="DAG ID")

    # Connections
    conn_parser = subparsers.add_parser("connections", help="List connections")

    # Variables
    var_parser = subparsers.add_parser("variables", help="List variables")

    # Pools
    pool_parser = subparsers.add_parser("pools", help="List pools")

    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")

    args = parser.parse_args()

    # Initialize API
    api = AirflowAPI()

    if args.command == "list":
        dags = api.list_dags(show_paused=not args.active)
        format_table(dags, "Available DAGs")

    elif args.command == "trigger":
        conf = json.loads(args.conf)
        success, message = api.trigger_dag(args.dag_id, conf)
        print(f"{'✅' if success else '❌'} {message}")

    elif args.command == "pause":
        if api.pause_dag(args.dag_id):
            print(f"✅ Paused {args.dag_id}")
        else:
            print(f"❌ Failed to pause {args.dag_id}")

    elif args.command == "unpause":
        if api.unpause_dag(args.dag_id):
            print(f"✅ Unpaused {args.dag_id}")
        else:
            print(f"❌ Failed to unpause {args.dag_id}")

    elif args.command == "runs":
        runs = api.list_dag_runs(args.dag_id, args.limit)
        format_table(runs, f"Runs for {args.dag_id}")

    elif args.command == "tasks":
        tasks = api.list_tasks(args.dag_id)
        print(f"\nTasks in {args.dag_id}:")
        for task in tasks:
            print(f"  • {task}")

    elif args.command == "test":
        success, output = api.test_task(args.dag_id, args.task_id)
        print(output)

    elif args.command == "clear":
        if api.clear_dag_runs(args.dag_id):
            print(f"✅ Cleared runs for {args.dag_id}")
        else:
            print(f"❌ Failed to clear runs")

    elif args.command == "connections":
        connections = api.get_connections()
        format_table(connections, "Connections")

    elif args.command == "variables":
        variables = api.get_variables()
        print("\nVariables:")
        for var in variables:
            print(f"  • {var}")

    elif args.command == "pools":
        pools = api.get_pools()
        format_table(pools, "Pools")

    elif args.command == "interactive":
        interactive_mode(api)

    else:
        parser.print_help()


def interactive_mode(api: AirflowAPI):
    """Interactive menu mode"""
    while True:
        print("\n" + "=" * 50)
        print("AIRFLOW API TOOL - INTERACTIVE MODE")
        print("=" * 50)
        print("\n1. List DAGs")
        print("2. Trigger DAG")
        print("3. View DAG Runs")
        print("4. Pause/Unpause DAG")
        print("5. List Tasks")
        print("6. Test Task")
        print("7. View Connections")
        print("8. View Variables")
        print("9. View Pools")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            dags = api.list_dags()
            format_table(dags, "Available DAGs")

        elif choice == "2":
            dag_id = input("Enter DAG ID: ").strip()
            if dag_id:
                success, message = api.trigger_dag(dag_id)
                print(f"{'✅' if success else '❌'} {message}")

        elif choice == "3":
            dag_id = input("Enter DAG ID: ").strip()
            if dag_id:
                runs = api.list_dag_runs(dag_id)
                format_table(runs, f"Runs for {dag_id}")

        elif choice == "4":
            dag_id = input("Enter DAG ID: ").strip()
            action = input("Pause (p) or Unpause (u)? ").strip().lower()
            if dag_id and action:
                if action == 'p':
                    success = api.pause_dag(dag_id)
                    print(f"{'✅ Paused' if success else '❌ Failed'} {dag_id}")
                elif action == 'u':
                    success = api.unpause_dag(dag_id)
                    print(f"{'✅ Unpaused' if success else '❌ Failed'} {dag_id}")

        elif choice == "5":
            dag_id = input("Enter DAG ID: ").strip()
            if dag_id:
                tasks = api.list_tasks(dag_id)
                print(f"\nTasks in {dag_id}:")
                for task in tasks:
                    print(f"  • {task}")

        elif choice == "6":
            dag_id = input("Enter DAG ID: ").strip()
            task_id = input("Enter Task ID: ").strip()
            if dag_id and task_id:
                success, output = api.test_task(dag_id, task_id)
                print(output[:500])  # Limit output

        elif choice == "7":
            connections = api.get_connections()
            format_table(connections, "Connections")

        elif choice == "8":
            variables = api.get_variables()
            print("\nVariables:")
            for var in variables:
                print(f"  • {var}")

        elif choice == "9":
            pools = api.get_pools()
            format_table(pools, "Pools")

        elif choice == "0":
            print("\n👋 Goodbye!")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    print("🚀 AIRFLOW API TOOL")
    print("Complete API access without browser")
    print("-" * 40)

    if len(sys.argv) > 1:
        main()
    else:
        # Default to interactive mode if no arguments
        api = AirflowAPI()
        interactive_mode(api)