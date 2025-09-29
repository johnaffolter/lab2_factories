"""
Monitoring and Maintenance DAG
Automated system monitoring, health checks, and maintenance tasks
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup

default_args = {
    'owner': 'mlops_admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['admin@mlops.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'monitoring_maintenance',
    default_args=default_args,
    description='System monitoring and automated maintenance',
    schedule_interval='*/30 * * * *',  # Run every 30 minutes
    catchup=False,
    tags=['monitoring', 'maintenance', 'health'],
)

def check_system_health(**context):
    """Check overall system health"""
    import random
    import json

    # Simulate health checks
    health_metrics = {
        'cpu_usage': random.uniform(20, 80),
        'memory_usage': random.uniform(40, 90),
        'disk_usage': random.uniform(30, 85),
        'active_connections': random.randint(10, 100),
        'error_rate': random.uniform(0, 5),
        'timestamp': datetime.now().isoformat()
    }

    # Determine health status
    if health_metrics['cpu_usage'] > 90 or health_metrics['memory_usage'] > 90:
        health_status = 'critical'
    elif health_metrics['cpu_usage'] > 75 or health_metrics['memory_usage'] > 80:
        health_status = 'warning'
    else:
        health_status = 'healthy'

    health_metrics['status'] = health_status

    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='health_metrics', value=json.dumps(health_metrics))
    context['task_instance'].xcom_push(key='health_status', value=health_status)

    print(f"System health: {health_status}")
    print(f"Metrics: {health_metrics}")

    return health_metrics

def decide_maintenance_action(**context):
    """Decide which maintenance action to take based on health status"""
    health_status = context['task_instance'].xcom_pull(key='health_status')

    if health_status == 'critical':
        return 'emergency_maintenance'
    elif health_status == 'warning':
        return 'preventive_maintenance'
    else:
        return 'routine_check'

def perform_emergency_maintenance(**context):
    """Perform emergency maintenance for critical issues"""
    import json

    health_metrics = json.loads(context['task_instance'].xcom_pull(key='health_metrics'))

    print("⚠️ PERFORMING EMERGENCY MAINTENANCE")
    print(f"Critical metrics: {health_metrics}")

    # Simulate emergency actions
    actions = []

    if health_metrics['cpu_usage'] > 90:
        actions.append('Killing high CPU processes')
        actions.append('Clearing process cache')

    if health_metrics['memory_usage'] > 90:
        actions.append('Clearing memory cache')
        actions.append('Restarting memory-intensive services')

    for action in actions:
        print(f"  - {action}")

    return {
        'maintenance_type': 'emergency',
        'actions_taken': actions,
        'timestamp': datetime.now().isoformat()
    }

def perform_preventive_maintenance(**context):
    """Perform preventive maintenance for warnings"""
    import json

    health_metrics = json.loads(context['task_instance'].xcom_pull(key='health_metrics'))

    print("🔧 Performing preventive maintenance")

    actions = [
        'Cleaning temporary files',
        'Optimizing database queries',
        'Rotating log files',
        'Updating cache'
    ]

    for action in actions:
        print(f"  - {action}")

    return {
        'maintenance_type': 'preventive',
        'actions_taken': actions,
        'timestamp': datetime.now().isoformat()
    }

def perform_routine_check(**context):
    """Perform routine system check"""
    print("✅ Performing routine system check")

    checks = [
        'Verifying service status',
        'Checking disk space',
        'Validating configurations',
        'Testing connectivity'
    ]

    for check in checks:
        print(f"  - {check}")

    return {
        'maintenance_type': 'routine',
        'checks_performed': checks,
        'timestamp': datetime.now().isoformat()
    }

def check_docker_containers(**context):
    """Check Docker container health"""
    print("🐳 Checking Docker containers...")

    # Simulate container check
    containers = {
        'mlops_postgres': 'running',
        'mlops_neo4j': 'running',
        'mlops_redis': 'running',
        'mlops_airflow': 'running'
    }

    unhealthy = [name for name, status in containers.items() if status != 'running']

    if unhealthy:
        print(f"⚠️ Unhealthy containers: {unhealthy}")
        context['task_instance'].xcom_push(key='unhealthy_containers', value=unhealthy)
    else:
        print("✅ All containers healthy")

    return containers

def check_database_connections(**context):
    """Check database connectivity"""
    print("🗄️ Checking database connections...")

    # Simulate database checks
    databases = {
        'postgresql': {'status': 'connected', 'latency_ms': 12},
        'neo4j': {'status': 'connected', 'latency_ms': 8},
        'redis': {'status': 'connected', 'latency_ms': 2}
    }

    for db, info in databases.items():
        print(f"  {db}: {info['status']} (latency: {info['latency_ms']}ms)")

    return databases

def cleanup_old_data(**context):
    """Clean up old data and logs"""
    import random

    print("🧹 Cleaning up old data...")

    # Simulate cleanup
    cleanup_stats = {
        'logs_deleted': random.randint(50, 200),
        'temp_files_removed': random.randint(100, 500),
        'space_freed_mb': random.randint(100, 1000),
        'cache_cleared': True
    }

    print(f"  Logs deleted: {cleanup_stats['logs_deleted']}")
    print(f"  Temp files removed: {cleanup_stats['temp_files_removed']}")
    print(f"  Space freed: {cleanup_stats['space_freed_mb']} MB")

    return cleanup_stats

def backup_critical_data(**context):
    """Backup critical system data"""
    print("💾 Backing up critical data...")

    # Simulate backup
    backups = [
        'Database configurations',
        'DAG files',
        'System configurations',
        'User data'
    ]

    for backup in backups:
        print(f"  ✅ Backed up: {backup}")

    backup_info = {
        'backup_location': 's3://mlops-backups/',
        'backup_size_mb': 256,
        'timestamp': datetime.now().isoformat()
    }

    return backup_info

def generate_health_report(**context):
    """Generate comprehensive health report"""
    import json

    # Gather all metrics from XCom
    health_metrics = json.loads(context['task_instance'].xcom_pull(key='health_metrics'))
    containers = context['task_instance'].xcom_pull(task_ids='check_docker_containers')
    databases = context['task_instance'].xcom_pull(task_ids='check_database_connections')

    print("📊 Generating health report...")

    report = {
        'timestamp': datetime.now().isoformat(),
        'system_health': health_metrics,
        'containers': containers,
        'databases': databases,
        'summary': f"System is {health_metrics['status']}"
    }

    print(f"Report generated: {report['summary']}")

    # Save report (simulate)
    report_path = f"/tmp/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"Report saved to: {report_path}")

    return report

# Define tasks
with dag:

    # System health check
    health_check = PythonOperator(
        task_id='check_system_health',
        python_callable=check_system_health,
    )

    # Branching based on health status
    maintenance_decision = BranchPythonOperator(
        task_id='decide_maintenance_action',
        python_callable=decide_maintenance_action,
    )

    # Maintenance actions (mutually exclusive branches)
    emergency_maintenance = PythonOperator(
        task_id='emergency_maintenance',
        python_callable=perform_emergency_maintenance,
    )

    preventive_maintenance = PythonOperator(
        task_id='preventive_maintenance',
        python_callable=perform_preventive_maintenance,
    )

    routine_check = PythonOperator(
        task_id='routine_check',
        python_callable=perform_routine_check,
    )

    # Join branches
    maintenance_complete = DummyOperator(
        task_id='maintenance_complete',
        trigger_rule='none_failed_min_one_success',
    )

    # Infrastructure checks task group
    with TaskGroup('infrastructure_checks') as infrastructure:

        docker_check = PythonOperator(
            task_id='check_docker_containers',
            python_callable=check_docker_containers,
        )

        db_check = PythonOperator(
            task_id='check_database_connections',
            python_callable=check_database_connections,
        )

        docker_check >> db_check

    # Maintenance tasks task group
    with TaskGroup('maintenance_tasks') as maintenance:

        cleanup = PythonOperator(
            task_id='cleanup_old_data',
            python_callable=cleanup_old_data,
        )

        backup = PythonOperator(
            task_id='backup_critical_data',
            python_callable=backup_critical_data,
        )

        cleanup >> backup

    # Generate report
    health_report = PythonOperator(
        task_id='generate_health_report',
        python_callable=generate_health_report,
        trigger_rule='all_done',
    )

    # Alert if critical
    send_alert = BashOperator(
        task_id='send_critical_alert',
        bash_command='echo "CRITICAL ALERT: System requires immediate attention!"',
        trigger_rule='all_failed',
    )

    # Define dependencies
    health_check >> maintenance_decision
    maintenance_decision >> [emergency_maintenance, preventive_maintenance, routine_check]
    [emergency_maintenance, preventive_maintenance, routine_check] >> maintenance_complete
    maintenance_complete >> infrastructure >> maintenance >> health_report