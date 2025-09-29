# 🚀 AIRFLOW ACCESS GUIDE - COMPLETE SOLUTION

## Problem: Browser "Request Header Fields Too Large" Error
Your browser is sending too many/large headers to Airflow, causing the 431 error. This is now FIXED with multiple access methods below.

## ✅ SOLUTION 1: Lightweight Web Interface (RECOMMENDED)

This bypasses all browser header issues by using a proxy interface.

```bash
# Start the lightweight interface
python3 airflow_web_interface.py
```

Then access: **http://localhost:5555**

Features:
- Beautiful, modern UI
- No header issues
- View all DAGs
- Trigger/Pause/Resume DAGs
- View run history
- Real-time statistics
- Auto-refresh every 30 seconds

## ✅ SOLUTION 2: Command Line API Tool

Full Airflow functionality without any browser.

```bash
# Make it executable
chmod +x airflow_api_tool.py

# Interactive mode (easiest)
python3 airflow_api_tool.py interactive

# Or use specific commands
python3 airflow_api_tool.py list                    # List all DAGs
python3 airflow_api_tool.py trigger mlops_data_pipeline  # Trigger a DAG
python3 airflow_api_tool.py runs mlops_data_pipeline     # View runs
python3 airflow_api_tool.py tasks simple_mlops_dag       # List tasks
```

## ✅ SOLUTION 3: Direct Docker Commands

Immediate access without any additional tools.

```bash
# List all DAGs
docker exec airflow-standalone airflow dags list

# Trigger a specific DAG
docker exec airflow-standalone airflow dags trigger mlops_data_pipeline

# View recent runs
docker exec airflow-standalone airflow dags list-runs --dag-id mlops_data_pipeline

# Pause a DAG
docker exec airflow-standalone airflow dags pause monitoring_maintenance

# Unpause a DAG
docker exec airflow-standalone airflow dags unpause monitoring_maintenance

# Test a specific task
docker exec airflow-standalone airflow tasks test mlops_data_pipeline extract_s3_data 2024-01-26

# View logs
docker logs -f airflow-standalone
```

## ✅ SOLUTION 4: CLI Interface Tool

Interactive menu system for easy navigation.

```bash
python3 airflow_cli_interface.py
```

Features:
- Interactive menu
- List and trigger DAGs
- Pause/unpause operations
- View run status
- Health checks

## ✅ SOLUTION 5: Fix Browser Access (If You Prefer)

To fix the browser header issue:

### Option A: Clear Browser Data
1. Clear ALL cookies and cache for localhost:8080
2. Close browser completely
3. Open new incognito/private window
4. Access: http://localhost:8080

### Option B: Use Different Browser
- If using Chrome, try Firefox
- If using Safari, try Chrome
- Use a browser you haven't used for this project before

### Option C: Use curl to test
```bash
# Test if Airflow is accessible
curl -s http://localhost:8080/health

# Login with curl
curl -X POST http://localhost:8080/login/ \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" \
  -c cookies.txt

# Access with cookies
curl -b cookies.txt http://localhost:8080/home
```

### Option D: Restart with optimized configuration
```bash
# Stop current Airflow
docker stop airflow-standalone
docker rm airflow-standalone

# Start with browser-friendly settings
./airflow_fixed_start.sh
```

## 📊 Your Custom DAGs

You have 3 custom DAGs created:

1. **mlops_data_pipeline** - Complete data pipeline with S3, PostgreSQL, ML training
2. **monitoring_maintenance** - System monitoring and health checks
3. **simple_mlops_dag** - Basic pipeline example

## 🔍 Quick Status Check

```bash
# Check if Airflow is running
docker ps | grep airflow

# Check Airflow health
curl http://localhost:8080/health

# View container logs
docker logs --tail 50 airflow-standalone
```

## 🎯 Recommended Approach

1. **Use the Lightweight Web Interface (Solution 1)**
   - No browser issues
   - Beautiful UI
   - Full functionality
   - Just run: `python3 airflow_web_interface.py`

2. **Or use the API Tool (Solution 2)**
   - Complete programmatic access
   - No UI needed
   - Scriptable/automatable
   - Run: `python3 airflow_api_tool.py interactive`

## ⚡ Quick Start Commands

```bash
# Option 1: Web Interface (BEST)
python3 airflow_web_interface.py
# Then open: http://localhost:5555

# Option 2: CLI Tool
python3 airflow_api_tool.py interactive

# Option 3: Direct trigger
docker exec airflow-standalone airflow dags trigger mlops_data_pipeline
```

## 🚨 Troubleshooting

If nothing works:

```bash
# Complete reset
docker stop airflow-standalone
docker stop mlops_postgres
docker rm airflow-standalone
docker rm mlops_postgres

# Fresh start
./airflow_fixed_start.sh

# Use the lightweight interface
python3 airflow_web_interface.py
```

---

**The browser header issue is now completely bypassed!** Use the lightweight web interface at http://localhost:5555 for the best experience.