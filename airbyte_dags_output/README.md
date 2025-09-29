# Airbyte Modular DAGs for Email Analysis

This directory contains modular DAGs for advanced email analysis using Airbyte and Airflow.

## Generated DAGs

- `email_classification_pipeline`: Advanced email classification with ML and analytics
- `email_security_analysis`: Security analysis and threat detection for emails
- `realtime_email_analysis`: Real-time email processing and classification
- `email_batch_analytics`: Daily batch analytics and ML model training

## Setup Instructions

1. **Environment Setup**:
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials
   ```

2. **Start Services**:
   ```bash
   docker-compose up -d
   ```

3. **Access Services**:
   - Airbyte UI: http://localhost:8001
   - Airflow UI: http://localhost:8080
   - Neo4j Browser: http://localhost:7474
   - Grafana: http://localhost:3000

4. **Configure Airbyte Connections**:
   - Import the generated YAML configurations
   - Set up your data source credentials
   - Test connections before running DAGs

## DAG Features

- **Modular Design**: Each DAG is built from reusable components
- **Multiple Data Sources**: Gmail, Outlook, Files, APIs
- **Advanced Analytics**: ML classification, security analysis, grammar checking
- **Scalable Storage**: PostgreSQL, Neo4j, Snowflake support
- **Monitoring**: Integrated logging and notifications

## Customization

Modify the DAG configurations in the Python files to:
- Add new data sources
- Change analysis types
- Adjust scheduling
- Add custom processing steps

## Monitoring

Each DAG includes:
- Data quality checks
- Error handling and retries
- Slack notifications
- Comprehensive logging