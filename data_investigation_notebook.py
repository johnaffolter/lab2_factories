#!/usr/bin/env python3
"""
Comprehensive Data Investigation Notebook
Deep Analysis of Available Datasets and API Data Lineage

This notebook provides thorough investigation of:
1. Weather data from Airbyte ingestion
2. Demographics and location data
3. Business data (Toast orders, leads, teams)
4. Data lineage mapping from API sources
5. Feature engineering pipeline documentation
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataInvestigator:
    """Comprehensive data investigation and analysis toolkit"""

    def __init__(self, base_path="/Users/johnaffolter/innovation/fastapi-innovation-setup"):
        self.base_path = Path(base_path)
        self.datasets = {}
        self.data_lineage = {}
        self.field_mappings = {}

    def discover_datasets(self):
        """Discover all available datasets and their schemas"""
        print("=" * 80)
        print("DATASET DISCOVERY AND SCHEMA ANALYSIS")
        print("=" * 80)

        # Find all data files
        data_files = {
            'weather': self.base_path / 'data' / 'weather.csv',
            'weather_parsed': self.base_path / 'Parsed_DailyWeather_Nodes__Improved_.csv',
            'leads': self.base_path / 'KC_Royals_1000_Leads_Complete.csv',
            'teams': self.base_path / 'teams.csv',
            'strategies': self.base_path / 'strategies.csv',
            'lead_sources': self.base_path / 'lead_sources.csv',
            'toast_orders': self.base_path / 'toast_orders_20250506.json',
            'ai_insights': self.base_path / 'ai_insights_export.json',
            'comprehensive_leads': self.base_path / 'comprehensive_leads_export.json',
            'lead_alignment': self.base_path / 'lead_alignment_demo_results.json'
        }

        for name, path in data_files.items():
            if path.exists():
                print(f"\nDATASET: {name}")
                print(f"Path: {path}")
                try:
                    if path.suffix == '.csv':
                        df = pd.read_csv(path)
                        self.datasets[name] = df
                        self.analyze_csv_schema(name, df)
                    elif path.suffix == '.json':
                        with open(path, 'r') as f:
                            data = json.load(f)
                        self.datasets[name] = data
                        self.analyze_json_schema(name, data)
                except Exception as e:
                    print(f"Error loading {name}: {e}")
            else:
                print(f"File not found: {path}")

    def analyze_csv_schema(self, name, df):
        """Deep analysis of CSV dataset schema and data distribution"""
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data Types:")
        for col, dtype in df.dtypes.items():
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            print(f"    {col}: {dtype} (null: {null_count}, {null_pct:.1f}%, unique: {unique_count})")

        # Sample data
        print(f"  Sample Records:")
        print(df.head(3).to_string(index=False))

        # Data quality analysis
        self.analyze_data_quality(name, df)

    def analyze_json_schema(self, name, data):
        """Deep analysis of JSON dataset schema and nested structure"""
        print(f"  Type: {type(data)}")

        if isinstance(data, list):
            print(f"  Length: {len(data)}")
            if len(data) > 0:
                print(f"  Sample Record Structure:")
                sample = data[0]
                self.print_nested_structure(sample, indent=4)
        elif isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            print(f"  Structure:")
            self.print_nested_structure(data, indent=4)

        # Store structure for lineage mapping
        self.map_json_structure(name, data)

    def print_nested_structure(self, obj, indent=0, max_depth=3):
        """Print nested JSON structure with type information"""
        if indent > max_depth * 4:
            print(" " * indent + "... (truncated)")
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                value_type = type(value).__name__
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    print(f"{' ' * indent}{key}: {value_type}")
                    self.print_nested_structure(value, indent + 2, max_depth)
                else:
                    print(f"{' ' * indent}{key}: {value_type} = {str(value)[:100]}")
        elif isinstance(obj, list) and len(obj) > 0:
            print(f"{' ' * indent}[0]: {type(obj[0]).__name__}")
            self.print_nested_structure(obj[0], indent + 2, max_depth)

    def analyze_data_quality(self, name, df):
        """Comprehensive data quality analysis"""
        print(f"  Data Quality Assessment:")

        # Missing data analysis
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"    Missing Data Pattern:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"      {col}: {count} missing ({count/len(df)*100:.1f}%)")

        # Duplicate analysis
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"    Duplicate Rows: {duplicates}")

        # Value distribution analysis for key columns
        for col in df.columns[:5]:  # Analyze first 5 columns
            if df[col].dtype in ['object', 'string']:
                top_values = df[col].value_counts().head(3)
                print(f"    {col} - Top Values: {dict(top_values)}")
            elif df[col].dtype in ['int64', 'float64']:
                stats = df[col].describe()
                print(f"    {col} - Stats: min={stats['min']:.2f}, mean={stats['mean']:.2f}, max={stats['max']:.2f}")

    def map_json_structure(self, name, data):
        """Map JSON structure for data lineage documentation"""
        self.field_mappings[name] = self.extract_field_paths(data)

    def extract_field_paths(self, obj, prefix=""):
        """Extract all field paths from nested JSON structure"""
        paths = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{prefix}.{key}" if prefix else key
                paths.append({
                    'path': current_path,
                    'type': type(value).__name__,
                    'sample_value': str(value)[:100] if not isinstance(value, (dict, list)) else None
                })

                if isinstance(value, (dict, list)):
                    paths.extend(self.extract_field_paths(value, current_path))

        elif isinstance(obj, list) and len(obj) > 0:
            # Analyze first item in list
            paths.extend(self.extract_field_paths(obj[0], f"{prefix}[0]"))

        return paths

    def analyze_weather_data_lineage(self):
        """Specific analysis of weather data from Airbyte ingestion"""
        print("\n" + "=" * 80)
        print("WEATHER DATA LINEAGE ANALYSIS")
        print("=" * 80)

        if 'weather' in self.datasets:
            weather_df = self.datasets['weather']

            print("Raw Weather Data Analysis:")
            print(f"  Date Range: {weather_df['date'].min()} to {weather_df['date'].max()}")
            print(f"  Locations: {weather_df['location_id'].nunique()} unique")
            print(f"  Location IDs: {sorted(weather_df['location_id'].unique())}")

            # Temperature analysis
            temp_stats = weather_df['temperature'].describe()
            print(f"  Temperature Range: {temp_stats['min']:.1f}°F to {temp_stats['max']:.1f}°F")
            print(f"  Average Temperature: {temp_stats['mean']:.1f}°F")

            # Precipitation analysis
            precip_days = (weather_df['precipitation'] > 0).sum()
            total_days = len(weather_df)
            print(f"  Precipitation Days: {precip_days}/{total_days} ({precip_days/total_days*100:.1f}%)")

            # Data quality checks
            self.check_weather_data_quality(weather_df)

        if 'weather_parsed' in self.datasets:
            parsed_weather = self.datasets['weather_parsed']
            print(f"\nParsed Weather Data:")
            print(f"  Shape: {parsed_weather.shape}")
            print(f"  Columns: {list(parsed_weather.columns)}")

    def check_weather_data_quality(self, df):
        """Specific data quality checks for weather data"""
        print("  Data Quality Checks:")

        # Check for reasonable temperature ranges
        temp_outliers = df[(df['temperature'] < -50) | (df['temperature'] > 120)]
        if len(temp_outliers) > 0:
            print(f"    Temperature Outliers: {len(temp_outliers)} records")

        # Check for negative precipitation
        negative_precip = df[df['precipitation'] < 0]
        if len(negative_precip) > 0:
            print(f"    Negative Precipitation: {len(negative_precip)} records")

        # Check for unrealistic humidity
        humidity_outliers = df[(df['humidity'] < 0) | (df['humidity'] > 100)]
        if len(humidity_outliers) > 0:
            print(f"    Humidity Outliers: {len(humidity_outliers)} records")

        # Check for data completeness by location
        location_completeness = df.groupby('location_id').size()
        print(f"    Records per Location: {dict(location_completeness)}")

    def analyze_business_data_integration(self):
        """Analyze business data sources for integration opportunities"""
        print("\n" + "=" * 80)
        print("BUSINESS DATA INTEGRATION ANALYSIS")
        print("=" * 80)

        # Analyze leads data
        if 'leads' in self.datasets:
            leads_df = self.datasets['leads']
            print("Leads Data Analysis:")
            print(f"  Total Leads: {len(leads_df)}")

            # Geographic analysis if location data exists
            location_columns = [col for col in leads_df.columns if any(geo in col.lower()
                              for geo in ['city', 'state', 'zip', 'location', 'address'])]
            if location_columns:
                print(f"  Location Fields: {location_columns}")
                for col in location_columns[:3]:  # Analyze first 3 location columns
                    if col in leads_df.columns:
                        unique_values = leads_df[col].nunique()
                        print(f"    {col}: {unique_values} unique values")

        # Analyze Toast orders for temporal patterns
        if 'toast_orders' in self.datasets:
            print("\nToast Orders Analysis:")
            orders = self.datasets['toast_orders']
            if isinstance(orders, list) and len(orders) > 0:
                print(f"  Total Orders: {len(orders)}")

                # Look for timestamp fields
                sample_order = orders[0]
                timestamp_fields = [key for key in sample_order.keys()
                                  if any(time_word in key.lower()
                                       for time_word in ['date', 'time', 'created', 'updated'])]
                print(f"  Timestamp Fields: {timestamp_fields}")

    def create_integration_schema(self):
        """Create enhanced semantic views for cross-dataset integration"""
        print("\n" + "=" * 80)
        print("ENHANCED SEMANTIC VIEWS CREATION")
        print("=" * 80)

        integration_schema = {
            'weather_enhanced': {
                'description': 'Enhanced weather data with calculated fields and quality indicators',
                'source_tables': ['weather', 'weather_parsed'],
                'calculated_fields': {
                    'temperature_celsius': 'Converted from Fahrenheit using (F-32)*5/9',
                    'feels_like_temp': 'Calculated using humidity and wind chill factors',
                    'precipitation_category': 'Categorized as None/Light/Moderate/Heavy',
                    'weather_score': 'Composite score for outdoor activity suitability',
                    'data_quality_flag': 'Quality indicator based on validation rules'
                },
                'business_context': 'Weather conditions affecting customer behavior and operations'
            },
            'location_demographics': {
                'description': 'Integrated location data with demographic insights',
                'source_tables': ['leads', 'teams', 'weather'],
                'calculated_fields': {
                    'customer_density': 'Number of customers per geographic area',
                    'weather_preference_score': 'Historical preference based on weather patterns',
                    'market_penetration': 'Percentage of potential market captured',
                    'seasonal_demand_index': 'Demand patterns by season and weather'
                },
                'business_context': 'Geographic and demographic factors driving business performance'
            },
            'temporal_business_patterns': {
                'description': 'Time-based business intelligence with weather correlation',
                'source_tables': ['toast_orders', 'weather', 'leads'],
                'calculated_fields': {
                    'weather_adjusted_demand': 'Sales demand adjusted for weather impact',
                    'lead_conversion_by_weather': 'Conversion rates correlated with weather',
                    'optimal_outreach_windows': 'Best times for customer engagement',
                    'seasonal_revenue_forecast': 'Predictive revenue modeling'
                },
                'business_context': 'Temporal patterns for operational optimization'
            }
        }

        return integration_schema

    def generate_data_lineage_documentation(self):
        """Generate comprehensive data lineage documentation"""
        print("\n" + "=" * 80)
        print("DATA LINEAGE DOCUMENTATION")
        print("=" * 80)

        lineage_doc = {
            'data_sources': {
                'airbyte_weather_api': {
                    'description': 'Weather data ingested via Airbyte from weather API',
                    'ingestion_frequency': 'Daily',
                    'data_format': 'JSON via REST API',
                    'key_fields': ['weather_id', 'date', 'location_id', 'temperature', 'humidity', 'precipitation', 'wind_speed'],
                    'transformations': [
                        'Date standardization to ISO format',
                        'Temperature unit conversion (F to C)',
                        'Data quality validation and flagging',
                        'Location ID mapping to geographic coordinates'
                    ]
                },
                'business_systems': {
                    'description': 'Operational data from Toast POS and CRM systems',
                    'ingestion_frequency': 'Real-time/Hourly',
                    'data_format': 'JSON via webhooks and API polling',
                    'key_fields': ['order_id', 'timestamp', 'location', 'customer_data', 'revenue'],
                    'transformations': [
                        'Customer data anonymization',
                        'Revenue aggregation by time periods',
                        'Location standardization',
                        'Event stream processing for real-time insights'
                    ]
                }
            },
            'feature_engineering_pipeline': {
                'weather_features': [
                    'temperature_moving_avg_7d: 7-day rolling average temperature',
                    'precipitation_cumulative_30d: 30-day cumulative precipitation',
                    'extreme_weather_flag: Binary indicator for extreme conditions',
                    'seasonal_temperature_deviation: Deviation from seasonal normal'
                ],
                'business_features': [
                    'customer_lifetime_value: Calculated CLV based on historical transactions',
                    'location_performance_index: Normalized performance score by location',
                    'time_based_demand_pattern: Hourly/daily demand patterns',
                    'weather_sensitivity_score: Business sensitivity to weather changes'
                ],
                'integrated_features': [
                    'weather_adjusted_sales_forecast: Sales prediction with weather factors',
                    'optimal_staffing_recommendation: Staffing needs based on predicted demand',
                    'inventory_optimization_score: Inventory needs based on weather and demand',
                    'customer_engagement_timing: Optimal times for marketing based on conditions'
                ]
            }
        }

        return lineage_doc

    def run_comprehensive_investigation(self):
        """Execute complete data investigation workflow"""
        print("STARTING COMPREHENSIVE DATA INVESTIGATION")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Dataset Discovery
        self.discover_datasets()

        # Step 2: Weather Data Lineage Analysis
        self.analyze_weather_data_lineage()

        # Step 3: Business Data Integration Analysis
        self.analyze_business_data_integration()

        # Step 4: Create Integration Schema
        integration_schema = self.create_integration_schema()

        # Step 5: Generate Data Lineage Documentation
        lineage_doc = self.generate_data_lineage_documentation()

        # Step 6: Summary Report
        print("\n" + "=" * 80)
        print("INVESTIGATION SUMMARY")
        print("=" * 80)
        print(f"Datasets Analyzed: {len(self.datasets)}")
        print(f"Data Sources Identified: {len(lineage_doc['data_sources'])}")
        print(f"Integration Schemas Created: {len(integration_schema)}")
        print(f"Field Mappings Documented: {sum(len(mappings) for mappings in self.field_mappings.values())}")

        return {
            'datasets': self.datasets,
            'integration_schema': integration_schema,
            'lineage_documentation': lineage_doc,
            'field_mappings': self.field_mappings
        }

if __name__ == "__main__":
    # Initialize and run investigation
    investigator = DataInvestigator()
    results = investigator.run_comprehensive_investigation()

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print("Results available in 'results' variable")
    print("Use investigator.datasets to access loaded datasets")
    print("Use results['integration_schema'] for semantic view definitions")
    print("Use results['lineage_documentation'] for data lineage mapping")