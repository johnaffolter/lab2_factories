#!/usr/bin/env python3
"""
Enhanced Semantic Views with Explainable Columns
Deep Query Analysis and Data Distribution Investigation

This creates sophisticated semantic views that explain:
1. Data lineage from Airbyte ingestion to transformed views
2. Feature engineering flows with business context
3. Cross-dataset integration opportunities
4. Weather and demographics integration patterns
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class SemanticViewBuilder:
    """Build enhanced semantic views with explainable columns and business context"""

    def __init__(self, base_path="/Users/johnaffolter/innovation/fastapi-innovation-setup"):
        self.base_path = Path(base_path)
        self.semantic_views = {}
        self.data_lineage = {}
        self.business_context = {}

    def create_weather_semantic_view(self):
        """Create enhanced weather view with business intelligence"""
        print("=" * 80)
        print("CREATING WEATHER SEMANTIC VIEW")
        print("=" * 80)

        # Load weather data
        weather_df = pd.read_csv(self.base_path / 'data' / 'weather.csv')
        weather_parsed_df = pd.read_csv(self.base_path / 'Parsed_DailyWeather_Nodes__Improved_.csv')

        print(f"Raw Weather Records: {len(weather_df)}")
        print(f"Parsed Weather Records: {len(weather_parsed_df)}")

        # Create comprehensive weather view
        weather_semantic = self.build_weather_features(weather_df, weather_parsed_df)

        # Document data lineage
        self.data_lineage['weather_semantic'] = {
            'source_systems': {
                'weather_api': 'Primary weather API ingested via Airbyte',
                'weather_service': 'Enhanced weather service with parsing'
            },
            'transformation_flow': [
                'API Response -> Airbyte Ingestion -> Raw Weather Table',
                'Raw Weather -> Parsing Service -> Enhanced Weather Attributes',
                'Enhanced Weather -> Business Logic -> Weather Intelligence View',
                'Weather Intelligence -> Feature Engineering -> Business Impact Metrics'
            ],
            'field_mappings': {
                'temperature': 'temperature -> temperature_fahrenheit (direct mapping)',
                'temperature_celsius': 'temperature_fahrenheit -> (F-32)*5/9 (calculated)',
                'weather_category': 'precipitation + condition -> business category (derived)',
                'outdoor_activity_score': 'temperature + humidity + wind -> activity suitability (calculated)',
                'business_impact_flag': 'weather_category + temperature_range -> operational impact (derived)'
            }
        }

        return weather_semantic

    def build_weather_features(self, raw_weather, parsed_weather):
        """Build comprehensive weather feature set with business context"""

        # Enhanced weather features
        weather_features = raw_weather.copy()

        # Temperature conversions and categorizations
        weather_features['temperature_celsius'] = (weather_features['temperature'] - 32) * 5/9
        weather_features['temperature_category'] = pd.cut(
            weather_features['temperature'],
            bins=[-float('inf'), 32, 50, 70, 85, float('inf')],
            labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
        )

        # Precipitation analysis
        weather_features['precipitation_category'] = pd.cut(
            weather_features['precipitation'],
            bins=[-0.01, 0, 0.1, 0.5, float('inf')],
            labels=['None', 'Light', 'Moderate', 'Heavy']
        )

        # Business impact scoring
        weather_features['outdoor_activity_score'] = self.calculate_outdoor_score(weather_features)
        weather_features['customer_comfort_index'] = self.calculate_comfort_index(weather_features)
        weather_features['operational_impact_flag'] = self.determine_operational_impact(weather_features)

        # Temporal features
        weather_features['date'] = pd.to_datetime(weather_features['date'])
        weather_features['day_of_week'] = weather_features['date'].dt.day_name()
        weather_features['is_weekend'] = weather_features['date'].dt.weekday >= 5
        weather_features['season'] = weather_features['date'].dt.month.apply(self.get_season)

        print("Weather Features Created:")
        for col in weather_features.columns:
            if col not in raw_weather.columns:
                unique_vals = weather_features[col].unique()
                print(f"  {col}: {len(unique_vals)} unique values - {list(unique_vals)[:5]}")

        return weather_features

    def calculate_outdoor_score(self, df):
        """Calculate outdoor activity suitability score (0-100)"""
        score = 50  # Base score

        # Temperature adjustments
        temp_adjustment = np.where(
            (df['temperature'] >= 60) & (df['temperature'] <= 80), 20,
            np.where((df['temperature'] >= 50) & (df['temperature'] < 90), 10, -20)
        )

        # Precipitation penalty
        precip_penalty = np.where(df['precipitation'] > 0, -15, 0)

        # Wind adjustment
        wind_adjustment = np.where(df['wind_speed'] > 15, -10, 0)

        # Humidity adjustment
        humidity_adjustment = np.where(
            (df['humidity'] >= 30) & (df['humidity'] <= 70), 5, -5
        )

        final_score = score + temp_adjustment + precip_penalty + wind_adjustment + humidity_adjustment
        return np.clip(final_score, 0, 100)

    def calculate_comfort_index(self, df):
        """Calculate customer comfort index for indoor/outdoor dining"""
        # Simplified comfort calculation
        temp_comfort = 100 - abs(df['temperature'] - 72) * 2  # Optimal at 72F
        humidity_comfort = 100 - abs(df['humidity'] - 50) * 1.5  # Optimal at 50%
        wind_comfort = np.maximum(0, 100 - df['wind_speed'] * 3)  # Less wind is better

        comfort_index = (temp_comfort + humidity_comfort + wind_comfort) / 3
        return np.clip(comfort_index, 0, 100)

    def determine_operational_impact(self, df):
        """Determine if weather conditions significantly impact operations"""
        extreme_temp = (df['temperature'] < 20) | (df['temperature'] > 95)
        heavy_precip = df['precipitation'] > 0.25
        high_wind = df['wind_speed'] > 20

        return (extreme_temp | heavy_precip | high_wind).astype(int)

    def get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def create_business_semantic_view(self):
        """Create integrated business intelligence view"""
        print("\n" + "=" * 80)
        print("CREATING BUSINESS SEMANTIC VIEW")
        print("=" * 80)

        # Load business data
        toast_orders_file = self.base_path / 'toast_orders_20250506.json'
        ai_insights_file = self.base_path / 'ai_insights_export.json'
        comprehensive_leads_file = self.base_path / 'comprehensive_leads_export.json'

        business_data = {}

        if toast_orders_file.exists():
            with open(toast_orders_file, 'r') as f:
                business_data['orders'] = json.load(f)

        if ai_insights_file.exists():
            with open(ai_insights_file, 'r') as f:
                business_data['insights'] = json.load(f)

        if comprehensive_leads_file.exists():
            with open(comprehensive_leads_file, 'r') as f:
                business_data['leads'] = json.load(f)

        # Create business semantic view
        business_semantic = self.build_business_features(business_data)

        # Document business data lineage
        self.data_lineage['business_semantic'] = {
            'source_systems': {
                'toast_pos': 'Toast Point of Sale system via API',
                'crm_system': 'Customer relationship management system',
                'ai_analytics': 'AI-powered business insights engine'
            },
            'transformation_flow': [
                'Toast POS API -> Order Events -> Raw Order Data',
                'Raw Orders -> Revenue Calculation -> Financial Metrics',
                'CRM API -> Lead Data -> Customer Intelligence',
                'Customer Intelligence -> AI Analysis -> Business Insights',
                'Multi-source Integration -> Business Intelligence View'
            ],
            'business_rules': {
                'revenue_potential': 'Calculated based on industry standards and company size',
                'priority_scoring': 'Weighted algorithm considering decision authority and timing',
                'engagement_likelihood': 'ML model prediction based on historical patterns',
                'operational_metrics': 'Real-time calculations from POS transaction data'
            }
        }

        return business_semantic

    def build_business_features(self, business_data):
        """Build comprehensive business intelligence features"""

        business_features = {}

        # Order analysis
        if 'orders' in business_data:
            orders = business_data['orders']
            print(f"Processing {len(orders)} Toast orders...")

            business_features['order_metrics'] = self.analyze_order_patterns(orders)

        # Lead analysis
        if 'leads' in business_data:
            leads = business_data['leads']
            print(f"Processing {len(leads)} comprehensive leads...")

            business_features['lead_intelligence'] = self.analyze_lead_patterns(leads)

        # AI insights integration
        if 'insights' in business_data:
            insights = business_data['insights']
            print("Processing AI insights...")

            business_features['ai_intelligence'] = self.process_ai_insights(insights)

        return business_features

    def analyze_order_patterns(self, orders):
        """Deep analysis of order patterns for business intelligence"""

        order_analysis = {
            'temporal_patterns': {},
            'revenue_patterns': {},
            'customer_patterns': {}
        }

        # Extract order data
        order_data = []
        for order in orders:
            order_info = {
                'order_id': order.get('guid'),
                'business_date': order.get('businessDate'),
                'opened_date': order.get('openedDate'),
                'closed_date': order.get('closedDate'),
                'number_of_guests': order.get('numberOfGuests', 0),
                'duration_minutes': order.get('duration', 0) / 60 if order.get('duration') else 0,
                'total_amount': 0,
                'items_count': 0
            }

            # Calculate total amount from checks
            if 'checks' in order:
                for check in order['checks']:
                    order_info['total_amount'] += check.get('totalAmount', 0)
                    if 'selections' in check:
                        order_info['items_count'] += len(check['selections'])

            order_data.append(order_info)

        # Convert to DataFrame for analysis
        orders_df = pd.DataFrame(order_data)

        if not orders_df.empty:
            # Temporal analysis
            if 'opened_date' in orders_df.columns:
                orders_df['opened_datetime'] = pd.to_datetime(orders_df['opened_date'])
                orders_df['hour'] = orders_df['opened_datetime'].dt.hour
                orders_df['day_of_week'] = orders_df['opened_datetime'].dt.day_name()

                order_analysis['temporal_patterns'] = {
                    'peak_hours': orders_df['hour'].value_counts().to_dict(),
                    'day_patterns': orders_df['day_of_week'].value_counts().to_dict(),
                    'average_duration': orders_df['duration_minutes'].mean()
                }

            # Revenue analysis
            order_analysis['revenue_patterns'] = {
                'total_revenue': orders_df['total_amount'].sum(),
                'average_order_value': orders_df['total_amount'].mean(),
                'revenue_distribution': orders_df['total_amount'].describe().to_dict()
            }

            # Customer patterns
            order_analysis['customer_patterns'] = {
                'average_party_size': orders_df['number_of_guests'].mean(),
                'party_size_distribution': orders_df['number_of_guests'].value_counts().to_dict()
            }

        print("Order Analysis Complete:")
        print(f"  Total Orders Analyzed: {len(orders_df)}")
        print(f"  Total Revenue: ${order_analysis['revenue_patterns'].get('total_revenue', 0):.2f}")
        print(f"  Average Order Value: ${order_analysis['revenue_patterns'].get('average_order_value', 0):.2f}")

        return order_analysis

    def analyze_lead_patterns(self, leads):
        """Analyze lead data for customer intelligence"""

        lead_analysis = {
            'industry_distribution': {},
            'geographic_patterns': {},
            'revenue_intelligence': {},
            'engagement_intelligence': {}
        }

        # Convert leads to DataFrame
        lead_data = []
        for lead in leads:
            lead_info = {
                'organization': lead.get('organization'),
                'industry': lead.get('industry'),
                'location': lead.get('location'),
                'revenue': lead.get('revenue'),
                'employees': lead.get('employees'),
                'priority_score': lead.get('priority_score', 0),
                'revenue_potential': lead.get('revenue_potential', 0),
                'engagement_likelihood': lead.get('engagement_likelihood', 0),
                'decision_maker_probability': lead.get('decision_maker_probability', 0)
            }
            lead_data.append(lead_info)

        leads_df = pd.DataFrame(lead_data)

        if not leads_df.empty:
            # Industry analysis
            industry_counts = leads_df['industry'].value_counts()
            industry_revenue = leads_df.groupby('industry')['revenue_potential'].sum()

            lead_analysis['industry_distribution'] = {
                'counts': industry_counts.to_dict(),
                'revenue_potential': industry_revenue.to_dict(),
                'top_industries': industry_counts.head(5).index.tolist()
            }

            # Geographic analysis
            location_counts = leads_df['location'].value_counts()
            lead_analysis['geographic_patterns'] = {
                'location_distribution': location_counts.to_dict(),
                'top_markets': location_counts.head(5).index.tolist()
            }

            # Revenue intelligence
            lead_analysis['revenue_intelligence'] = {
                'total_pipeline': leads_df['revenue_potential'].sum(),
                'average_deal_size': leads_df['revenue_potential'].mean(),
                'high_value_leads': len(leads_df[leads_df['revenue_potential'] > 50000])
            }

            # Engagement intelligence
            lead_analysis['engagement_intelligence'] = {
                'high_priority_leads': len(leads_df[leads_df['priority_score'] > 8.0]),
                'high_engagement_probability': len(leads_df[leads_df['engagement_likelihood'] > 0.8]),
                'confirmed_decision_makers': len(leads_df[leads_df['decision_maker_probability'] > 0.9])
            }

        print("Lead Analysis Complete:")
        print(f"  Total Leads Analyzed: {len(leads_df)}")
        print(f"  Total Pipeline Value: ${lead_analysis['revenue_intelligence'].get('total_pipeline', 0):,.0f}")
        print(f"  Top Industry: {lead_analysis['industry_distribution']['top_industries'][0] if lead_analysis['industry_distribution']['top_industries'] else 'N/A'}")

        return lead_analysis

    def process_ai_insights(self, insights):
        """Process AI-generated business insights"""

        ai_intelligence = {
            'summary_metrics': insights.get('summary', {}),
            'industry_intelligence': insights.get('industry_analysis', {}),
            'opportunity_ranking': insights.get('top_opportunities', []),
            'strategic_recommendations': insights.get('recommendations', [])
        }

        print("AI Intelligence Processing:")
        summary = ai_intelligence['summary_metrics']
        print(f"  Total Leads: {summary.get('total_leads', 0)}")
        print(f"  Revenue Potential: ${summary.get('total_revenue_potential', 0):,}")
        print(f"  High Priority Leads: {summary.get('high_priority_leads', 0)}")

        return ai_intelligence

    def create_integrated_semantic_view(self):
        """Create master integrated view combining all data sources"""
        print("\n" + "=" * 80)
        print("CREATING INTEGRATED SEMANTIC VIEW")
        print("=" * 80)

        # Combine weather and business views
        weather_view = self.create_weather_semantic_view()
        business_view = self.create_business_semantic_view()

        # Create integration mappings
        integration_schema = {
            'weather_business_correlation': {
                'description': 'Correlation between weather conditions and business performance',
                'integration_keys': ['date', 'location', 'business_date'],
                'calculated_metrics': {
                    'weather_adjusted_revenue': 'Revenue adjusted for weather impact factors',
                    'optimal_conditions_flag': 'Binary indicator for ideal business conditions',
                    'weather_sensitivity_score': 'Business sensitivity to weather changes',
                    'predicted_customer_traffic': 'Traffic prediction based on weather patterns'
                }
            },
            'location_intelligence': {
                'description': 'Geographic intelligence combining weather and customer data',
                'integration_keys': ['location_id', 'geographic_coordinates'],
                'calculated_metrics': {
                    'market_weather_profile': 'Weather patterns affecting each market',
                    'location_performance_index': 'Performance normalized for weather conditions',
                    'seasonal_demand_forecast': 'Seasonal demand patterns by location'
                }
            }
        }

        # Document complete data lineage
        complete_lineage = {
            'source_to_insight_flow': [
                'Weather API -> Airbyte -> Raw Weather Data',
                'Toast POS API -> Order Management -> Transaction Data',
                'CRM Systems -> Lead Management -> Customer Intelligence',
                'Multi-source ETL -> Data Warehouse -> Integrated Views',
                'Integrated Views -> AI Analytics -> Business Intelligence',
                'Business Intelligence -> Decision Support -> Operational Insights'
            ],
            'feature_engineering_pipeline': {
                'weather_features': [
                    'temperature -> temperature_category (binning)',
                    'precipitation -> weather_impact_score (calculation)',
                    'multiple_weather_vars -> outdoor_activity_score (composite)',
                    'date -> seasonal_patterns (temporal)'
                ],
                'business_features': [
                    'order_data -> revenue_metrics (aggregation)',
                    'lead_data -> opportunity_scoring (ml_algorithm)',
                    'customer_data -> engagement_probability (predictive)',
                    'temporal_data -> demand_patterns (time_series)'
                ],
                'integrated_features': [
                    'weather + revenue -> weather_adjusted_performance (correlation)',
                    'location + weather + business -> market_intelligence (fusion)',
                    'time + weather + demand -> predictive_analytics (forecasting)'
                ]
            }
        }

        print("Integration Schema Created:")
        print(f"  Weather Data Sources: {len(self.data_lineage.get('weather_semantic', {}).get('source_systems', {}))}")
        print(f"  Business Data Sources: {len(self.data_lineage.get('business_semantic', {}).get('source_systems', {}))}")
        print(f"  Integration Points: {len(integration_schema)}")

        return {
            'weather_view': weather_view,
            'business_view': business_view,
            'integration_schema': integration_schema,
            'complete_lineage': complete_lineage
        }

    def generate_explainable_documentation(self):
        """Generate comprehensive documentation of all semantic views"""
        print("\n" + "=" * 80)
        print("GENERATING EXPLAINABLE DOCUMENTATION")
        print("=" * 80)

        documentation = {
            'semantic_views_overview': {
                'purpose': 'Provide business-friendly views of complex data with explainable transformations',
                'approach': 'Layer business context on top of technical data structures',
                'benefits': [
                    'Improved data accessibility for business users',
                    'Clear data lineage and transformation logic',
                    'Business-relevant feature engineering',
                    'Integration-ready data structures'
                ]
            },
            'data_lineage_explanation': self.data_lineage,
            'business_context': {
                'weather_intelligence': 'Weather data transformed into business impact metrics',
                'customer_intelligence': 'Lead and customer data enhanced with engagement predictions',
                'operational_intelligence': 'Transaction data with performance indicators',
                'predictive_intelligence': 'Forward-looking metrics for decision support'
            },
            'integration_patterns': {
                'temporal_integration': 'Time-based correlation across data sources',
                'geographic_integration': 'Location-based data fusion',
                'behavioral_integration': 'Customer behavior patterns across touchpoints',
                'operational_integration': 'Real-time business metrics with contextual data'
            }
        }

        return documentation

if __name__ == "__main__":
    print("ENHANCED SEMANTIC VIEWS BUILDER")
    print("Creating explainable data views with business context")
    print("=" * 80)

    # Initialize builder
    builder = SemanticViewBuilder()

    # Create integrated semantic view
    integrated_view = builder.create_integrated_semantic_view()

    # Generate documentation
    documentation = builder.generate_explainable_documentation()

    print("\n" + "=" * 80)
    print("SEMANTIC VIEWS CREATION COMPLETE")
    print("=" * 80)
    print("Views created with full data lineage documentation")
    print("All transformations documented with business context")
    print("Integration patterns established for cross-dataset analysis")