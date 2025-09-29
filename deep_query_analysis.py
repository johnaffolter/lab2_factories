#!/usr/bin/env python3
"""
Deep Query Analysis and Data Distribution Investigation
Comprehensive SQL-like analysis of data patterns and distributions

This performs deep queries to understand:
1. Data distribution patterns across all datasets
2. Cross-dataset correlation analysis
3. Temporal pattern identification
4. Business intelligence query patterns
5. Data quality and completeness analysis
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class DeepQueryAnalyzer:
    """Perform comprehensive query analysis across all data sources"""

    def __init__(self, base_path="/Users/johnaffolter/innovation/fastapi-innovation-setup"):
        self.base_path = Path(base_path)
        self.datasets = {}
        self.query_results = {}
        self.distribution_analysis = {}
        self.correlation_matrices = {}

    def load_all_datasets(self):
        """Load all available datasets for analysis"""
        print("=" * 80)
        print("LOADING ALL DATASETS FOR DEEP ANALYSIS")
        print("=" * 80)

        # CSV datasets
        csv_files = {
            'weather': 'data/weather.csv',
            'weather_parsed': 'Parsed_DailyWeather_Nodes__Improved_.csv',
            'teams': 'teams.csv',
            'strategies': 'strategies.csv',
            'lead_sources': 'lead_sources.csv'
        }

        for name, file_path in csv_files.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    df = pd.read_csv(full_path)
                    self.datasets[name] = df
                    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    print(f"Error loading {name}: {e}")

        # JSON datasets
        json_files = {
            'toast_orders': 'toast_orders_20250506.json',
            'ai_insights': 'ai_insights_export.json',
            'comprehensive_leads': 'comprehensive_leads_export.json',
            'lead_alignment': 'lead_alignment_demo_results.json'
        }

        for name, file_path in json_files.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    self.datasets[name] = data
                    if isinstance(data, list):
                        print(f"Loaded {name}: {len(data)} records")
                    else:
                        print(f"Loaded {name}: {type(data).__name__} structure")
                except Exception as e:
                    print(f"Error loading {name}: {e}")

    def run_weather_distribution_analysis(self):
        """Deep analysis of weather data distributions"""
        print("\n" + "=" * 80)
        print("WEATHER DATA DISTRIBUTION ANALYSIS")
        print("=" * 80)

        if 'weather' not in self.datasets:
            print("Weather data not available")
            return

        weather_df = self.datasets['weather']

        # Temperature distribution analysis
        temp_stats = {
            'descriptive_stats': weather_df['temperature'].describe(),
            'distribution_shape': {
                'skewness': stats.skew(weather_df['temperature']),
                'kurtosis': stats.kurtosis(weather_df['temperature']),
                'normality_test': stats.shapiro(weather_df['temperature'])
            },
            'outlier_analysis': self.detect_outliers(weather_df['temperature'])
        }

        print("Temperature Distribution:")
        print(f"  Mean: {temp_stats['descriptive_stats']['mean']:.2f}°F")
        print(f"  Std Dev: {temp_stats['descriptive_stats']['std']:.2f}°F")
        print(f"  Skewness: {temp_stats['distribution_shape']['skewness']:.3f}")
        print(f"  Kurtosis: {temp_stats['distribution_shape']['kurtosis']:.3f}")

        # Precipitation patterns
        precip_analysis = {
            'total_days': len(weather_df),
            'dry_days': len(weather_df[weather_df['precipitation'] == 0]),
            'wet_days': len(weather_df[weather_df['precipitation'] > 0]),
            'precipitation_quantiles': weather_df['precipitation'].quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict(),
            'max_precipitation': weather_df['precipitation'].max()
        }

        print("\nPrecipitation Patterns:")
        print(f"  Dry Days: {precip_analysis['dry_days']}/{precip_analysis['total_days']} ({precip_analysis['dry_days']/precip_analysis['total_days']*100:.1f}%)")
        print(f"  Wet Days: {precip_analysis['wet_days']}/{precip_analysis['total_days']} ({precip_analysis['wet_days']/precip_analysis['total_days']*100:.1f}%)")
        print(f"  Max Precipitation: {precip_analysis['max_precipitation']:.2f} inches")

        # Location-based analysis
        location_analysis = weather_df.groupby('location_id').agg({
            'temperature': ['mean', 'std', 'min', 'max'],
            'precipitation': ['sum', 'mean'],
            'humidity': ['mean'],
            'wind_speed': ['mean']
        }).round(2)

        print("\nLocation-based Weather Patterns:")
        for location_id in weather_df['location_id'].unique():
            location_data = weather_df[weather_df['location_id'] == location_id]
            print(f"  Location {location_id}:")
            print(f"    Records: {len(location_data)}")
            print(f"    Avg Temp: {location_data['temperature'].mean():.1f}°F")
            print(f"    Total Precip: {location_data['precipitation'].sum():.2f} inches")

        self.distribution_analysis['weather'] = {
            'temperature': temp_stats,
            'precipitation': precip_analysis,
            'location_patterns': location_analysis.to_dict()
        }

        return self.distribution_analysis['weather']

    def run_parsed_weather_analysis(self):
        """Analyze the more comprehensive parsed weather dataset"""
        print("\n" + "=" * 80)
        print("PARSED WEATHER DATA DEEP ANALYSIS")
        print("=" * 80)

        if 'weather_parsed' not in self.datasets:
            print("Parsed weather data not available")
            return

        parsed_df = self.datasets['weather_parsed']

        # Temporal analysis
        parsed_df['date'] = pd.to_datetime(parsed_df['date'])
        parsed_df['month'] = parsed_df['date'].dt.month
        parsed_df['day_of_year'] = parsed_df['date'].dt.dayofyear

        # Seasonal temperature patterns
        seasonal_analysis = parsed_df.groupby('month').agg({
            'temperature_avg': ['mean', 'std'],
            'temperature_high': ['mean', 'max'],
            'temperature_low': ['mean', 'min'],
            'precipitation': ['sum', 'mean'],
            'humidity': ['mean']
        }).round(2)

        print("Seasonal Temperature Patterns:")
        for month in sorted(parsed_df['month'].unique()):
            month_data = parsed_df[parsed_df['month'] == month]
            if len(month_data) > 0:
                print(f"  Month {month}:")
                print(f"    Records: {len(month_data)}")
                print(f"    Avg Temp: {month_data['temperature_avg'].mean():.1f}°F")
                print(f"    Temp Range: {month_data['temperature_low'].mean():.1f}°F - {month_data['temperature_high'].mean():.1f}°F")

        # Weather condition analysis
        condition_analysis = parsed_df['condition'].value_counts()
        print("\nWeather Condition Distribution:")
        for condition, count in condition_analysis.head(10).items():
            percentage = (count / len(parsed_df)) * 100
            print(f"  {condition}: {count} days ({percentage:.1f}%)")

        # Location communities analysis
        if 'communities' in parsed_df.columns:
            print("\nLocation Communities Analysis:")
            unique_communities = set()
            for communities_str in parsed_df['communities'].dropna():
                try:
                    communities = eval(communities_str) if isinstance(communities_str, str) else communities_str
                    if isinstance(communities, list):
                        unique_communities.update(communities)
                except:
                    pass
            print(f"  Unique Community IDs: {len(unique_communities)}")

        # Data quality assessment
        quality_assessment = {
            'completeness': {},
            'consistency': {},
            'accuracy': {}
        }

        for col in parsed_df.columns:
            missing_pct = (parsed_df[col].isnull().sum() / len(parsed_df)) * 100
            quality_assessment['completeness'][col] = missing_pct

        print("\nData Quality Assessment:")
        print("  Completeness (% missing):")
        for col, missing_pct in quality_assessment['completeness'].items():
            if missing_pct > 0:
                print(f"    {col}: {missing_pct:.1f}%")

        self.distribution_analysis['weather_parsed'] = {
            'seasonal_patterns': seasonal_analysis.to_dict(),
            'condition_distribution': condition_analysis.to_dict(),
            'quality_assessment': quality_assessment
        }

        return self.distribution_analysis['weather_parsed']

    def run_business_data_queries(self):
        """Deep query analysis of business data"""
        print("\n" + "=" * 80)
        print("BUSINESS DATA QUERY ANALYSIS")
        print("=" * 80)

        # Toast Orders Analysis
        if 'toast_orders' in self.datasets:
            orders = self.datasets['toast_orders']
            print(f"Analyzing {len(orders)} Toast orders...")

            order_metrics = self.analyze_order_data_distribution(orders)
            self.query_results['order_analysis'] = order_metrics

        # Comprehensive Leads Analysis
        if 'comprehensive_leads' in self.datasets:
            leads = self.datasets['comprehensive_leads']
            print(f"Analyzing {len(leads)} comprehensive leads...")

            lead_metrics = self.analyze_lead_data_distribution(leads)
            self.query_results['lead_analysis'] = lead_metrics

        # AI Insights Analysis
        if 'ai_insights' in self.datasets:
            insights = self.datasets['ai_insights']
            print("Analyzing AI insights data...")

            insight_metrics = self.analyze_insights_distribution(insights)
            self.query_results['insights_analysis'] = insight_metrics

    def analyze_order_data_distribution(self, orders):
        """Comprehensive analysis of Toast order data patterns"""

        # Extract order features
        order_features = []
        payment_methods = []
        item_categories = []

        for order in orders:
            # Basic order info
            order_info = {
                'order_value': 0,
                'num_checks': len(order.get('checks', [])),
                'num_guests': order.get('numberOfGuests', 0),
                'duration_hours': (order.get('duration', 0) / 3600) if order.get('duration') else 0,
                'has_void': order.get('voided', False),
                'service_type': order.get('source', 'Unknown')
            }

            # Extract payment and item data
            if 'checks' in order:
                for check in order['checks']:
                    order_info['order_value'] += check.get('totalAmount', 0)

                    # Payment methods
                    if 'payments' in check:
                        for payment in check['payments']:
                            payment_methods.append({
                                'type': payment.get('type'),
                                'amount': payment.get('amount', 0),
                                'card_type': payment.get('cardType')
                            })

                    # Menu items
                    if 'selections' in check:
                        for selection in check['selections']:
                            item_categories.append({
                                'item_name': selection.get('displayName'),
                                'price': selection.get('price', 0),
                                'quantity': selection.get('quantity', 1)
                            })

            order_features.append(order_info)

        # Convert to DataFrames for analysis
        orders_df = pd.DataFrame(order_features)
        payments_df = pd.DataFrame(payment_methods)
        items_df = pd.DataFrame(item_categories)

        analysis_results = {}

        if not orders_df.empty:
            # Order value distribution
            analysis_results['order_value_distribution'] = {
                'mean': orders_df['order_value'].mean(),
                'median': orders_df['order_value'].median(),
                'std': orders_df['order_value'].std(),
                'min': orders_df['order_value'].min(),
                'max': orders_df['order_value'].max(),
                'quartiles': orders_df['order_value'].quantile([0.25, 0.5, 0.75]).to_dict()
            }

            # Party size analysis
            analysis_results['party_size_distribution'] = {
                'mean_guests': orders_df['num_guests'].mean(),
                'guest_distribution': orders_df['num_guests'].value_counts().to_dict(),
                'max_party_size': orders_df['num_guests'].max()
            }

            # Service time analysis
            analysis_results['service_time_distribution'] = {
                'mean_duration_hours': orders_df['duration_hours'].mean(),
                'median_duration_hours': orders_df['duration_hours'].median(),
                'duration_quartiles': orders_df['duration_hours'].quantile([0.25, 0.5, 0.75]).to_dict()
            }

            print("Order Analysis Results:")
            print(f"  Average Order Value: ${analysis_results['order_value_distribution']['mean']:.2f}")
            print(f"  Median Order Value: ${analysis_results['order_value_distribution']['median']:.2f}")
            print(f"  Average Party Size: {analysis_results['party_size_distribution']['mean_guests']:.1f}")
            print(f"  Average Service Time: {analysis_results['service_time_distribution']['mean_duration_hours']:.1f} hours")

        if not payments_df.empty:
            # Payment method analysis
            payment_type_dist = payments_df['type'].value_counts()
            analysis_results['payment_methods'] = {
                'type_distribution': payment_type_dist.to_dict(),
                'average_payment_amount': payments_df['amount'].mean()
            }

            print("  Payment Methods:")
            for method, count in payment_type_dist.items():
                percentage = (count / len(payments_df)) * 100
                print(f"    {method}: {count} ({percentage:.1f}%)")

        if not items_df.empty:
            # Menu item analysis
            item_price_stats = items_df['price'].describe()
            analysis_results['menu_items'] = {
                'total_unique_items': items_df['item_name'].nunique(),
                'average_item_price': item_price_stats['mean'],
                'price_range': {'min': item_price_stats['min'], 'max': item_price_stats['max']},
                'top_items': items_df['item_name'].value_counts().head(5).to_dict()
            }

            print(f"  Menu Items: {analysis_results['menu_items']['total_unique_items']} unique items")
            print(f"  Average Item Price: ${analysis_results['menu_items']['average_item_price']:.2f}")

        return analysis_results

    def analyze_lead_data_distribution(self, leads):
        """Comprehensive analysis of lead data distributions"""

        # Convert leads to structured format
        lead_features = []
        for lead in leads:
            lead_info = {
                'industry': lead.get('industry'),
                'location': lead.get('location'),
                'revenue_potential': lead.get('revenue_potential', 0),
                'priority_score': lead.get('priority_score', 0),
                'decision_maker_prob': lead.get('decision_maker_probability', 0),
                'engagement_likelihood': lead.get('engagement_likelihood', 0),
                'num_decision_signals': len(lead.get('decision_signals', [])),
                'num_tags': len(lead.get('tags', []))
            }
            lead_features.append(lead_info)

        leads_df = pd.DataFrame(lead_features)

        analysis_results = {}

        if not leads_df.empty:
            # Industry distribution
            industry_dist = leads_df['industry'].value_counts()
            industry_revenue = leads_df.groupby('industry')['revenue_potential'].agg(['sum', 'mean', 'count'])

            analysis_results['industry_analysis'] = {
                'distribution': industry_dist.to_dict(),
                'revenue_by_industry': industry_revenue.to_dict(),
                'top_industries_by_count': industry_dist.head(5).index.tolist(),
                'top_industries_by_revenue': industry_revenue['sum'].sort_values(ascending=False).head(5).index.tolist()
            }

            # Revenue potential distribution
            revenue_stats = leads_df['revenue_potential'].describe()
            analysis_results['revenue_distribution'] = {
                'total_pipeline': leads_df['revenue_potential'].sum(),
                'mean_deal_size': revenue_stats['mean'],
                'median_deal_size': revenue_stats['50%'],
                'large_deals': len(leads_df[leads_df['revenue_potential'] > 75000]),
                'small_deals': len(leads_df[leads_df['revenue_potential'] < 25000])
            }

            # Lead quality analysis
            analysis_results['lead_quality'] = {
                'high_priority_leads': len(leads_df[leads_df['priority_score'] > 8.0]),
                'confirmed_decision_makers': len(leads_df[leads_df['decision_maker_prob'] > 0.9]),
                'high_engagement_probability': len(leads_df[leads_df['engagement_likelihood'] > 0.8]),
                'average_decision_signals': leads_df['num_decision_signals'].mean()
            }

            # Geographic distribution
            location_dist = leads_df['location'].value_counts()
            analysis_results['geographic_distribution'] = {
                'location_counts': location_dist.to_dict(),
                'top_markets': location_dist.head(5).index.tolist()
            }

            print("Lead Analysis Results:")
            print(f"  Total Pipeline Value: ${analysis_results['revenue_distribution']['total_pipeline']:,}")
            print(f"  Average Deal Size: ${analysis_results['revenue_distribution']['mean_deal_size']:,.0f}")
            print(f"  High Priority Leads: {analysis_results['lead_quality']['high_priority_leads']}")
            print(f"  Top Industry: {analysis_results['industry_analysis']['top_industries_by_count'][0]}")

        return analysis_results

    def analyze_insights_distribution(self, insights):
        """Analyze AI insights data structure and distributions"""

        analysis_results = {
            'summary_metrics': insights.get('summary', {}),
            'industry_analysis': insights.get('industry_analysis', {}),
            'recommendations_count': len(insights.get('recommendations', [])),
            'top_opportunities_count': len(insights.get('top_opportunities', []))
        }

        # Priority distribution analysis
        if 'priority_distribution' in insights:
            priority_dist = insights['priority_distribution']
            total_leads = sum(priority_dist.values())
            analysis_results['priority_analysis'] = {
                'total_leads': total_leads,
                'high_priority_percentage': (priority_dist.get('high', 0) / total_leads * 100) if total_leads > 0 else 0,
                'medium_priority_percentage': (priority_dist.get('medium', 0) / total_leads * 100) if total_leads > 0 else 0,
                'low_priority_percentage': (priority_dist.get('low', 0) / total_leads * 100) if total_leads > 0 else 0
            }

        print("AI Insights Analysis:")
        print(f"  Total Leads: {analysis_results['summary_metrics'].get('total_leads', 0)}")
        print(f"  High Priority: {analysis_results['priority_analysis']['high_priority_percentage']:.1f}%")
        print(f"  Recommendations Generated: {analysis_results['recommendations_count']}")

        return analysis_results

    def detect_outliers(self, data_series):
        """Detect outliers using IQR method"""
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]

        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data_series)) * 100,
            'outlier_values': outliers.tolist(),
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }

    def run_cross_dataset_correlation_analysis(self):
        """Analyze correlations between different datasets"""
        print("\n" + "=" * 80)
        print("CROSS-DATASET CORRELATION ANALYSIS")
        print("=" * 80)

        correlations = {}

        # Weather-Business correlation opportunities
        if 'weather' in self.datasets and 'toast_orders' in self.datasets:
            print("Analyzing Weather-Business correlations...")

            # Time-based correlation analysis
            weather_df = self.datasets['weather']
            orders = self.datasets['toast_orders']

            # Extract order dates and aggregate by date
            order_dates = []
            for order in orders:
                if 'businessDate' in order:
                    date_str = str(order['businessDate'])
                    if len(date_str) == 8:  # Format: YYYYMMDD
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                        order_dates.append(formatted_date)

            if order_dates:
                order_date_counts = pd.Series(order_dates).value_counts()
                correlations['weather_business'] = {
                    'potential_correlation_dates': len(set(order_dates)),
                    'weather_data_dates': len(weather_df['date'].unique()),
                    'overlapping_analysis': 'Temporal correlation analysis possible'
                }

                print(f"  Order Dates Available: {len(order_date_counts)}")
                print(f"  Weather Dates Available: {len(weather_df['date'].unique())}")

        # Lead-Industry correlation
        if 'comprehensive_leads' in self.datasets and 'ai_insights' in self.datasets:
            leads = self.datasets['comprehensive_leads']
            insights = self.datasets['ai_insights']

            industry_correlation = self.analyze_industry_patterns(leads, insights)
            correlations['industry_patterns'] = industry_correlation

        self.correlation_matrices = correlations
        return correlations

    def analyze_industry_patterns(self, leads, insights):
        """Analyze patterns in industry data across datasets"""

        # Extract industry information from leads
        lead_industries = [lead.get('industry') for lead in leads if lead.get('industry')]
        lead_industry_counts = pd.Series(lead_industries).value_counts()

        # Extract industry information from insights
        insights_industries = insights.get('industry_analysis', {}).get('distribution', {})

        # Find common industries
        common_industries = set(lead_industry_counts.index) & set(insights_industries.keys())

        correlation_analysis = {
            'lead_industries': len(lead_industry_counts),
            'insights_industries': len(insights_industries),
            'common_industries': len(common_industries),
            'industry_overlap_percentage': (len(common_industries) / len(set(lead_industry_counts.index) | set(insights_industries.keys()))) * 100
        }

        print(f"Industry Pattern Analysis:")
        print(f"  Industries in Leads: {correlation_analysis['lead_industries']}")
        print(f"  Industries in Insights: {correlation_analysis['insights_industries']}")
        print(f"  Common Industries: {correlation_analysis['common_industries']}")
        print(f"  Overlap Percentage: {correlation_analysis['industry_overlap_percentage']:.1f}%")

        return correlation_analysis

    def generate_query_summary_report(self):
        """Generate comprehensive summary of all query analyses"""
        print("\n" + "=" * 80)
        print("QUERY ANALYSIS SUMMARY REPORT")
        print("=" * 80)

        summary_report = {
            'datasets_analyzed': len(self.datasets),
            'distribution_analyses': len(self.distribution_analysis),
            'query_results': len(self.query_results),
            'correlation_analyses': len(self.correlation_matrices),
            'key_findings': []
        }

        # Key findings from weather analysis
        if 'weather' in self.distribution_analysis:
            weather_findings = self.distribution_analysis['weather']
            summary_report['key_findings'].append({
                'category': 'weather',
                'finding': f"Temperature range: {weather_findings['temperature']['descriptive_stats']['min']:.1f}°F - {weather_findings['temperature']['descriptive_stats']['max']:.1f}°F",
                'insight': f"Precipitation occurs {weather_findings['precipitation']['wet_days']}/{weather_findings['precipitation']['total_days']} days"
            })

        # Key findings from business analysis
        if 'order_analysis' in self.query_results:
            order_findings = self.query_results['order_analysis']
            summary_report['key_findings'].append({
                'category': 'business',
                'finding': f"Average order value: ${order_findings.get('order_value_distribution', {}).get('mean', 0):.2f}",
                'insight': f"Service efficiency: {order_findings.get('service_time_distribution', {}).get('mean_duration_hours', 0):.1f} hours average"
            })

        # Key findings from lead analysis
        if 'lead_analysis' in self.query_results:
            lead_findings = self.query_results['lead_analysis']
            summary_report['key_findings'].append({
                'category': 'sales',
                'finding': f"Total pipeline: ${lead_findings.get('revenue_distribution', {}).get('total_pipeline', 0):,}",
                'insight': f"High-quality leads: {lead_findings.get('lead_quality', {}).get('high_priority_leads', 0)} identified"
            })

        print("Summary Report Generated:")
        print(f"  Datasets Analyzed: {summary_report['datasets_analyzed']}")
        print(f"  Distribution Analyses: {summary_report['distribution_analyses']}")
        print(f"  Business Query Results: {summary_report['query_results']}")

        print("\nKey Findings:")
        for finding in summary_report['key_findings']:
            print(f"  {finding['category'].upper()}: {finding['finding']}")
            print(f"    Insight: {finding['insight']}")

        return summary_report

    def run_comprehensive_analysis(self):
        """Execute complete deep query analysis workflow"""
        print("DEEP QUERY ANALYSIS - COMPREHENSIVE INVESTIGATION")
        print(f"Analysis Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Load all datasets
        self.load_all_datasets()

        # Step 2: Weather data distribution analysis
        self.run_weather_distribution_analysis()

        # Step 3: Parsed weather analysis
        self.run_parsed_weather_analysis()

        # Step 4: Business data queries
        self.run_business_data_queries()

        # Step 5: Cross-dataset correlations
        self.run_cross_dataset_correlation_analysis()

        # Step 6: Generate summary report
        summary_report = self.generate_query_summary_report()

        print("\n" + "=" * 80)
        print("DEEP QUERY ANALYSIS COMPLETE")
        print("=" * 80)
        print("All data distributions analyzed")
        print("Cross-dataset correlations identified")
        print("Business intelligence patterns documented")

        return {
            'datasets': self.datasets,
            'distribution_analysis': self.distribution_analysis,
            'query_results': self.query_results,
            'correlation_matrices': self.correlation_matrices,
            'summary_report': summary_report
        }

if __name__ == "__main__":
    # Initialize and run comprehensive analysis
    analyzer = DeepQueryAnalyzer()
    results = analyzer.run_comprehensive_analysis()

    print("\nAnalysis results available in 'results' dictionary")
    print("Use results['distribution_analysis'] for detailed distributions")
    print("Use results['query_results'] for business intelligence queries")
    print("Use results['correlation_matrices'] for cross-dataset correlations")