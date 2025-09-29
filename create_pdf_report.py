#!/usr/bin/env python3
"""
PDF Report Generator for Comprehensive Data Lineage Analysis
Creates a professional PDF report from the data lineage documentation
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import json
import os
from datetime import datetime

class DataLineagePDFGenerator:
    """Generates comprehensive PDF report from data lineage documentation"""

    def __init__(self, lineage_file: str, output_file: str):
        self.lineage_file = lineage_file
        self.output_file = output_file
        self.doc = SimpleDocTemplate(output_file, pagesize=A4)
        self.styles = getSampleStyleSheet()
        self.story = []

        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )

        self.heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )

        self.heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.blue
        )

        self.code_style = ParagraphStyle(
            'Code',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Courier',
            textColor=colors.darkgreen,
            leftIndent=20,
            spaceAfter=6
        )

    def load_lineage_data(self):
        """Load the lineage documentation JSON"""
        with open(self.lineage_file, 'r') as f:
            return json.load(f)

    def add_title_page(self):
        """Create title page"""
        self.story.append(Spacer(1, 2*inch))

        title = Paragraph("Complete Data Lineage Analysis", self.title_style)
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))

        subtitle = Paragraph("From Airbyte Ingestion to Business Intelligence", self.styles['Heading2'])
        subtitle.alignment = TA_CENTER
        self.story.append(subtitle)
        self.story.append(Spacer(1, 1*inch))

        # Project details
        details = [
            "Lab 2 - Factory Pattern Implementation",
            "MLOps Data Pipeline Analysis",
            f"Generated: {datetime.now().strftime('%B %d, %Y')}",
            "Author: AI Data Analysis System"
        ]

        for detail in details:
            p = Paragraph(detail, self.styles['Normal'])
            p.alignment = TA_CENTER
            self.story.append(p)
            self.story.append(Spacer(1, 12))

        self.story.append(PageBreak())

    def add_executive_summary(self, data):
        """Add executive summary section"""
        summary = data['summary_report']['executive_summary']

        self.story.append(Paragraph("Executive Summary", self.heading1_style))

        # Summary content
        summary_text = f"""
        This comprehensive analysis documents the complete data lineage from external API sources
        through Airbyte ingestion to business intelligence views. The system integrates {summary['data_sources']}
        through {summary['transformation_layers']} to deliver {summary['business_value']}.
        """
        self.story.append(Paragraph(summary_text, self.styles['Normal']))
        self.story.append(Spacer(1, 20))

        # Technical achievements
        self.story.append(Paragraph("Technical Achievements", self.heading2_style))
        achievements = data['summary_report']['technical_achievements']
        for achievement in achievements:
            bullet = f"• {achievement}"
            self.story.append(Paragraph(bullet, self.styles['Normal']))
            self.story.append(Spacer(1, 6))

        self.story.append(Spacer(1, 20))

        # Business impact
        self.story.append(Paragraph("Business Impact Areas", self.heading2_style))
        impacts = data['summary_report']['business_impact']
        for impact in impacts:
            bullet = f"• {impact}"
            self.story.append(Paragraph(bullet, self.styles['Normal']))
            self.story.append(Spacer(1, 6))

        self.story.append(PageBreak())

    def add_airbyte_ingestion_section(self, data):
        """Add Airbyte ingestion documentation"""
        ingestion = data['lineage_documentation']['airbyte_ingestion']

        self.story.append(Paragraph("Airbyte Data Ingestion", self.heading1_style))

        # Weather API ingestion
        weather = ingestion['weather_api_ingestion']
        self.story.append(Paragraph("Weather API Integration", self.heading2_style))

        weather_details = [
            ["Source System", weather['source_system']],
            ["API Endpoint", weather['api_endpoint_pattern']],
            ["Frequency", weather['ingestion_frequency']],
            ["Authentication", weather['authentication']],
            ["Data Format", weather['data_format']]
        ]

        weather_table = Table(weather_details, colWidths=[2*inch, 4*inch])
        weather_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        self.story.append(weather_table)
        self.story.append(Spacer(1, 20))

        # API Response Structure
        self.story.append(Paragraph("Weather API Response Structure", self.heading2_style))
        api_structure = weather['raw_api_response_structure']
        for field, description in api_structure.items():
            field_text = f"<b>{field}</b>: {description}"
            self.story.append(Paragraph(field_text, self.styles['Normal']))
            self.story.append(Spacer(1, 4))

        self.story.append(Spacer(1, 15))

        # Transformation steps
        self.story.append(Paragraph("Airbyte Transformation Pipeline", self.heading2_style))
        steps = weather['airbyte_transformation_steps']
        for step in steps:
            self.story.append(Paragraph(f"• {step}", self.styles['Normal']))
            self.story.append(Spacer(1, 4))

        self.story.append(PageBreak())

        # Business data ingestion
        business = ingestion['business_data_ingestion']
        self.story.append(Paragraph("Business Data Integration", self.heading1_style))

        # Toast POS integration
        toast = business['toast_pos_integration']
        self.story.append(Paragraph("Toast POS API Integration", self.heading2_style))

        toast_text = f"""
        The Toast Point of Sale system provides comprehensive transaction data through their REST API.
        Authentication uses OAuth 2.0 with refresh tokens, and data is ingested both real-time via webhooks
        and through hourly batch synchronization. The API returns complex nested JSON structures that require
        sophisticated flattening strategies to extract meaningful business metrics.
        """
        self.story.append(Paragraph(toast_text, self.styles['Normal']))
        self.story.append(Spacer(1, 15))

        # Key API objects
        self.story.append(Paragraph("Toast API Object Structure", self.heading2_style))
        api_objects = toast['key_api_objects']
        for obj, description in api_objects.items():
            obj_text = f"<b>{obj}</b>: {description}"
            self.story.append(Paragraph(obj_text, self.styles['Normal']))
            self.story.append(Spacer(1, 4))

        self.story.append(PageBreak())

    def add_feature_engineering_section(self, data):
        """Add feature engineering documentation"""
        features = data['lineage_documentation']['feature_engineering']

        self.story.append(Paragraph("Feature Engineering Pipeline", self.heading1_style))

        # Weather features
        weather_features = features['weather_feature_engineering']
        self.story.append(Paragraph("Weather Feature Engineering", self.heading2_style))

        # Temperature features
        temp_features = weather_features['feature_categories']['temperature_features']
        self.story.append(Paragraph("Temperature Feature Derivation", self.heading2_style))

        temp_text = f"""
        <b>Source Transformation:</b> {temp_features['source_transformation']}<br/>
        <b>Business Context:</b> {temp_features['business_context']}
        """
        self.story.append(Paragraph(temp_text, self.styles['Normal']))
        self.story.append(Spacer(1, 10))

        self.story.append(Paragraph("Derived Temperature Features:", self.styles['Heading3']))
        for feature in temp_features['derived_features']:
            self.story.append(Paragraph(f"• {feature}", self.code_style))

        self.story.append(Spacer(1, 15))

        # Composite weather features
        composite = weather_features['feature_categories']['composite_weather_features']
        self.story.append(Paragraph("Composite Weather Intelligence", self.heading2_style))

        # Outdoor activity score
        outdoor_score = composite['outdoor_activity_score']
        self.story.append(Paragraph("Outdoor Activity Score Algorithm", self.styles['Heading3']))

        score_text = f"""
        <b>Formula:</b> {outdoor_score['formula']}<br/>
        <b>Business Use:</b> {outdoor_score['business_use']}
        """
        self.story.append(Paragraph(score_text, self.styles['Normal']))
        self.story.append(Spacer(1, 10))

        self.story.append(Paragraph("Calculation Logic:", self.styles['Normal']))
        for logic in outdoor_score['calculation_logic']:
            self.story.append(Paragraph(f"• {logic}", self.code_style))

        self.story.append(PageBreak())

        # Business features
        business_features = features['business_feature_engineering']
        self.story.append(Paragraph("Business Feature Engineering", self.heading1_style))

        # Order features
        order_features = business_features['order_features']
        self.story.append(Paragraph("Revenue and Operational Metrics", self.heading2_style))

        # Revenue metrics
        self.story.append(Paragraph("Revenue Metrics:", self.styles['Heading3']))
        for metric in order_features['revenue_metrics']:
            self.story.append(Paragraph(f"• {metric}", self.code_style))

        self.story.append(Spacer(1, 10))

        # Operational metrics
        self.story.append(Paragraph("Operational Metrics:", self.styles['Heading3']))
        for metric in order_features['operational_metrics']:
            self.story.append(Paragraph(f"• {metric}", self.code_style))

        self.story.append(PageBreak())

    def add_semantic_views_section(self, data):
        """Add semantic views documentation"""
        views = data['lineage_documentation']['semantic_views']

        self.story.append(Paragraph("Semantic Business Views", self.heading1_style))

        # Weather intelligence view
        weather_view = views['weather_intelligence_view']
        self.story.append(Paragraph("Weather Intelligence View", self.heading2_style))

        weather_purpose = f"""
        <b>Purpose:</b> {weather_view['purpose']}<br/>
        <b>Business Context:</b> {weather_view['business_context']}
        """
        self.story.append(Paragraph(weather_purpose, self.styles['Normal']))
        self.story.append(Spacer(1, 15))

        # Explainable columns
        self.story.append(Paragraph("Explainable Column Definitions", self.heading2_style))
        explainable = weather_view['explainable_columns']
        for column, explanation in explainable.items():
            column_text = f"<b>{column}</b>: {explanation}"
            self.story.append(Paragraph(column_text, self.styles['Normal']))
            self.story.append(Spacer(1, 6))

        self.story.append(Spacer(1, 20))

        # Customer intelligence view
        customer_view = views['customer_intelligence_view']
        self.story.append(Paragraph("Customer Intelligence View", self.heading2_style))

        customer_purpose = f"""
        <b>Purpose:</b> {customer_view['purpose']}<br/>
        <b>Business Context:</b> {customer_view['business_context']}
        """
        self.story.append(Paragraph(customer_purpose, self.styles['Normal']))
        self.story.append(Spacer(1, 15))

        # Customer explainable columns
        self.story.append(Paragraph("Customer Intelligence Column Definitions", self.heading2_style))
        customer_explainable = customer_view['explainable_columns']
        for column, explanation in customer_explainable.items():
            column_text = f"<b>{column}</b>: {explanation}"
            self.story.append(Paragraph(column_text, self.styles['Normal']))
            self.story.append(Spacer(1, 6))

        self.story.append(PageBreak())

    def add_complete_lineage_section(self, data):
        """Add complete lineage overview"""
        lineage = data['lineage_documentation']['complete_lineage']

        self.story.append(Paragraph("Complete Data Lineage Flow", self.heading1_style))

        # Overview
        overview = lineage['data_lineage_overview']
        overview_text = f"""
        <b>Purpose:</b> {overview['purpose']}<br/>
        <b>Scope:</b> {overview['scope']}<br/>
        <b>Methodology:</b> {overview['methodology']}
        """
        self.story.append(Paragraph(overview_text, self.styles['Normal']))
        self.story.append(Spacer(1, 20))

        # End-to-end flow
        flow = lineage['end_to_end_flow']
        self.story.append(Paragraph("Five-Stage Data Pipeline", self.heading2_style))

        stages = [
            ('Stage 1: Ingestion', flow['stage_1_ingestion']),
            ('Stage 2: Processing', flow['stage_2_processing']),
            ('Stage 3: Feature Engineering', flow['stage_3_feature_engineering']),
            ('Stage 4: Semantic Views', flow['stage_4_semantic_views']),
            ('Stage 5: Analytics', flow['stage_5_analytics'])
        ]

        for stage_name, stage_data in stages:
            self.story.append(Paragraph(stage_name, self.styles['Heading3']))
            stage_text = f"""
            <b>Description:</b> {stage_data['description']}<br/>
            <b>Output:</b> {stage_data['output']}
            """
            self.story.append(Paragraph(stage_text, self.styles['Normal']))
            self.story.append(Spacer(1, 10))

        self.story.append(Spacer(1, 20))

        # Quality gates
        quality = lineage['data_quality_lineage']
        self.story.append(Paragraph("Data Quality Assurance", self.heading2_style))

        self.story.append(Paragraph("Quality Gates:", self.styles['Heading3']))
        for gate in quality['quality_gates']:
            self.story.append(Paragraph(f"• {gate}", self.styles['Normal']))
            self.story.append(Spacer(1, 4))

        self.story.append(Spacer(1, 15))

        self.story.append(Paragraph("Quality Metrics:", self.styles['Heading3']))
        for metric in quality['quality_metrics']:
            self.story.append(Paragraph(f"• {metric}", self.styles['Normal']))
            self.story.append(Spacer(1, 4))

        self.story.append(PageBreak())

    def add_transformation_audit_section(self, data):
        """Add transformation audit trail"""
        audit = data['lineage_documentation']['complete_lineage']['transformation_audit_trail']

        self.story.append(Paragraph("Transformation Audit Trail", self.heading1_style))

        # Weather transformations
        weather_trans = audit['weather_transformations']
        self.story.append(Paragraph("Weather Data Transformations", self.heading2_style))

        for field, transformation in weather_trans.items():
            field_text = f"<b>{field}</b>: {transformation}"
            self.story.append(Paragraph(field_text, self.code_style))
            self.story.append(Spacer(1, 8))

        self.story.append(Spacer(1, 20))

        # Business transformations
        business_trans = audit['business_transformations']
        self.story.append(Paragraph("Business Data Transformations", self.heading2_style))

        for field, transformation in business_trans.items():
            field_text = f"<b>{field}</b>: {transformation}"
            self.story.append(Paragraph(field_text, self.code_style))
            self.story.append(Spacer(1, 8))

        self.story.append(PageBreak())

    def add_data_governance_section(self, data):
        """Add data governance information"""
        governance = data['summary_report']['data_governance']

        self.story.append(Paragraph("Data Governance Framework", self.heading1_style))

        governance_text = f"""
        The comprehensive data lineage system implements robust governance capabilities to ensure
        data quality, compliance, and business transparency. This framework supports enterprise-grade
        data management requirements while enabling self-service analytics for business users.
        """
        self.story.append(Paragraph(governance_text, self.styles['Normal']))
        self.story.append(Spacer(1, 20))

        # Governance capabilities
        capabilities = [
            ("Lineage Tracking", governance['lineage_tracking']),
            ("Quality Assurance", governance['quality_assurance']),
            ("Audit Capability", governance['audit_capability']),
            ("Business Definitions", governance['business_definitions'])
        ]

        for capability, description in capabilities:
            self.story.append(Paragraph(capability, self.heading2_style))
            self.story.append(Paragraph(description, self.styles['Normal']))
            self.story.append(Spacer(1, 15))

        self.story.append(PageBreak())

    def add_conclusion(self):
        """Add conclusion section"""
        self.story.append(Paragraph("Conclusion", self.heading1_style))

        conclusion_text = """
        This comprehensive data lineage analysis demonstrates a sophisticated approach to modern data
        engineering that bridges the gap between technical implementation and business value. The system
        successfully integrates multiple external data sources through Airbyte, applies intelligent
        feature engineering, and delivers business-friendly semantic views that enable self-service analytics.

        The five-stage pipeline (Ingestion → Processing → Feature Engineering → Semantic Views → Analytics)
        provides a scalable foundation for enterprise data operations. Quality gates and audit trails ensure
        data integrity, while explainable columns and business context make the system accessible to
        non-technical stakeholders.

        Key achievements include weather-business correlation analysis, customer intelligence scoring,
        and operational optimization capabilities. The complete field-level lineage documentation supports
        compliance requirements while enabling confident decision-making based on trusted data.

        This implementation serves as a template for modern data engineering practices that prioritize
        both technical excellence and business usability.
        """

        self.story.append(Paragraph(conclusion_text, self.styles['Normal']))
        self.story.append(Spacer(1, 30))

        # Footer
        footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        footer = Paragraph(footer_text, self.styles['Normal'])
        footer.alignment = TA_CENTER
        self.story.append(footer)

    def generate_pdf(self):
        """Generate the complete PDF report"""
        print("Loading data lineage documentation...")
        data = self.load_lineage_data()

        print("Creating title page...")
        self.add_title_page()

        print("Adding executive summary...")
        self.add_executive_summary(data)

        print("Adding Airbyte ingestion section...")
        self.add_airbyte_ingestion_section(data)

        print("Adding feature engineering section...")
        self.add_feature_engineering_section(data)

        print("Adding semantic views section...")
        self.add_semantic_views_section(data)

        print("Adding complete lineage section...")
        self.add_complete_lineage_section(data)

        print("Adding transformation audit section...")
        self.add_transformation_audit_section(data)

        print("Adding data governance section...")
        self.add_data_governance_section(data)

        print("Adding conclusion...")
        self.add_conclusion()

        print(f"Building PDF document: {self.output_file}")
        self.doc.build(self.story)

        print(f"PDF report generated successfully: {self.output_file}")
        return self.output_file

def main():
    """Main execution function"""
    lineage_file = "/Users/johnaffolter/lab_2_homework/lab2_factories/data_lineage_documentation.json"
    output_file = "/Users/johnaffolter/lab_2_homework/lab2_factories/Complete_Data_Lineage_Analysis_Report.pdf"

    if not os.path.exists(lineage_file):
        print(f"Error: Lineage documentation file not found: {lineage_file}")
        return

    generator = DataLineagePDFGenerator(lineage_file, output_file)

    try:
        result_file = generator.generate_pdf()
        print(f"\n✓ PDF report created successfully!")
        print(f"📄 File location: {result_file}")
        print(f"📊 Report contains comprehensive data lineage analysis")
        print(f"🔍 Includes API mappings, feature engineering, and business intelligence views")

    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("Ensure reportlab is installed: pip install reportlab")

if __name__ == "__main__":
    main()