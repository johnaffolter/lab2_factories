#!/usr/bin/env python3
"""
Advanced OCR Grading System with Business Domain Classification
Comprehensive document analysis, attachment classification, and intelligent grading
"""

import json
import os
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import spacy
from datetime import datetime
import re
import statistics
from collections import Counter, defaultdict

# Advanced ML imports
try:
    import torch
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available - using rule-based classification")

class BusinessDomain(Enum):
    """Business domain classifications"""
    FINANCE = "finance"
    MARKETING = "marketing"
    EXECUTIVE = "executive"
    MANAGEMENT = "management"
    RESTAURANT = "restaurant"
    OPERATIONS = "operations"
    RETAIL = "retail"
    PROFESSIONAL_SERVICES = "professional_services"
    DATA_ANALYTICS = "data_analytics"
    INTEGRATION = "integration"
    HUMAN_RESOURCES = "human_resources"
    SALES = "sales"
    CUSTOMER_SERVICE = "customer_service"
    LEGAL = "legal"
    COMPLIANCE = "compliance"
    STRATEGY = "strategy"

class DocumentType(Enum):
    """Document type classifications"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    REPORT = "report"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    EMAIL = "email"
    PROPOSAL = "proposal"
    MEMO = "memo"
    FINANCIAL_STATEMENT = "financial_statement"
    MARKETING_MATERIAL = "marketing_material"
    POLICY_DOCUMENT = "policy_document"
    TRAINING_MATERIAL = "training_material"
    PERFORMANCE_REVIEW = "performance_review"
    MENU = "menu"
    INVENTORY = "inventory"
    SCHEDULE = "schedule"
    COMPLIANCE_REPORT = "compliance_report"

class QualityGrade(Enum):
    """Quality grading levels"""
    EXCELLENT = "A+"
    VERY_GOOD = "A"
    GOOD = "B+"
    SATISFACTORY = "B"
    NEEDS_IMPROVEMENT = "C+"
    POOR = "C"
    VERY_POOR = "D"
    FAILED = "F"

@dataclass
class OCRQualityMetrics:
    """OCR quality assessment metrics"""
    text_confidence: float
    character_accuracy: float
    word_accuracy: float
    line_detection_score: float
    layout_preservation: float
    noise_level: float
    resolution_quality: float
    contrast_score: float
    overall_quality: float

@dataclass
class BusinessEntityExtraction:
    """Extracted business entities"""
    organizations: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    monetary_amounts: List[str] = field(default_factory=list)
    addresses: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    product_names: List[str] = field(default_factory=list)
    department_names: List[str] = field(default_factory=list)
    key_metrics: List[str] = field(default_factory=list)

@dataclass
class DomainClassificationResult:
    """Domain classification result"""
    primary_domain: BusinessDomain
    secondary_domains: List[BusinessDomain]
    confidence_scores: Dict[BusinessDomain, float]
    reasoning: str
    key_indicators: List[str]

@dataclass
class DocumentAnalysisResult:
    """Comprehensive document analysis result"""
    document_type: DocumentType
    business_domain: DomainClassificationResult
    ocr_quality: OCRQualityMetrics
    extracted_text: str
    extracted_entities: BusinessEntityExtraction
    content_summary: str
    actionable_insights: List[str]
    quality_grade: QualityGrade
    improvement_suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    confidence_level: float

class DomainClassificationOntology:
    """Comprehensive business domain classification ontology"""

    def __init__(self):
        self.domain_patterns = self._build_domain_patterns()
        self.document_type_patterns = self._build_document_type_patterns()
        self.entity_patterns = self._build_entity_patterns()
        self.quality_criteria = self._build_quality_criteria()

    def _build_domain_patterns(self) -> Dict[BusinessDomain, Dict[str, List[str]]]:
        """Build comprehensive domain classification patterns"""
        return {
            BusinessDomain.FINANCE: {
                "keywords": [
                    "revenue", "profit", "loss", "budget", "forecast", "financial", "accounting",
                    "balance sheet", "income statement", "cash flow", "roi", "investment",
                    "dividend", "equity", "debt", "liability", "asset", "depreciation",
                    "audit", "tax", "billing", "invoice", "payment", "credit", "debit"
                ],
                "entities": ["CFO", "Finance Director", "Accountant", "Financial Analyst"],
                "document_types": ["financial_statement", "invoice", "receipt", "budget"],
                "patterns": [
                    r"\$[\d,]+\.?\d*",  # Currency amounts
                    r"\d+\.\d+%",       # Percentages
                    r"Q[1-4]\s+\d{4}",  # Quarterly references
                    r"FY\s*\d{4}"       # Fiscal year
                ]
            },

            BusinessDomain.MARKETING: {
                "keywords": [
                    "campaign", "brand", "advertising", "promotion", "market", "customer",
                    "engagement", "conversion", "lead", "funnel", "acquisition", "retention",
                    "segmentation", "targeting", "positioning", "awareness", "reach",
                    "impression", "click-through", "social media", "content marketing",
                    "seo", "sem", "analytics", "demographics", "psychographics"
                ],
                "entities": ["CMO", "Marketing Manager", "Brand Manager", "Digital Marketer"],
                "document_types": ["marketing_material", "presentation", "report"],
                "patterns": [
                    r"CTR[\s:]+\d+\.\d+%",     # Click-through rate
                    r"CPM[\s:]+\$\d+",         # Cost per mille
                    r"ROAS[\s:]+\d+\.\d+",     # Return on ad spend
                    r"CAC[\s:]+\$\d+"          # Customer acquisition cost
                ]
            },

            BusinessDomain.EXECUTIVE: {
                "keywords": [
                    "strategic", "vision", "mission", "leadership", "governance", "board",
                    "stakeholder", "shareholder", "executive", "decision", "directive",
                    "policy", "transformation", "innovation", "competitive", "market share",
                    "growth", "expansion", "merger", "acquisition", "partnership",
                    "performance", "kpi", "objectives", "goals", "roadmap"
                ],
                "entities": ["CEO", "COO", "President", "Vice President", "Director"],
                "document_types": ["memo", "proposal", "presentation", "policy_document"],
                "patterns": [
                    r"Board\s+of\s+Directors",
                    r"Executive\s+Summary",
                    r"Strategic\s+Initiative",
                    r"C-level|C-suite"
                ]
            },

            BusinessDomain.MANAGEMENT: {
                "keywords": [
                    "project", "team", "resource", "timeline", "milestone", "deliverable",
                    "scope", "requirement", "specification", "coordination", "planning",
                    "execution", "monitoring", "control", "risk", "quality", "process",
                    "workflow", "efficiency", "productivity", "performance", "review",
                    "feedback", "development", "training", "delegation", "accountability"
                ],
                "entities": ["Project Manager", "Team Lead", "Program Manager", "Supervisor"],
                "document_types": ["report", "memo", "schedule", "performance_review"],
                "patterns": [
                    r"Project\s+\w+",
                    r"Phase\s+[IVX\d]+",
                    r"Sprint\s+\d+",
                    r"Milestone\s+\d+"
                ]
            },

            BusinessDomain.RESTAURANT: {
                "keywords": [
                    "menu", "recipe", "ingredient", "food", "beverage", "service", "kitchen",
                    "dining", "server", "chef", "cook", "hostess", "reservation", "table",
                    "order", "ticket", "pos", "inventory", "supplier", "vendor",
                    "food cost", "labor cost", "covers", "turnover", "upsell", "special"
                ],
                "entities": ["Chef", "Manager", "Server", "Host", "Bartender"],
                "document_types": ["menu", "inventory", "schedule", "receipt"],
                "patterns": [
                    r"\$\d+\.\d{2}",           # Menu prices
                    r"Table\s+\d+",           # Table numbers
                    r"\d+\s+covers",          # Customer count
                    r"Food\s+Cost\s+%"        # Food cost percentage
                ]
            },

            BusinessDomain.OPERATIONS: {
                "keywords": [
                    "operations", "logistics", "supply chain", "procurement", "vendor",
                    "supplier", "distribution", "warehouse", "inventory", "stock",
                    "fulfillment", "shipping", "delivery", "transportation", "route",
                    "optimization", "efficiency", "capacity", "utilization", "maintenance",
                    "facility", "equipment", "safety", "compliance", "quality control"
                ],
                "entities": ["Operations Manager", "Supply Chain Manager", "Logistics Coordinator"],
                "document_types": ["inventory", "schedule", "compliance_report"],
                "patterns": [
                    r"SKU[\s#:]+\w+",
                    r"\d+\s+units",
                    r"Lead\s+Time",
                    r"SOP[\s#:]+\d+"
                ]
            },

            BusinessDomain.RETAIL: {
                "keywords": [
                    "retail", "store", "merchandise", "product", "sku", "barcode",
                    "pricing", "discount", "promotion", "sale", "customer", "shopper",
                    "checkout", "cashier", "transaction", "returns", "exchange",
                    "loyalty", "rewards", "membership", "seasonal", "fashion",
                    "category", "buyer", "visual merchandising", "floor plan"
                ],
                "entities": ["Store Manager", "Buyer", "Merchandiser", "Sales Associate"],
                "document_types": ["inventory", "receipt", "marketing_material"],
                "patterns": [
                    r"SKU[\s#:]+\d+",
                    r"UPC[\s#:]+\d+",
                    r"\d+%\s+off",
                    r"Store\s+#?\d+"
                ]
            },

            BusinessDomain.PROFESSIONAL_SERVICES: {
                "keywords": [
                    "consulting", "advisory", "professional", "expertise", "specialized",
                    "client", "engagement", "project", "deliverable", "methodology",
                    "framework", "best practices", "industry", "sector", "domain",
                    "knowledge", "experience", "certification", "accreditation",
                    "billing", "hourly", "retainer", "scope", "statement of work"
                ],
                "entities": ["Consultant", "Advisor", "Principal", "Partner", "Associate"],
                "document_types": ["proposal", "contract", "report", "presentation"],
                "patterns": [
                    r"\$\d+\s+per\s+hour",
                    r"SOW[\s#:]+\d+",
                    r"Engagement\s+\w+",
                    r"Billable\s+Hours"
                ]
            },

            BusinessDomain.DATA_ANALYTICS: {
                "keywords": [
                    "data", "analytics", "analysis", "metrics", "kpi", "dashboard",
                    "visualization", "report", "insight", "trend", "pattern", "correlation",
                    "regression", "prediction", "model", "algorithm", "machine learning",
                    "ai", "artificial intelligence", "business intelligence", "etl",
                    "pipeline", "warehouse", "lake", "governance", "quality"
                ],
                "entities": ["Data Analyst", "Data Scientist", "BI Developer", "Analytics Manager"],
                "document_types": ["report", "presentation", "spreadsheet"],
                "patterns": [
                    r"R²\s*=\s*\d+\.\d+",
                    r"p-value\s*[<>=]\s*\d+\.\d+",
                    r"accuracy\s*:\s*\d+\.\d+%",
                    r"SQL\s+Query"
                ]
            },

            BusinessDomain.INTEGRATION: {
                "keywords": [
                    "integration", "api", "endpoint", "webhook", "connector", "sync",
                    "data flow", "pipeline", "etl", "middleware", "bridge", "interface",
                    "protocol", "authentication", "authorization", "mapping", "transformation",
                    "validation", "error handling", "monitoring", "logging", "automation",
                    "orchestration", "workflow", "batch", "real-time", "streaming"
                ],
                "entities": ["Integration Engineer", "System Architect", "DevOps Engineer"],
                "document_types": ["report", "memo", "training_material"],
                "patterns": [
                    r"API\s+v?\d+\.\d+",
                    r"HTTP\s+\d{3}",
                    r"JSON\s+payload",
                    r"OAuth\s+2\.0"
                ]
            }
        }

    def _build_document_type_patterns(self) -> Dict[DocumentType, Dict[str, Any]]:
        """Build document type classification patterns"""
        return {
            DocumentType.INVOICE: {
                "keywords": ["invoice", "bill", "amount due", "payment terms", "line item"],
                "required_fields": ["invoice_number", "date", "amount", "vendor"],
                "layout_patterns": ["header_vendor", "line_items", "total_amount"]
            },
            DocumentType.RECEIPT: {
                "keywords": ["receipt", "purchase", "transaction", "total", "change"],
                "required_fields": ["date", "amount", "merchant"],
                "layout_patterns": ["merchant_info", "items", "payment_method"]
            },
            DocumentType.CONTRACT: {
                "keywords": ["agreement", "contract", "terms", "conditions", "party"],
                "required_fields": ["parties", "terms", "signatures"],
                "layout_patterns": ["title", "parties", "terms", "signatures"]
            },
            DocumentType.FINANCIAL_STATEMENT: {
                "keywords": ["balance sheet", "income statement", "cash flow", "assets"],
                "required_fields": ["period", "amounts", "categories"],
                "layout_patterns": ["header", "categories", "amounts", "totals"]
            },
            DocumentType.MENU: {
                "keywords": ["menu", "appetizer", "entree", "dessert", "beverage"],
                "required_fields": ["items", "prices", "descriptions"],
                "layout_patterns": ["categories", "items", "prices"]
            }
        }

    def _build_entity_patterns(self) -> Dict[str, str]:
        """Build entity extraction patterns"""
        return {
            "monetary_amounts": r"\$[\d,]+\.?\d*|USD\s+[\d,]+\.?\d*",
            "phone_numbers": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email_addresses": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "dates": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b",
            "addresses": r"\d+\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)",
            "percentages": r"\d+\.?\d*%",
            "product_codes": r"\b[A-Z]{2,4}[-]?\d{3,6}\b",
            "employee_ids": r"\bEMP[-]?\d{4,6}\b",
            "invoice_numbers": r"\bINV[-]?\d{4,8}\b"
        }

    def _build_quality_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Build quality assessment criteria"""
        return {
            "text_confidence": {
                "excellent": 95,
                "good": 85,
                "satisfactory": 75,
                "poor": 60
            },
            "layout_preservation": {
                "excellent": 90,
                "good": 80,
                "satisfactory": 70,
                "poor": 50
            },
            "entity_extraction": {
                "excellent": 95,
                "good": 85,
                "satisfactory": 75,
                "poor": 60
            }
        }

class AdvancedOCRProcessor:
    """Advanced OCR processing with quality assessment"""

    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Advanced image preprocessing for optimal OCR"""

        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        if image is None:
            raise ValueError("Could not load image")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(denoised)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def extract_text_with_confidence(self, processed_image: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Extract text with detailed confidence metrics"""

        # Get text with confidence data
        data = pytesseract.image_to_data(
            processed_image,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )

        # Extract text
        text = pytesseract.image_to_string(processed_image, config=self.tesseract_config)

        # Calculate confidence metrics
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]

        metrics = {
            "mean_confidence": statistics.mean(confidences) if confidences else 0,
            "median_confidence": statistics.median(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "total_words": len([w for w in data['text'] if w.strip()]),
            "high_confidence_words": len([c for c in confidences if c >= 80]),
            "low_confidence_words": len([c for c in confidences if c < 60])
        }

        return text, metrics

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality for OCR"""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Resolution assessment
        height, width = gray.shape
        resolution_score = min(100, (height * width) / 50000)  # Normalize to 100

        # Contrast assessment
        contrast = gray.std()
        contrast_score = min(100, contrast / 50 * 100)  # Normalize to 100

        # Noise level assessment
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_level = 100 - min(100, laplacian_var / 1000 * 100)

        # Sharpness assessment
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, sharpness / 500 * 100)

        return {
            "resolution_quality": resolution_score,
            "contrast_score": contrast_score,
            "noise_level": noise_level,
            "sharpness_score": sharpness_score
        }

class BusinessDomainClassifier:
    """Advanced business domain classification system"""

    def __init__(self):
        self.ontology = DomainClassificationOntology()
        if TRANSFORMERS_AVAILABLE:
            self.nlp_classifier = self._initialize_transformer_model()
        else:
            self.nlp_classifier = None

        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

    def _initialize_transformer_model(self):
        """Initialize transformer-based classification model"""
        try:
            # Use a pre-trained model for business text classification
            classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            return classifier
        except Exception as e:
            print(f"Could not load transformer model: {e}")
            return None

    def classify_domain(self, text: str, document_metadata: Dict[str, Any] = None) -> DomainClassificationResult:
        """Classify business domain with comprehensive analysis"""

        text_lower = text.lower()
        domain_scores = {}
        key_indicators = []

        # Rule-based classification
        for domain, patterns in self.ontology.domain_patterns.items():
            score = 0
            indicators = []

            # Keyword matching
            keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in text_lower)
            keyword_score = (keyword_matches / len(patterns["keywords"])) * 40
            score += keyword_score

            if keyword_matches > 0:
                indicators.extend([kw for kw in patterns["keywords"] if kw in text_lower][:3])

            # Entity matching
            entity_matches = sum(1 for entity in patterns["entities"] if entity.lower() in text_lower)
            entity_score = (entity_matches / len(patterns["entities"])) * 20
            score += entity_score

            # Pattern matching
            pattern_matches = 0
            for pattern in patterns["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1

            pattern_score = (pattern_matches / len(patterns["patterns"])) * 30
            score += pattern_score

            # Document type alignment
            if document_metadata and "document_type" in document_metadata:
                doc_type = document_metadata["document_type"]
                if doc_type in patterns.get("document_types", []):
                    score += 10
                    indicators.append(f"document_type:{doc_type}")

            domain_scores[domain] = min(100, score)
            if score > 20:
                key_indicators.extend(indicators)

        # Determine primary and secondary domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain = sorted_domains[0][0]
        secondary_domains = [domain for domain, score in sorted_domains[1:4] if score > 30]

        # Generate reasoning
        reasoning = self._generate_classification_reasoning(
            primary_domain, domain_scores[primary_domain], key_indicators
        )

        return DomainClassificationResult(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            confidence_scores=domain_scores,
            reasoning=reasoning,
            key_indicators=list(set(key_indicators))
        )

    def _generate_classification_reasoning(self, domain: BusinessDomain, score: float, indicators: List[str]) -> str:
        """Generate human-readable reasoning for classification"""

        confidence_level = "high" if score > 70 else "medium" if score > 50 else "low"

        reasoning = f"Classified as {domain.value} domain with {confidence_level} confidence ({score:.1f}%). "

        if indicators:
            top_indicators = indicators[:3]
            reasoning += f"Key indicators: {', '.join(top_indicators)}. "

        if score > 80:
            reasoning += "Strong domain-specific vocabulary and patterns detected."
        elif score > 60:
            reasoning += "Clear domain indicators present with some ambiguity."
        else:
            reasoning += "Domain classification based on limited indicators."

        return reasoning

class BusinessEntityExtractor:
    """Extract business entities from text"""

    def __init__(self):
        self.ontology = DomainClassificationOntology()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None

    def extract_entities(self, text: str) -> BusinessEntityExtraction:
        """Extract comprehensive business entities"""

        entities = BusinessEntityExtraction()

        # Pattern-based extraction
        for entity_type, pattern in self.ontology.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)

            if entity_type == "monetary_amounts":
                entities.monetary_amounts.extend(matches)
            elif entity_type == "phone_numbers":
                entities.phone_numbers.extend(matches)
            elif entity_type == "email_addresses":
                entities.email_addresses.extend(matches)
            elif entity_type == "dates":
                entities.dates.extend(matches)
            elif entity_type == "addresses":
                entities.addresses.extend(matches)

        # spaCy-based NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON"]:
                    entities.people.append(ent.text)
                elif ent.label_ in ["ORG"]:
                    entities.organizations.append(ent.text)
                elif ent.label_ in ["DATE"]:
                    entities.dates.append(ent.text)
                elif ent.label_ in ["MONEY"]:
                    entities.monetary_amounts.append(ent.text)

        # Business-specific entity extraction
        entities.department_names.extend(self._extract_departments(text))
        entities.product_names.extend(self._extract_products(text))
        entities.key_metrics.extend(self._extract_metrics(text))

        # Deduplicate and clean
        entities = self._clean_entities(entities)

        return entities

    def _extract_departments(self, text: str) -> List[str]:
        """Extract department names"""
        departments = [
            "finance", "accounting", "marketing", "sales", "hr", "human resources",
            "it", "engineering", "operations", "legal", "compliance", "customer service",
            "procurement", "logistics", "quality assurance", "research", "development"
        ]

        found = []
        text_lower = text.lower()
        for dept in departments:
            if dept in text_lower:
                found.append(dept.title())

        return found

    def _extract_products(self, text: str) -> List[str]:
        """Extract product names using patterns"""
        # Look for capitalized product-like terms
        product_pattern = r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+(?:Pro|Plus|Premium|Standard|Basic|Enterprise))\b'
        matches = re.findall(product_pattern, text)
        return matches[:10]  # Limit to top 10

    def _extract_metrics(self, text: str) -> List[str]:
        """Extract key business metrics"""
        metric_patterns = {
            "ROI": r'ROI\s*:?\s*\d+\.?\d*%?',
            "Revenue": r'revenue\s*:?\s*\$?[\d,]+\.?\d*',
            "Profit": r'profit\s*:?\s*\$?[\d,]+\.?\d*',
            "Growth": r'growth\s*:?\s*\d+\.?\d*%',
            "Conversion": r'conversion\s*(?:rate)?\s*:?\s*\d+\.?\d*%'
        }

        found_metrics = []
        for metric_name, pattern in metric_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_metrics.extend([f"{metric_name}: {match}" for match in matches[:2]])

        return found_metrics

    def _clean_entities(self, entities: BusinessEntityExtraction) -> BusinessEntityExtraction:
        """Clean and deduplicate extracted entities"""

        # Remove duplicates and clean
        entities.organizations = list(set([org.strip() for org in entities.organizations if len(org.strip()) > 2]))
        entities.people = list(set([person.strip() for person in entities.people if len(person.strip()) > 2]))
        entities.dates = list(set([date.strip() for date in entities.dates]))
        entities.monetary_amounts = list(set([amt.strip() for amt in entities.monetary_amounts]))
        entities.addresses = list(set([addr.strip() for addr in entities.addresses]))
        entities.phone_numbers = list(set([phone.strip() for phone in entities.phone_numbers]))
        entities.email_addresses = list(set([email.strip() for email in entities.email_addresses]))
        entities.product_names = list(set([prod.strip() for prod in entities.product_names if len(prod.strip()) > 2]))
        entities.department_names = list(set([dept.strip() for dept in entities.department_names]))
        entities.key_metrics = list(set([metric.strip() for metric in entities.key_metrics]))

        return entities

class ComprehensiveDocumentAnalyzer:
    """Main document analysis orchestrator"""

    def __init__(self):
        self.ocr_processor = AdvancedOCRProcessor()
        self.domain_classifier = BusinessDomainClassifier()
        self.entity_extractor = BusinessEntityExtractor()
        self.ontology = DomainClassificationOntology()

    def analyze_document(self, file_path: str, file_content: bytes = None) -> DocumentAnalysisResult:
        """Perform comprehensive document analysis"""

        start_time = datetime.now()

        try:
            # Process image and extract text
            if file_content:
                # Convert bytes to image
                nparr = np.frombuffer(file_content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(file_path)

            if image is None:
                raise ValueError("Could not process image")

            # Preprocess image
            processed_image = self.ocr_processor.preprocess_image(image)

            # Extract text with confidence
            extracted_text, ocr_metrics = self.ocr_processor.extract_text_with_confidence(processed_image)

            # Assess image quality
            quality_metrics = self.ocr_processor.assess_image_quality(image)

            # Create OCR quality metrics
            ocr_quality = OCRQualityMetrics(
                text_confidence=ocr_metrics["mean_confidence"],
                character_accuracy=self._estimate_character_accuracy(ocr_metrics),
                word_accuracy=self._estimate_word_accuracy(ocr_metrics),
                line_detection_score=self._estimate_line_detection(extracted_text),
                layout_preservation=self._estimate_layout_preservation(extracted_text),
                noise_level=quality_metrics["noise_level"],
                resolution_quality=quality_metrics["resolution_quality"],
                contrast_score=quality_metrics["contrast_score"],
                overall_quality=self._calculate_overall_quality(ocr_metrics, quality_metrics)
            )

            # Classify document type
            document_type = self._classify_document_type(extracted_text)

            # Classify business domain
            business_domain = self.domain_classifier.classify_domain(
                extracted_text,
                {"document_type": document_type}
            )

            # Extract entities
            extracted_entities = self.entity_extractor.extract_entities(extracted_text)

            # Generate content summary
            content_summary = self._generate_content_summary(extracted_text, business_domain, extracted_entities)

            # Generate actionable insights
            actionable_insights = self._generate_actionable_insights(
                extracted_text, business_domain, extracted_entities, document_type
            )

            # Calculate quality grade
            quality_grade = self._calculate_quality_grade(ocr_quality, business_domain.confidence_scores)

            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                ocr_quality, extracted_text, business_domain
            )

            # Calculate confidence level
            confidence_level = self._calculate_overall_confidence(
                ocr_quality, business_domain.confidence_scores, extracted_entities
            )

            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            return DocumentAnalysisResult(
                document_type=document_type,
                business_domain=business_domain,
                ocr_quality=ocr_quality,
                extracted_text=extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                extracted_entities=extracted_entities,
                content_summary=content_summary,
                actionable_insights=actionable_insights,
                quality_grade=quality_grade,
                improvement_suggestions=improvement_suggestions,
                metadata={
                    "file_path": file_path,
                    "processing_timestamp": datetime.now().isoformat(),
                    "image_dimensions": f"{image.shape[1]}x{image.shape[0]}",
                    "text_length": len(extracted_text),
                    "entity_count": self._count_entities(extracted_entities)
                },
                processing_time=processing_time,
                confidence_level=confidence_level
            )

        except Exception as e:
            # Return error result
            return DocumentAnalysisResult(
                document_type=DocumentType.EMAIL,  # Default
                business_domain=DomainClassificationResult(
                    primary_domain=BusinessDomain.OPERATIONS,
                    secondary_domains=[],
                    confidence_scores={},
                    reasoning=f"Analysis failed: {str(e)}",
                    key_indicators=[]
                ),
                ocr_quality=OCRQualityMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0),
                extracted_text="",
                extracted_entities=BusinessEntityExtraction(),
                content_summary=f"Document analysis failed: {str(e)}",
                actionable_insights=[],
                quality_grade=QualityGrade.FAILED,
                improvement_suggestions=["Fix document processing error"],
                metadata={"error": str(e)},
                processing_time=(datetime.now() - start_time).total_seconds(),
                confidence_level=0.0
            )

    def _estimate_character_accuracy(self, ocr_metrics: Dict[str, Any]) -> float:
        """Estimate character-level accuracy"""
        base_confidence = ocr_metrics["mean_confidence"]
        confidence_std = ocr_metrics["confidence_std"]

        # Higher std dev indicates inconsistent recognition
        consistency_penalty = min(20, confidence_std / 2)

        return max(0, base_confidence - consistency_penalty)

    def _estimate_word_accuracy(self, ocr_metrics: Dict[str, Any]) -> float:
        """Estimate word-level accuracy"""
        total_words = ocr_metrics["total_words"]
        high_conf_words = ocr_metrics["high_confidence_words"]

        if total_words == 0:
            return 0

        return (high_conf_words / total_words) * 100

    def _estimate_line_detection(self, text: str) -> float:
        """Estimate line detection quality"""
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        # Heuristic: good line detection should have reasonable line lengths
        avg_line_length = statistics.mean([len(line) for line in non_empty_lines]) if non_empty_lines else 0

        # Optimal line length is around 50-80 characters
        if 30 <= avg_line_length <= 100:
            return 90
        elif 20 <= avg_line_length <= 120:
            return 75
        else:
            return 60

    def _estimate_layout_preservation(self, text: str) -> float:
        """Estimate layout preservation quality"""
        # Check for preserved formatting elements
        formatting_indicators = [
            len(re.findall(r'\n\s*\n', text)),  # Paragraph breaks
            len(re.findall(r'\t', text)),       # Tabs
            len(re.findall(r'^\s+', text, re.MULTILINE)),  # Indentation
        ]

        # Normalize to 0-100 scale
        layout_score = sum(formatting_indicators) * 10
        return min(100, layout_score)

    def _calculate_overall_quality(self, ocr_metrics: Dict[str, Any], quality_metrics: Dict[str, float]) -> float:
        """Calculate overall OCR quality score"""
        weights = {
            "text_confidence": 0.3,
            "resolution_quality": 0.2,
            "contrast_score": 0.2,
            "word_accuracy": 0.2,
            "noise_level": 0.1
        }

        text_confidence = ocr_metrics["mean_confidence"]
        word_accuracy = (ocr_metrics["high_confidence_words"] / max(1, ocr_metrics["total_words"])) * 100

        overall = (
            weights["text_confidence"] * text_confidence +
            weights["resolution_quality"] * quality_metrics["resolution_quality"] +
            weights["contrast_score"] * quality_metrics["contrast_score"] +
            weights["word_accuracy"] * word_accuracy +
            weights["noise_level"] * (100 - quality_metrics["noise_level"])
        )

        return overall

    def _classify_document_type(self, text: str) -> DocumentType:
        """Classify document type based on content"""
        text_lower = text.lower()

        # Simple rule-based classification
        if any(word in text_lower for word in ["invoice", "bill", "amount due"]):
            return DocumentType.INVOICE
        elif any(word in text_lower for word in ["receipt", "total", "purchase"]):
            return DocumentType.RECEIPT
        elif any(word in text_lower for word in ["contract", "agreement", "terms"]):
            return DocumentType.CONTRACT
        elif any(word in text_lower for word in ["balance sheet", "income statement"]):
            return DocumentType.FINANCIAL_STATEMENT
        elif any(word in text_lower for word in ["menu", "appetizer", "entree"]):
            return DocumentType.MENU
        elif any(word in text_lower for word in ["report", "analysis", "summary"]):
            return DocumentType.REPORT
        elif any(word in text_lower for word in ["presentation", "slide", "agenda"]):
            return DocumentType.PRESENTATION
        else:
            return DocumentType.EMAIL  # Default

    def _generate_content_summary(self, text: str, domain: DomainClassificationResult,
                                 entities: BusinessEntityExtraction) -> str:
        """Generate intelligent content summary"""

        summary_parts = []

        # Domain context
        summary_parts.append(f"Document classified as {domain.primary_domain.value} domain content")

        # Key entities
        if entities.organizations:
            summary_parts.append(f"mentions {len(entities.organizations)} organization(s)")

        if entities.monetary_amounts:
            summary_parts.append(f"contains {len(entities.monetary_amounts)} monetary reference(s)")

        if entities.key_metrics:
            summary_parts.append(f"includes {len(entities.key_metrics)} business metric(s)")

        # Content characteristics
        word_count = len(text.split())
        if word_count > 500:
            summary_parts.append("extensive document")
        elif word_count > 200:
            summary_parts.append("moderate-length document")
        else:
            summary_parts.append("brief document")

        return ". ".join(summary_parts).capitalize() + "."

    def _generate_actionable_insights(self, text: str, domain: DomainClassificationResult,
                                     entities: BusinessEntityExtraction, doc_type: DocumentType) -> List[str]:
        """Generate actionable business insights"""

        insights = []

        # Domain-specific insights
        if domain.primary_domain == BusinessDomain.FINANCE:
            if entities.monetary_amounts:
                insights.append("Review financial figures for accuracy and compliance")
            insights.append("Ensure proper financial approval workflows are followed")

        elif domain.primary_domain == BusinessDomain.MARKETING:
            insights.append("Analyze campaign performance metrics if present")
            insights.append("Verify brand consistency and messaging alignment")

        elif domain.primary_domain == BusinessDomain.RESTAURANT:
            if doc_type == DocumentType.MENU:
                insights.append("Review menu pricing strategy and food cost percentages")
            insights.append("Monitor inventory levels and supplier relationships")

        elif domain.primary_domain == BusinessDomain.OPERATIONS:
            insights.append("Assess operational efficiency and process optimization opportunities")
            insights.append("Review compliance with safety and quality standards")

        # General insights based on entities
        if entities.dates:
            insights.append("Track important dates and deadlines mentioned")

        if entities.people:
            insights.append("Follow up with key personnel mentioned in the document")

        # Quality-based insights
        if len(text.split()) < 50:
            insights.append("Document may require additional detail or context")

        return insights[:5]  # Limit to top 5 insights

    def _calculate_quality_grade(self, ocr_quality: OCRQualityMetrics,
                                domain_confidence: Dict[BusinessDomain, float]) -> QualityGrade:
        """Calculate overall quality grade"""

        # Combine OCR quality and classification confidence
        overall_quality = ocr_quality.overall_quality
        best_domain_confidence = max(domain_confidence.values()) if domain_confidence else 50

        combined_score = (overall_quality * 0.7) + (best_domain_confidence * 0.3)

        if combined_score >= 95:
            return QualityGrade.EXCELLENT
        elif combined_score >= 90:
            return QualityGrade.VERY_GOOD
        elif combined_score >= 85:
            return QualityGrade.GOOD
        elif combined_score >= 75:
            return QualityGrade.SATISFACTORY
        elif combined_score >= 65:
            return QualityGrade.NEEDS_IMPROVEMENT
        elif combined_score >= 50:
            return QualityGrade.POOR
        elif combined_score >= 30:
            return QualityGrade.VERY_POOR
        else:
            return QualityGrade.FAILED

    def _generate_improvement_suggestions(self, ocr_quality: OCRQualityMetrics,
                                         text: str, domain: DomainClassificationResult) -> List[str]:
        """Generate specific improvement suggestions"""

        suggestions = []

        # OCR quality improvements
        if ocr_quality.text_confidence < 80:
            suggestions.append("Improve image quality - increase resolution or enhance contrast")

        if ocr_quality.noise_level > 20:
            suggestions.append("Reduce image noise through better scanning or preprocessing")

        if ocr_quality.layout_preservation < 70:
            suggestions.append("Preserve document layout better during scanning")

        # Content improvements
        if len(text.split()) < 100:
            suggestions.append("Document may benefit from additional detail or context")

        # Domain-specific suggestions
        primary_confidence = domain.confidence_scores.get(domain.primary_domain, 0)
        if primary_confidence < 70:
            suggestions.append("Include more domain-specific terminology for clearer classification")

        if not any(domain.confidence_scores[d] > 60 for d in domain.confidence_scores):
            suggestions.append("Document purpose and domain could be more clearly defined")

        return suggestions[:4]  # Limit to top 4 suggestions

    def _calculate_overall_confidence(self, ocr_quality: OCRQualityMetrics,
                                     domain_confidence: Dict[BusinessDomain, float],
                                     entities: BusinessEntityExtraction) -> float:
        """Calculate overall analysis confidence"""

        # OCR confidence component
        ocr_conf = ocr_quality.overall_quality / 100

        # Domain classification confidence
        domain_conf = max(domain_confidence.values()) / 100 if domain_confidence else 0.5

        # Entity extraction confidence (based on number and variety)
        entity_count = self._count_entities(entities)
        entity_conf = min(1.0, entity_count / 10)  # Normalize to 0-1

        # Weighted combination
        overall = (ocr_conf * 0.4) + (domain_conf * 0.4) + (entity_conf * 0.2)

        return overall

    def _count_entities(self, entities: BusinessEntityExtraction) -> int:
        """Count total extracted entities"""
        return (
            len(entities.organizations) + len(entities.people) + len(entities.dates) +
            len(entities.monetary_amounts) + len(entities.addresses) +
            len(entities.phone_numbers) + len(entities.email_addresses) +
            len(entities.product_names) + len(entities.department_names) +
            len(entities.key_metrics)
        )

def main():
    """Demonstrate the OCR grading system"""
    print("🔍 ADVANCED OCR GRADING SYSTEM")
    print("Comprehensive document analysis with business domain classification")
    print("=" * 80)

    # Initialize analyzer
    analyzer = ComprehensiveDocumentAnalyzer()

    # Simulate document analysis (since we don't have actual images)
    sample_documents = [
        {
            "type": "Financial Invoice",
            "text": """INVOICE #INV-2024-001
ABC Corporation
123 Business Street
Invoice Date: March 15, 2024
Amount Due: $2,547.83
Payment Terms: Net 30
Line Items:
- Professional Services: $2,000.00
- Travel Expenses: $347.83
- Tax: $200.00
Total: $2,547.83""",
            "domain": "Finance"
        },
        {
            "type": "Restaurant Menu",
            "text": """DINNER MENU
Appetizers
- Calamari Rings: $12.95
- Caesar Salad: $8.95
Entrees
- Grilled Salmon: $24.95
- Ribeye Steak: $32.95
- Pasta Primavera: $18.95
Desserts
- Chocolate Cake: $7.95
- Tiramisu: $8.95""",
            "domain": "Restaurant"
        },
        {
            "type": "Marketing Report",
            "text": """Q1 MARKETING PERFORMANCE REPORT
Campaign Analytics Dashboard
- Total Impressions: 1,247,583
- Click-Through Rate: 3.2%
- Conversion Rate: 2.1%
- Cost Per Acquisition: $45.67
- Return on Ad Spend: 4.2x
Key Performance Indicators:
- Brand Awareness: +23%
- Customer Engagement: +15%
- Lead Generation: 847 qualified leads
Recommendations: Increase social media budget by 20%""",
            "domain": "Marketing"
        }
    ]

    results = []

    for i, doc in enumerate(sample_documents):
        print(f"\n📄 Analyzing Document {i+1}: {doc['type']}")
        print("-" * 60)

        # Simulate OCR metrics
        import random
        simulated_ocr_quality = OCRQualityMetrics(
            text_confidence=random.uniform(85, 95),
            character_accuracy=random.uniform(90, 98),
            word_accuracy=random.uniform(88, 96),
            line_detection_score=random.uniform(80, 95),
            layout_preservation=random.uniform(75, 90),
            noise_level=random.uniform(5, 15),
            resolution_quality=random.uniform(85, 95),
            contrast_score=random.uniform(80, 90),
            overall_quality=random.uniform(82, 92)
        )

        # Analyze domain
        domain_result = analyzer.domain_classifier.classify_domain(doc["text"])

        # Extract entities
        entities = analyzer.entity_extractor.extract_entities(doc["text"])

        # Create analysis result
        result = DocumentAnalysisResult(
            document_type=analyzer._classify_document_type(doc["text"]),
            business_domain=domain_result,
            ocr_quality=simulated_ocr_quality,
            extracted_text=doc["text"][:200] + "...",
            extracted_entities=entities,
            content_summary=analyzer._generate_content_summary(doc["text"], domain_result, entities),
            actionable_insights=analyzer._generate_actionable_insights(
                doc["text"], domain_result, entities, analyzer._classify_document_type(doc["text"])
            ),
            quality_grade=analyzer._calculate_quality_grade(simulated_ocr_quality, domain_result.confidence_scores),
            improvement_suggestions=analyzer._generate_improvement_suggestions(
                simulated_ocr_quality, doc["text"], domain_result
            ),
            metadata={"simulated": True},
            processing_time=0.85,
            confidence_level=analyzer._calculate_overall_confidence(
                simulated_ocr_quality, domain_result.confidence_scores, entities
            )
        )

        results.append(result)

        # Display results
        print(f"Document Type: {result.document_type.value}")
        print(f"Business Domain: {result.business_domain.primary_domain.value}")
        print(f"Quality Grade: {result.quality_grade.value}")
        print(f"Overall Confidence: {result.confidence_level:.2f}")
        print(f"OCR Quality: {result.ocr_quality.overall_quality:.1f}%")

        print(f"\nKey Entities Found:")
        if result.extracted_entities.organizations:
            print(f"  Organizations: {', '.join(result.extracted_entities.organizations[:3])}")
        if result.extracted_entities.monetary_amounts:
            print(f"  Amounts: {', '.join(result.extracted_entities.monetary_amounts[:3])}")
        if result.extracted_entities.key_metrics:
            print(f"  Metrics: {', '.join(result.extracted_entities.key_metrics[:2])}")

        print(f"\nContent Summary: {result.content_summary}")

        if result.actionable_insights:
            print(f"\nActionable Insights:")
            for insight in result.actionable_insights[:3]:
                print(f"  • {insight}")

    print(f"\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")

    grade_distribution = Counter([r.quality_grade.value for r in results])
    domain_distribution = Counter([r.business_domain.primary_domain.value for r in results])
    avg_confidence = statistics.mean([r.confidence_level for r in results])

    print(f"Documents Analyzed: {len(results)}")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Grade Distribution: {dict(grade_distribution)}")
    print(f"Domain Distribution: {dict(domain_distribution)}")

    print(f"\n✅ OCR Grading System Demonstration Complete")
    print(f"🎯 Features: Multi-domain classification, Entity extraction, Quality assessment")
    print(f"🏢 Supports: Finance, Marketing, Restaurant, Operations, Retail, Professional Services, Data Analytics, Integration")

    return analyzer, results

if __name__ == "__main__":
    main()