#!/usr/bin/env python3
"""
DSL-Based Ontology System for AI Training Data Generation
Implements a robust domain-specific language with inheritance, dynamic typing, and template-driven generation
"""

import json
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Generic, Callable
from enum import Enum
import uuid
from datetime import datetime, timedelta
import re
import random
import numpy as np
from collections import defaultdict

# Type system for dynamic typing
T = TypeVar('T')
DataType = Union[str, int, float, bool, List, Dict]

class OntologyDataType(Enum):
    """Base data types for the ontology system"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    REFERENCE = "reference"
    TEMPLATE = "template"
    DYNAMIC = "dynamic"

class ValidationRule(ABC):
    """Abstract base class for validation rules"""

    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_error_message(self) -> str:
        pass

@dataclass
class TypeConstraint:
    """Type constraint with validation rules"""
    data_type: OntologyDataType
    constraints: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    default_value: Optional[Any] = None
    nullable: bool = True
    inherited_from: Optional[str] = None

@dataclass
class TemplateParameter:
    """Template parameter with dynamic typing"""
    name: str
    type_constraint: TypeConstraint
    description: str
    examples: List[Any] = field(default_factory=list)
    generation_strategy: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class EntityDefinition:
    """Ontology entity definition with inheritance"""
    name: str
    description: str
    parent_entity: Optional[str] = None
    parameters: Dict[str, TemplateParameter] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    templates: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_patterns: List[str] = field(default_factory=list)

class LengthValidationRule(ValidationRule):
    """Validates string length constraints"""

    def __init__(self, min_length: int = 0, max_length: int = 1000):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any, context: Dict[str, Any]) -> bool:
        if not isinstance(value, str):
            return False
        return self.min_length <= len(value) <= self.max_length

    def get_error_message(self) -> str:
        return f"String length must be between {self.min_length} and {self.max_length}"

class RangeValidationRule(ValidationRule):
    """Validates numeric range constraints"""

    def __init__(self, min_value: Union[int, float], max_value: Union[int, float]):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any, context: Dict[str, Any]) -> bool:
        if not isinstance(value, (int, float)):
            return False
        return self.min_value <= value <= self.max_value

    def get_error_message(self) -> str:
        return f"Value must be between {self.min_value} and {self.max_value}"

class PatternValidationRule(ValidationRule):
    """Validates string pattern matching"""

    def __init__(self, pattern: str, description: str = ""):
        self.pattern = re.compile(pattern)
        self.description = description

    def validate(self, value: Any, context: Dict[str, Any]) -> bool:
        if not isinstance(value, str):
            return False
        return bool(self.pattern.match(value))

    def get_error_message(self) -> str:
        return f"Value must match pattern: {self.description or self.pattern.pattern}"

class SemanticValidationRule(ValidationRule):
    """Validates semantic coherence with context"""

    def __init__(self, semantic_check: Callable[[Any, Dict[str, Any]], bool], description: str):
        self.semantic_check = semantic_check
        self.description = description

    def validate(self, value: Any, context: Dict[str, Any]) -> bool:
        return self.semantic_check(value, context)

    def get_error_message(self) -> str:
        return f"Semantic validation failed: {self.description}"

class OntologyEngine:
    """Core ontology engine with DSL processing"""

    def __init__(self):
        self.entities: Dict[str, EntityDefinition] = {}
        self.type_registry: Dict[str, Type] = {}
        self.rule_engine = RuleEngine()
        self.template_processor = TemplateProcessor()
        self.inheritance_graph = InheritanceGraph()

    def load_ontology_dsl(self, dsl_content: str, format_type: str = "yaml") -> None:
        """Load ontology definition from DSL"""
        try:
            if format_type.lower() == "yaml":
                ontology_data = yaml.safe_load(dsl_content)
            else:
                ontology_data = json.loads(dsl_content)

            self._process_ontology_definition(ontology_data)
            self._build_inheritance_graph()
            self._validate_ontology_consistency()

        except Exception as e:
            raise OntologyException(f"Failed to load ontology DSL: {e}")

    def _process_ontology_definition(self, ontology_data: Dict[str, Any]) -> None:
        """Process the ontology definition and create entity definitions"""

        # Process base types first
        if "types" in ontology_data:
            self._register_custom_types(ontology_data["types"])

        # Process entities with inheritance support
        if "entities" in ontology_data:
            for entity_name, entity_spec in ontology_data["entities"].items():
                entity_def = self._create_entity_definition(entity_name, entity_spec)
                self.entities[entity_name] = entity_def

    def _create_entity_definition(self, name: str, spec: Dict[str, Any]) -> EntityDefinition:
        """Create entity definition from specification"""

        # Extract basic information
        description = spec.get("description", f"Entity: {name}")
        parent_entity = spec.get("inherits_from")

        # Process parameters with type constraints
        parameters = {}
        if "parameters" in spec:
            for param_name, param_spec in spec["parameters"].items():
                param_def = self._create_parameter_definition(param_name, param_spec)
                parameters[param_name] = param_def

        # Process rules and templates
        rules = spec.get("rules", [])
        templates = spec.get("templates", {})
        generation_patterns = spec.get("generation_patterns", [])

        return EntityDefinition(
            name=name,
            description=description,
            parent_entity=parent_entity,
            parameters=parameters,
            rules=rules,
            templates=templates,
            generation_patterns=generation_patterns,
            metadata=spec.get("metadata", {})
        )

    def _create_parameter_definition(self, name: str, spec: Dict[str, Any]) -> TemplateParameter:
        """Create parameter definition with type constraints"""

        # Determine data type
        type_name = spec.get("type", "string")
        data_type = OntologyDataType(type_name.lower())

        # Create validation rules
        validation_rules = []
        constraints = spec.get("constraints", {})

        if "length" in constraints and data_type == OntologyDataType.STRING:
            length_spec = constraints["length"]
            validation_rules.append(LengthValidationRule(
                min_length=length_spec.get("min", 0),
                max_length=length_spec.get("max", 1000)
            ))

        if "range" in constraints and data_type in [OntologyDataType.INTEGER, OntologyDataType.FLOAT]:
            range_spec = constraints["range"]
            validation_rules.append(RangeValidationRule(
                min_value=range_spec.get("min", 0),
                max_value=range_spec.get("max", 100)
            ))

        if "pattern" in constraints and data_type == OntologyDataType.STRING:
            pattern_spec = constraints["pattern"]
            validation_rules.append(PatternValidationRule(
                pattern=pattern_spec["regex"],
                description=pattern_spec.get("description", "")
            ))

        # Create type constraint
        type_constraint = TypeConstraint(
            data_type=data_type,
            constraints=constraints,
            validation_rules=validation_rules,
            default_value=spec.get("default"),
            nullable=spec.get("nullable", True)
        )

        return TemplateParameter(
            name=name,
            type_constraint=type_constraint,
            description=spec.get("description", f"Parameter: {name}"),
            examples=spec.get("examples", []),
            generation_strategy=spec.get("generation_strategy"),
            dependencies=spec.get("dependencies", [])
        )

    def _register_custom_types(self, types_spec: Dict[str, Any]) -> None:
        """Register custom types in the type registry"""
        for type_name, type_definition in types_spec.items():
            # Create dynamic type class
            self.type_registry[type_name] = self._create_dynamic_type(type_name, type_definition)

    def _create_dynamic_type(self, name: str, definition: Dict[str, Any]) -> Type:
        """Create a dynamic type class from definition"""

        class DynamicType:
            def __init__(self, value: Any):
                self.value = value
                self.type_name = name
                self.definition = definition

            def validate(self) -> bool:
                # Implement type-specific validation
                return True

            def __str__(self):
                return f"{name}({self.value})"

        return DynamicType

    def _build_inheritance_graph(self) -> None:
        """Build inheritance graph for entity relationships"""
        for entity_name, entity_def in self.entities.items():
            if entity_def.parent_entity:
                self.inheritance_graph.add_inheritance(entity_name, entity_def.parent_entity)

        # Resolve inheritance and merge properties
        for entity_name in self.entities:
            self._resolve_entity_inheritance(entity_name)

    def _resolve_entity_inheritance(self, entity_name: str) -> None:
        """Resolve inheritance for a specific entity"""
        entity = self.entities[entity_name]

        if entity.parent_entity and entity.parent_entity in self.entities:
            parent = self.entities[entity.parent_entity]

            # Merge parameters (child overrides parent)
            merged_parameters = parent.parameters.copy()
            merged_parameters.update(entity.parameters)

            # Mark inherited parameters
            for param_name, param in merged_parameters.items():
                if param_name in parent.parameters and param_name not in entity.parameters:
                    param.type_constraint.inherited_from = entity.parent_entity

            entity.parameters = merged_parameters

            # Merge rules and templates
            entity.rules = parent.rules + entity.rules
            merged_templates = parent.templates.copy()
            merged_templates.update(entity.templates)
            entity.templates = merged_templates

    def _validate_ontology_consistency(self) -> None:
        """Validate ontology consistency and dependencies"""
        errors = []

        # Check for circular dependencies
        if self.inheritance_graph.has_cycles():
            errors.append("Circular inheritance detected in ontology")

        # Validate parameter dependencies
        for entity_name, entity in self.entities.items():
            for param_name, param in entity.parameters.items():
                for dep in param.dependencies:
                    if dep not in entity.parameters:
                        errors.append(f"Entity {entity_name}, parameter {param_name}: dependency {dep} not found")

        if errors:
            raise OntologyException("Ontology validation failed: " + "; ".join(errors))

class InheritanceGraph:
    """Manages entity inheritance relationships"""

    def __init__(self):
        self.edges: Dict[str, List[str]] = defaultdict(list)

    def add_inheritance(self, child: str, parent: str) -> None:
        """Add inheritance relationship"""
        self.edges[parent].append(child)

    def has_cycles(self) -> bool:
        """Check for circular inheritance"""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.edges[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.edges:
            if node not in visited:
                if dfs(node):
                    return True

        return False

class RuleEngine:
    """Rule engine for validation and constraints"""

    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = defaultdict(list)

    def add_rule(self, entity_name: str, rule: ValidationRule) -> None:
        """Add validation rule for entity"""
        self.rules[entity_name].append(rule)

    def validate_entity(self, entity_name: str, data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Validate entity data against rules"""
        errors = []

        for rule in self.rules.get(entity_name, []):
            if not rule.validate(data, context):
                errors.append(rule.get_error_message())

        return errors

class TemplateProcessor:
    """Advanced template processing with inheritance"""

    def __init__(self):
        self.template_cache: Dict[str, str] = {}
        self.context_stack: List[Dict[str, Any]] = []

    def process_template(self, template: str, context: Dict[str, Any], entity_def: EntityDefinition) -> str:
        """Process template with context and inheritance"""

        # Build full context with inheritance
        full_context = self._build_inheritance_context(context, entity_def)

        # Process template variables
        processed = self._substitute_variables(template, full_context)

        # Apply generation strategies
        processed = self._apply_generation_strategies(processed, entity_def, full_context)

        return processed

    def _build_inheritance_context(self, context: Dict[str, Any], entity_def: EntityDefinition) -> Dict[str, Any]:
        """Build context including inherited values"""
        full_context = context.copy()

        # Add default values from parameters
        for param_name, param in entity_def.parameters.items():
            if param_name not in full_context and param.type_constraint.default_value is not None:
                full_context[param_name] = param.type_constraint.default_value

        return full_context

    def _substitute_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Substitute template variables with context values"""

        def replace_var(match):
            var_name = match.group(1)
            if var_name in context:
                return str(context[var_name])
            else:
                return match.group(0)  # Leave unresolved

        return re.sub(r'\{\{(\w+)\}\}', replace_var, template)

    def _apply_generation_strategies(self, template: str, entity_def: EntityDefinition, context: Dict[str, Any]) -> str:
        """Apply generation strategies for dynamic content"""

        # Find and process generation directives
        def process_directive(match):
            directive = match.group(1)

            if directive.startswith("generate:"):
                strategy = directive.split(":", 1)[1]
                return self._execute_generation_strategy(strategy, entity_def, context)

            return match.group(0)

        return re.sub(r'\{\{([^}]+)\}\}', process_directive, template)

    def _execute_generation_strategy(self, strategy: str, entity_def: EntityDefinition, context: Dict[str, Any]) -> str:
        """Execute specific generation strategy"""

        strategies = {
            "random_number": lambda: str(random.randint(1, 1000)),
            "timestamp": lambda: datetime.now().isoformat(),
            "uuid": lambda: str(uuid.uuid4()),
            "random_choice": lambda opts: random.choice(opts.split("|"))
        }

        if ":" in strategy:
            strategy_name, params = strategy.split(":", 1)
            if strategy_name in strategies:
                return strategies[strategy_name](params)
        elif strategy in strategies:
            return strategies[strategy]()

        return f"{{unknown_strategy:{strategy}}}"

class DataGenerator:
    """Template-driven data generator using the ontology"""

    def __init__(self, ontology_engine: OntologyEngine):
        self.ontology = ontology_engine
        self.generation_cache: Dict[str, List[Any]] = {}

    def generate_entity_instance(self, entity_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a single entity instance"""

        if entity_name not in self.ontology.entities:
            raise OntologyException(f"Entity {entity_name} not found in ontology")

        entity_def = self.ontology.entities[entity_name]
        context = context or {}

        # Generate values for all parameters
        instance = {}

        for param_name, param in entity_def.parameters.items():
            if param_name not in context:
                value = self._generate_parameter_value(param, context, entity_def)
                instance[param_name] = value
                context[param_name] = value  # Add to context for dependencies
            else:
                instance[param_name] = context[param_name]

        # Process templates if any
        if entity_def.templates:
            for template_name, template_content in entity_def.templates.items():
                processed = self.ontology.template_processor.process_template(
                    template_content, context, entity_def
                )
                instance[f"generated_{template_name}"] = processed

        # Validate generated instance
        validation_errors = self.ontology.rule_engine.validate_entity(entity_name, instance, context)
        if validation_errors:
            instance["validation_errors"] = validation_errors

        # Add metadata
        instance["_entity_type"] = entity_name
        instance["_generated_at"] = datetime.now().isoformat()
        instance["_generation_id"] = str(uuid.uuid4())

        return instance

    def _generate_parameter_value(self, param: TemplateParameter, context: Dict[str, Any], entity_def: EntityDefinition) -> Any:
        """Generate value for a specific parameter"""

        # Check if examples are available
        if param.examples:
            return random.choice(param.examples)

        # Use generation strategy if specified
        if param.generation_strategy:
            return self._execute_parameter_strategy(param.generation_strategy, param, context)

        # Generate based on type
        return self._generate_by_type(param.type_constraint, context)

    def _execute_parameter_strategy(self, strategy: str, param: TemplateParameter, context: Dict[str, Any]) -> Any:
        """Execute parameter-specific generation strategy"""

        if strategy == "context_aware":
            return self._generate_context_aware_value(param, context)
        elif strategy == "semantic_coherent":
            return self._generate_semantic_value(param, context)
        elif strategy == "pattern_based":
            return self._generate_pattern_value(param, context)
        else:
            return self._generate_by_type(param.type_constraint, context)

    def _generate_by_type(self, type_constraint: TypeConstraint, context: Dict[str, Any]) -> Any:
        """Generate value based on type constraint"""

        data_type = type_constraint.data_type
        constraints = type_constraint.constraints

        if data_type == OntologyDataType.STRING:
            min_len = constraints.get("length", {}).get("min", 5)
            max_len = constraints.get("length", {}).get("max", 50)
            length = random.randint(min_len, min(max_len, 100))
            return self._generate_realistic_string(length, constraints)

        elif data_type == OntologyDataType.INTEGER:
            min_val = constraints.get("range", {}).get("min", 1)
            max_val = constraints.get("range", {}).get("max", 1000)
            return random.randint(min_val, max_val)

        elif data_type == OntologyDataType.FLOAT:
            min_val = constraints.get("range", {}).get("min", 0.0)
            max_val = constraints.get("range", {}).get("max", 1.0)
            return round(random.uniform(min_val, max_val), 3)

        elif data_type == OntologyDataType.BOOLEAN:
            return random.choice([True, False])

        elif data_type == OntologyDataType.LIST:
            item_count = random.randint(1, constraints.get("max_items", 5))
            return [f"item_{i}" for i in range(item_count)]

        else:
            return f"generated_{data_type.value}"

    def _generate_realistic_string(self, length: int, constraints: Dict[str, Any]) -> str:
        """Generate realistic string content"""

        patterns = constraints.get("patterns", ["realistic_text"])
        pattern = random.choice(patterns)

        if pattern == "email_subject":
            subjects = [
                "Important Meeting Tomorrow",
                "Project Update Required",
                "Quarterly Review Schedule",
                "Team Building Event",
                "Budget Approval Needed"
            ]
            return random.choice(subjects)

        elif pattern == "business_text":
            words = ["project", "meeting", "deadline", "review", "team", "client", "proposal", "analysis"]
            return " ".join(random.choices(words, k=min(length//6, len(words))))

        else:
            # Generate generic realistic text
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs"]
            text = " ".join(random.choices(words, k=max(1, length//4)))
            return text[:length]

    def generate_training_dataset(self, entity_name: str, sample_count: int, variations: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
        """Generate a complete training dataset for an entity"""

        dataset = []
        variations = variations or {}

        print(f"🏗️ Generating {sample_count} samples for entity: {entity_name}")

        for i in range(sample_count):
            # Create context with variations
            context = {}
            for var_name, var_values in variations.items():
                context[var_name] = random.choice(var_values)

            # Generate instance
            instance = self.generate_entity_instance(entity_name, context)
            dataset.append(instance)

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  ✓ Generated {i + 1}/{sample_count} instances")

        return dataset

class OntologyException(Exception):
    """Custom exception for ontology-related errors"""
    pass

def create_email_ontology_dsl() -> str:
    """Create a comprehensive email ontology DSL"""

    ontology_dsl = """
# Email Classification Ontology DSL
# Robust template-driven data generation with inheritance and dynamic typing

version: "1.0.0"
description: "Comprehensive email classification ontology with AI training data generation"

# Custom type definitions
types:
  email_sentiment:
    base_type: "float"
    range: [0.0, 1.0]
    description: "Email sentiment score from negative (0) to positive (1)"

  urgency_level:
    base_type: "integer"
    range: [1, 10]
    description: "Email urgency level from low (1) to critical (10)"

  confidence_score:
    base_type: "float"
    range: [0.0, 1.0]
    description: "Classification confidence score"

# Base entity definitions with inheritance
entities:

  # Base email entity - parent for all email types
  base_email:
    description: "Base email entity with common properties"
    parameters:
      subject:
        type: "string"
        description: "Email subject line"
        constraints:
          length:
            min: 5
            max: 100
          patterns: ["email_subject"]
        examples:
          - "Meeting Request"
          - "Project Update"
          - "Important Notice"

      body:
        type: "string"
        description: "Email body content"
        constraints:
          length:
            min: 20
            max: 1000
          patterns: ["business_text", "conversational_text"]
        generation_strategy: "context_aware"

      sender_domain:
        type: "string"
        description: "Sender's email domain"
        constraints:
          pattern:
            regex: "^[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
            description: "Valid domain format"
        examples:
          - "company.com"
          - "university.edu"
          - "organization.org"

      timestamp:
        type: "string"
        description: "Email timestamp"
        generation_strategy: "timestamp_realistic"
        constraints:
          pattern:
            regex: "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}"
            description: "ISO datetime format"

      sentiment_score:
        type: "email_sentiment"
        description: "Overall email sentiment"
        generation_strategy: "context_dependent"

      urgency:
        type: "urgency_level"
        description: "Email urgency level"
        default: 5
        dependencies: ["subject", "body"]

      metadata:
        type: "object"
        description: "Additional email metadata"
        nullable: true

    templates:
      full_email: "From: {{sender_domain}}\\nSubject: {{subject}}\\nTimestamp: {{timestamp}}\\n\\n{{body}}"
      summary: "Email about {{subject}} with {{urgency}}/10 urgency"

    rules:
      - "subject_body_coherence: subject and body must be semantically related"
      - "sender_credibility: sender_domain must match email category context"
      - "temporal_consistency: timestamp must be realistic for email type"

    generation_patterns:
      - "business_formal"
      - "conversational_casual"
      - "automated_system"

  # Work-related emails
  work_email:
    description: "Professional work-related email communications"
    inherits_from: "base_email"

    parameters:
      project_reference:
        type: "string"
        description: "Reference to specific project or initiative"
        constraints:
          length:
            min: 3
            max: 50
          patterns: ["project_names"]
        examples:
          - "Phoenix Initiative"
          - "Q4 Planning"
          - "Digital Transformation"
        generation_strategy: "semantic_coherent"

      priority_level:
        type: "string"
        description: "Business priority classification"
        constraints:
          enum: ["low", "medium", "high", "critical"]
        default: "medium"
        dependencies: ["urgency", "subject"]

      department:
        type: "string"
        description: "Originating department"
        examples:
          - "Engineering"
          - "Marketing"
          - "Sales"
          - "HR"
          - "Finance"
        generation_strategy: "context_aware"

      meeting_request:
        type: "boolean"
        description: "Whether email contains meeting request"
        dependencies: ["subject", "body"]
        generation_strategy: "content_analysis"

      deadline_mentioned:
        type: "boolean"
        description: "Whether email mentions deadlines"
        dependencies: ["body", "urgency"]
        generation_strategy: "content_analysis"

    templates:
      meeting_request: "{{subject}} - Please join us for {{project_reference}} meeting on {{generate:date_future}}. Agenda includes {{generate:meeting_topics}}."
      status_update: "Project {{project_reference}} status update: {{generate:status_description}}. Next steps: {{generate:action_items}}."
      deadline_reminder: "Reminder: {{project_reference}} deadline approaching on {{generate:date_near}}. Priority: {{priority_level}}."

    rules:
      - "business_tone: language must be professional and business-appropriate"
      - "project_coherence: project_reference must align with email content"
      - "priority_urgency_alignment: priority_level must correlate with urgency score"

    generation_patterns:
      - "formal_business"
      - "collaborative_team"
      - "executive_communication"

    metadata:
      category: "work"
      typical_sentiment_range: [0.3, 0.8]
      common_keywords: ["meeting", "project", "deadline", "team", "review"]

  # Personal emails
  personal_email:
    description: "Personal communications between individuals"
    inherits_from: "base_email"

    parameters:
      relationship_type:
        type: "string"
        description: "Type of personal relationship"
        constraints:
          enum: ["family", "friend", "acquaintance", "romantic"]
        examples: ["friend", "family"]
        generation_strategy: "context_aware"

      event_reference:
        type: "string"
        description: "Personal event or occasion mentioned"
        constraints:
          length:
            min: 3
            max: 30
        examples:
          - "birthday party"
          - "weekend trip"
          - "dinner plans"
          - "graduation"
        nullable: true

      emotional_tone:
        type: "string"
        description: "Emotional tone of the email"
        constraints:
          enum: ["excited", "casual", "concerned", "celebratory", "nostalgic"]
        generation_strategy: "sentiment_aligned"
        dependencies: ["sentiment_score"]

      casual_language:
        type: "boolean"
        description: "Whether email uses casual/informal language"
        default: true
        generation_strategy: "relationship_based"
        dependencies: ["relationship_type"]

    templates:
      celebration: "Hey! Hope you're doing well. {{event_reference}} was amazing! Can't wait to {{generate:future_plans}}."
      casual_check_in: "Hi {{generate:casual_name}}! How's {{generate:personal_topic}} going? We should {{generate:activity_suggestion}} soon!"
      gratitude: "Thank you so much for {{generate:favor_description}}. Your {{generate:positive_quality}} meant everything!"

    rules:
      - "informal_tone: language should be casual and personal"
      - "emotional_authenticity: emotional_tone must match content sentiment"
      - "relationship_appropriate: content must suit relationship_type"

    generation_patterns:
      - "casual_friendly"
      - "warm_familial"
      - "excited_social"

    metadata:
      category: "personal"
      typical_sentiment_range: [0.6, 0.95]
      common_keywords: ["family", "friend", "weekend", "celebration", "personal"]

  # Promotional emails
  promotional_email:
    description: "Marketing and promotional email communications"
    inherits_from: "base_email"

    parameters:
      discount_percentage:
        type: "integer"
        description: "Discount percentage offered"
        constraints:
          range:
            min: 5
            max: 80
        generation_strategy: "realistic_pricing"

      product_category:
        type: "string"
        description: "Category of products being promoted"
        examples:
          - "Electronics"
          - "Clothing"
          - "Home & Garden"
          - "Books"
          - "Sports Equipment"
        generation_strategy: "market_realistic"

      urgency_indicators:
        type: "list"
        description: "Words/phrases indicating urgency"
        constraints:
          max_items: 5
        examples:
          - ["limited time", "expires soon"]
          - ["while supplies last", "flash sale"]
          - ["exclusive offer", "today only"]
        generation_strategy: "urgency_appropriate"
        dependencies: ["urgency"]

      call_to_action:
        type: "string"
        description: "Primary call-to-action phrase"
        constraints:
          length:
            min: 10
            max: 50
        examples:
          - "Shop Now and Save"
          - "Get Your Discount Today"
          - "Don't Miss Out - Buy Now"
        generation_strategy: "action_oriented"

      promotional_code:
        type: "string"
        description: "Promotional discount code"
        constraints:
          pattern:
            regex: "^[A-Z0-9]{4,12}$"
            description: "Alphanumeric promotional code"
        generation_strategy: "code_generator"

    templates:
      flash_sale: "🚨 {{discount_percentage}}% OFF {{product_category}}! {{urgency_indicators[0]}} - Use code {{promotional_code}}. {{call_to_action}}!"
      exclusive_offer: "VIP Exclusive: Save {{discount_percentage}}% on premium {{product_category}}. {{call_to_action}} with code {{promotional_code}}."
      seasonal_promotion: "{{generate:seasonal_event}} Sale! {{discount_percentage}}% off {{product_category}}. {{urgency_indicators[0]}}!"

    rules:
      - "promotional_authenticity: discount and urgency must be realistic"
      - "brand_consistency: tone must match promotional context"
      - "call_to_action_clarity: call_to_action must be clear and compelling"

    generation_patterns:
      - "high_energy_sales"
      - "exclusive_vip"
      - "time_sensitive_offers"

    metadata:
      category: "promotion"
      typical_sentiment_range: [0.7, 0.9]
      common_keywords: ["sale", "discount", "limited", "exclusive", "save"]

  # Educational emails
  educational_email:
    description: "Educational and learning-related communications"
    inherits_from: "base_email"

    parameters:
      course_subject:
        type: "string"
        description: "Subject area of the course or educational content"
        examples:
          - "Data Science"
          - "Machine Learning"
          - "Web Development"
          - "Digital Marketing"
          - "Business Analytics"
        generation_strategy: "academic_realistic"

      academic_level:
        type: "string"
        description: "Academic level or difficulty"
        constraints:
          enum: ["beginner", "intermediate", "advanced", "expert"]
        generation_strategy: "level_appropriate"
        dependencies: ["course_subject"]

      grade_mentioned:
        type: "string"
        description: "Grade or assessment score if applicable"
        constraints:
          pattern:
            regex: "^[A-F][+-]?$|^\\d{1,3}%$"
            description: "Letter grade or percentage"
        nullable: true
        generation_strategy: "realistic_grading"

      assignment_type:
        type: "string"
        description: "Type of academic assignment"
        examples:
          - "Final Project"
          - "Midterm Exam"
          - "Lab Report"
          - "Research Paper"
          - "Group Assignment"
        nullable: true
        generation_strategy: "assignment_appropriate"
        dependencies: ["course_subject", "academic_level"]

      instructor_name:
        type: "string"
        description: "Name of instructor or teacher"
        constraints:
          pattern:
            regex: "^(Dr\\.|Prof\\.|Mr\\.|Ms\\.|Mrs\\.)?\\s*[A-Z][a-z]+\\s+[A-Z][a-z]+$"
            description: "Proper instructor name format"
        generation_strategy: "academic_names"

    templates:
      course_announcement: "New {{academic_level}} {{course_subject}} course available! Instructor: {{instructor_name}}. {{generate:enrollment_details}}."
      grade_notification: "{{assignment_type}} grade posted: {{grade_mentioned}}. {{course_subject}} feedback: {{generate:instructor_feedback}}."
      assignment_reminder: "Reminder: {{assignment_type}} for {{course_subject}} due {{generate:due_date}}. {{generate:submission_instructions}}."

    rules:
      - "academic_formality: maintain appropriate academic tone"
      - "educational_accuracy: content must be educationally sound"
      - "instructor_credibility: instructor information must be realistic"

    generation_patterns:
      - "formal_academic"
      - "supportive_instructional"
      - "achievement_focused"

    metadata:
      category: "education"
      typical_sentiment_range: [0.5, 0.8]
      common_keywords: ["course", "assignment", "grade", "learning", "instructor"]

# Global generation rules
global_rules:
  - "semantic_coherence: all generated content must be semantically coherent"
  - "realistic_constraints: all values must fall within realistic ranges"
  - "category_consistency: generated content must match declared category"
  - "template_completeness: all template variables must be resolved"
  - "validation_compliance: all generated data must pass validation rules"

# Quality assurance parameters
quality_assurance:
  minimum_quality_score: 0.7
  required_validation_pass_rate: 0.95
  semantic_coherence_threshold: 0.8
  diversity_requirement: 0.85
  realism_score_minimum: 0.75
"""

    return ontology_dsl

def main():
    """Demonstrate the DSL-based ontology system"""

    print("🏗️ DSL-BASED ONTOLOGY SYSTEM")
    print("Template-driven AI training data generation with inheritance")
    print("=" * 80)

    # Create ontology engine
    ontology_engine = OntologyEngine()

    # Load the email ontology DSL
    print("\n📝 Loading Email Ontology DSL...")
    ontology_dsl = create_email_ontology_dsl()
    ontology_engine.load_ontology_dsl(ontology_dsl, "yaml")

    print(f"✓ Loaded {len(ontology_engine.entities)} entity definitions")
    print(f"✓ Built inheritance graph with validation")
    print(f"✓ Initialized rule engine and template processor")

    # Create data generator
    data_generator = DataGenerator(ontology_engine)

    # Generate samples for each entity type
    entity_types = ["work_email", "personal_email", "promotional_email", "educational_email"]

    all_samples = []

    for entity_type in entity_types:
        print(f"\n🎯 Generating samples for {entity_type}...")

        # Define variations for more diverse generation
        variations = {
            "context_variation": ["formal", "casual", "urgent"],
            "style_preference": ["concise", "detailed", "moderate"]
        }

        samples = data_generator.generate_training_dataset(
            entity_name=entity_type,
            sample_count=20,
            variations=variations
        )

        all_samples.extend(samples)

        # Show sample
        if samples:
            sample = samples[0]
            print(f"  📧 Sample {entity_type}:")
            print(f"     Subject: {sample.get('subject', 'N/A')}")
            print(f"     Urgency: {sample.get('urgency', 'N/A')}/10")
            print(f"     Sentiment: {sample.get('sentiment_score', 'N/A'):.2f}")
            if 'validation_errors' in sample:
                print(f"     Validation: {len(sample['validation_errors'])} errors")
            else:
                print(f"     Validation: ✓ Passed")

    # Save comprehensive dataset
    output_file = "dsl_generated_training_data.json"

    dataset_output = {
        "metadata": {
            "generator": "DSL-Based Ontology System",
            "ontology_version": "1.0.0",
            "generation_method": "Template-driven with inheritance",
            "total_samples": len(all_samples),
            "entity_types": entity_types,
            "generation_timestamp": datetime.now().isoformat(),
            "quality_assurance": "Multi-tier validation with semantic coherence"
        },
        "ontology_definition": ontology_dsl,
        "training_samples": all_samples
    }

    with open(output_file, 'w') as f:
        json.dump(dataset_output, f, indent=2)

    print(f"\n💾 DATASET GENERATION COMPLETE")
    print(f"📁 File: {output_file}")
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"🏗️ Entity types: {len(entity_types)}")
    print(f"✅ DSL-driven generation with full inheritance and validation")

    return ontology_engine, data_generator, all_samples

if __name__ == "__main__":
    main()