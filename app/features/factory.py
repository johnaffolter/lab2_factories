"""
Factory Pattern Implementation for Feature Generation

This module implements the Factory Method design pattern to create feature generators
dynamically. It demonstrates several design patterns and best practices:

Design Patterns Implemented:
    - Factory Method: Dynamic creation of feature generators
    - Registry Pattern: Centralized registration of generator types
    - Strategy Pattern: Interchangeable feature extraction algorithms
    - Singleton (optional): Single factory instance if needed

Author: MLOps Team
Version: 2.0.0
"""

from typing import Dict, Any, List, Optional, Type, Union
from functools import lru_cache
import json
from datetime import datetime
from .base import BaseFeatureGenerator
from .generators import (
    SpamFeatureGenerator,
    AverageWordLengthFeatureGenerator,
    EmailEmbeddingsFeatureGenerator,
    RawEmailFeatureGenerator,
    NonTextCharacterFeatureGenerator,
    CausalImpactFeatureGenerator
)
from app.dataclasses import Email


# Constant list of available generators with metadata
GENERATORS: Dict[str, Type[BaseFeatureGenerator]] = {
    "spam": SpamFeatureGenerator,
    "word_length": AverageWordLengthFeatureGenerator,
    "email_embeddings": EmailEmbeddingsFeatureGenerator,
    "raw_email": RawEmailFeatureGenerator,
    "non_text": NonTextCharacterFeatureGenerator,
    "causal_impact": CausalImpactFeatureGenerator
}

# Generator metadata for documentation
GENERATOR_METADATA = {
    "spam": {
        "description": "Detects spam-related keywords and patterns",
        "category": "content_analysis",
        "version": "1.0.0",
        "performance": "O(n) where n is text length"
    },
    "word_length": {
        "description": "Analyzes word length statistics",
        "category": "linguistic_analysis",
        "version": "1.0.0",
        "performance": "O(n) where n is word count"
    },
    "email_embeddings": {
        "description": "Generates semantic embeddings for emails",
        "category": "ml_features",
        "version": "1.0.0",
        "performance": "O(1) - currently using simple length calculation"
    },
    "raw_email": {
        "description": "Extracts raw email content as features",
        "category": "raw_features",
        "version": "1.0.0",
        "performance": "O(1) - direct extraction"
    },
    "non_text": {
        "description": "Counts non-alphanumeric characters",
        "category": "content_analysis",
        "version": "1.1.0",
        "performance": "O(n) where n is text length"
    },
    "causal_impact": {
        "description": "Generates causal impact and intervention features",
        "category": "causal_analysis",
        "version": "1.0.0",
        "performance": "O(1) - direct computation"
    }
}


class FeatureGeneratorFactory:
    """
    Factory for creating and managing feature generators using the Factory Method pattern.

    This class serves as a centralized factory for creating feature generator instances
    based on their type identifiers. It implements the Factory Method pattern to allow
    flexible creation of different feature extractors without coupling the client code
    to specific generator implementations.

    Attributes:
        _generators (Dict[str, Type[BaseFeatureGenerator]]): Registry of available generators
        _cache (Dict[str, BaseFeatureGenerator]): Cache of instantiated generators
        _statistics (Dict[str, Any]): Usage statistics for monitoring

    Design Patterns:
        - Factory Method: Creates objects without specifying exact classes
        - Registry: Maintains a registry of available generator types
        - Lazy Initialization: Generators created only when needed

    Examples:
        >>> factory = FeatureGeneratorFactory()
        >>> features = factory.generate_all_features(email)
        >>> spam_gen = factory.create_generator("spam")
    """

    def __init__(self):
        """
        Initialize the factory with generator registry and monitoring.

        Sets up the internal registry of generators and initializes
        tracking mechanisms for usage statistics and caching.
        """
        self._generators: Dict[str, Type[BaseFeatureGenerator]] = GENERATORS
        self._cache: Dict[str, BaseFeatureGenerator] = {}
        self._statistics = {
            "created_count": 0,
            "feature_generation_count": 0,
            "last_used": None,
            "generator_usage": {name: 0 for name in GENERATORS.keys()}
        }

    def create_generator(self, generator_type: str) -> BaseFeatureGenerator:
        """
        Create a feature generator instance using Factory Method pattern.

        Args:
            generator_type (str): Type identifier for the generator
                Must be one of: 'spam', 'word_length', 'email_embeddings',
                'raw_email', 'non_text'

        Returns:
            BaseFeatureGenerator: Instance of the requested generator

        Raises:
            ValueError: If generator_type is not recognized
            RuntimeError: If generator instantiation fails

        Examples:
            >>> factory = FeatureGeneratorFactory()
            >>> spam_gen = factory.create_generator("spam")
            >>> features = spam_gen.generate_features(email)
        """
        if generator_type not in self._generators:
            available = ", ".join(self._generators.keys())
            raise ValueError(
                f"Unknown generator type: '{generator_type}'. "
                f"Available types: {available}"
            )

        # Use cached instance if available (Flyweight pattern)
        if generator_type not in self._cache:
            try:
                generator_class = self._generators[generator_type]
                self._cache[generator_type] = generator_class()
                self._statistics["created_count"] += 1
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create generator '{generator_type}': {str(e)}"
                )

        self._statistics["generator_usage"][generator_type] += 1
        self._statistics["last_used"] = datetime.now().isoformat()

        return self._cache[generator_type]

    def generate_all_features(self,
                            email: Email,
                            generator_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate features using multiple generators (Composite pattern).

        This method orchestrates multiple feature generators to extract
        a comprehensive feature set from an email. It demonstrates the
        Composite pattern by treating individual generators uniformly.

        Args:
            email (Email): Email object to extract features from
            generator_names (Optional[List[str]]): Specific generators to use
                If None, all available generators are used

        Returns:
            Dict[str, Any]: Combined features from all generators
                Keys are prefixed with generator name to avoid conflicts
                Format: {generator_name}_{feature_name}

        Raises:
            ValueError: If email is None or invalid
            RuntimeError: If feature generation fails

        Examples:
            >>> factory = FeatureGeneratorFactory()
            >>> email = Email(subject="Test", body="Content")
            >>> features = factory.generate_all_features(email)
            >>> print(features["spam_has_spam_words"])
        """
        if email is None:
            raise ValueError("Email cannot be None")

        if generator_names is None:
            generator_names = list(self._generators.keys())

        all_features = {}
        errors = []

        for gen_name in generator_names:
            try:
                generator = self.create_generator(gen_name)
                features = generator.generate_features(email)

                # Prefix features with generator name (namespace pattern)
                for feature_name, value in features.items():
                    prefixed_name = f"{gen_name}_{feature_name}"
                    all_features[prefixed_name] = value

            except Exception as e:
                errors.append(f"{gen_name}: {str(e)}")

        if errors and not all_features:
            raise RuntimeError(f"Feature generation failed: {'; '.join(errors)}")

        self._statistics["feature_generation_count"] += 1
        return all_features

    @classmethod
    def get_available_generators(cls) -> List[Dict[str, Any]]:
        """
        Get detailed information about available generators.

        This class method provides metadata about all registered generators
        including their features, descriptions, and performance characteristics.

        Returns:
            List[Dict[str, Any]]: List of generator information dictionaries
                Each dict contains: name, features, description, category, version

        Examples:
            >>> generators = FeatureGeneratorFactory.get_available_generators()
            >>> for gen in generators:
            ...     print(f"{gen['name']}: {gen['description']}")
        """
        generators_info = []
        for name, generator_class in GENERATORS.items():
            generator = generator_class()
            metadata = GENERATOR_METADATA.get(name, {})

            generators_info.append({
                "name": name,
                "features": generator.feature_names,
                "description": metadata.get("description", "No description available"),
                "category": metadata.get("category", "uncategorized"),
                "version": metadata.get("version", "1.0.0"),
                "performance": metadata.get("performance", "O(n)")
            })
        return generators_info

    @staticmethod
    def register_generator(name: str,
                          generator_class: Type[BaseFeatureGenerator],
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new generator type (Open/Closed Principle).

        Allows extending the factory with new generator types without
        modifying the factory class itself, following the Open/Closed Principle.

        Args:
            name (str): Unique identifier for the generator
            generator_class (Type[BaseFeatureGenerator]): Generator class to register
            metadata (Optional[Dict[str, Any]]): Optional metadata for the generator

        Raises:
            ValueError: If name is already registered or invalid

        Examples:
            >>> class CustomGenerator(BaseFeatureGenerator):
            ...     pass
            >>> FeatureGeneratorFactory.register_generator(
            ...     "custom", CustomGenerator,
            ...     {"description": "Custom feature extraction"}
            ... )
        """
        if name in GENERATORS:
            raise ValueError(f"Generator '{name}' is already registered")

        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid generator name: '{name}'")

        GENERATORS[name] = generator_class
        if metadata:
            GENERATOR_METADATA[name] = metadata

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for monitoring and optimization.

        Returns:
            Dict[str, Any]: Statistics including usage counts and timing

        Examples:
            >>> factory = FeatureGeneratorFactory()
            >>> stats = factory.get_statistics()
            >>> print(f"Generators created: {stats['created_count']}")
        """
        return self._statistics.copy()

    def reset_cache(self) -> None:
        """
        Clear the generator cache to free memory.

        Useful for long-running applications to prevent memory leaks.
        """
        self._cache.clear()
        self._statistics["created_count"] = 0

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"FeatureGeneratorFactory with {len(self._generators)} "
                f"generators: {', '.join(self._generators.keys())}")

    def __repr__(self) -> str:
        """Developer-friendly representation for debugging."""
        return (f"FeatureGeneratorFactory("
                f"generators={list(self._generators.keys())}, "
                f"cached={list(self._cache.keys())})")

    def __len__(self) -> int:
        """Return number of available generator types."""
        return len(self._generators)

    def __contains__(self, generator_type: str) -> bool:
        """Check if a generator type is available."""
        return generator_type in self._generators

    def __getitem__(self, generator_type: str) -> BaseFeatureGenerator:
        """Dictionary-like access to generators."""
        return self.create_generator(generator_type)

    def __iter__(self):
        """Iterate over available generator names."""
        return iter(self._generators.keys())


# Singleton instance (optional usage)
_factory_instance: Optional[FeatureGeneratorFactory] = None


def get_factory_instance() -> FeatureGeneratorFactory:
    """
    Get singleton instance of the factory (Singleton pattern).

    This function ensures only one factory instance exists globally,
    useful for maintaining consistent state and caching.

    Returns:
        FeatureGeneratorFactory: Singleton factory instance

    Examples:
        >>> factory = get_factory_instance()
        >>> factory2 = get_factory_instance()
        >>> assert factory is factory2  # Same instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = FeatureGeneratorFactory()
    return _factory_instance