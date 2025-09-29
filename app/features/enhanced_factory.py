"""
Enhanced Feature Generator Factory with Advanced Design Patterns

This module implements multiple design patterns with comprehensive documentation:
- Factory Method Pattern: Dynamic object creation based on type
- Strategy Pattern: Interchangeable feature extraction algorithms
- Singleton Pattern: Single factory instance across application
- Builder Pattern: Complex feature pipeline construction
- Observer Pattern: Feature extraction event notifications
- Chain of Responsibility: Sequential feature processing

Author: MLOps Team
Version: 2.0.0
License: MIT
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from functools import lru_cache, singledispatch
from enum import Enum, auto
import threading
import inspect
import json
from datetime import datetime

from app.dataclasses import Email


class FeatureType(Enum):
    """
    Enumeration of feature types for type-safe feature categorization.

    Attributes:
        SPAM: Spam detection features
        LINGUISTIC: Language and text analysis features
        STATISTICAL: Statistical text metrics
        SEMANTIC: Meaning and context features
        STRUCTURAL: Document structure features
    """
    SPAM = auto()
    LINGUISTIC = auto()
    STATISTICAL = auto()
    SEMANTIC = auto()
    STRUCTURAL = auto()


@dataclass
class FeatureMetadata:
    """
    Metadata container for feature generators with rich documentation.

    Attributes:
        name (str): Unique identifier for the feature generator
        description (str): Human-readable description
        version (str): Semantic version number
        author (str): Creator of the feature generator
        feature_type (FeatureType): Category of features produced
        parameters (Dict[str, Any]): Configuration parameters
        dependencies (List[str]): Required dependencies
        performance_metrics (Dict[str, float]): Performance benchmarks

    Examples:
        >>> metadata = FeatureMetadata(
        ...     name="spam_detector",
        ...     description="Advanced spam detection using ML",
        ...     version="1.0.0",
        ...     feature_type=FeatureType.SPAM
        ... )
    """
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "System"
    feature_type: FeatureType = FeatureType.STATISTICAL
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of metadata."""
        return f"{self.name} v{self.version} - {self.description}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"FeatureMetadata(name='{self.name}', "
                f"type={self.feature_type.name}, "
                f"version='{self.version}')")

    def to_json(self) -> str:
        """Serialize metadata to JSON."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "feature_type": self.feature_type.name,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "performance_metrics": self.performance_metrics
        }
        return json.dumps(data, indent=2)


class FeatureGeneratorInterface(ABC):
    """
    Abstract interface for all feature generators implementing Strategy Pattern.

    This interface ensures consistent behavior across all feature generators
    while allowing flexible implementation of extraction algorithms.

    Design Pattern: Strategy Pattern
    Purpose: Define a family of algorithms, encapsulate each one, and make them interchangeable

    Methods:
        generate_features: Extract features from email
        validate_input: Ensure input meets requirements
        get_metadata: Return generator metadata
        get_feature_names: List features this generator produces
    """

    def __init__(self):
        """Initialize generator with metadata."""
        self._metadata = self._create_metadata()
        self._performance_tracker = {}

    @abstractmethod
    def _create_metadata(self) -> FeatureMetadata:
        """
        Create metadata for this generator.

        Returns:
            FeatureMetadata: Generator metadata
        """
        pass

    @abstractmethod
    def generate_features(self, email: Email) -> Dict[str, Any]:
        """
        Extract features from email content.

        Args:
            email (Email): Email object to process

        Returns:
            Dict[str, Any]: Dictionary of feature names to values

        Raises:
            ValueError: If email is invalid
            RuntimeError: If feature extraction fails
        """
        pass

    @abstractmethod
    def validate_input(self, email: Email) -> bool:
        """
        Validate email meets generator requirements.

        Args:
            email (Email): Email to validate

        Returns:
            bool: True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this generator produces.

        Returns:
            List[str]: Feature names
        """
        pass

    def get_metadata(self) -> FeatureMetadata:
        """Return generator metadata."""
        return self._metadata

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__}({self._metadata.name})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"<{self.__class__.__name__} "
                f"name='{self._metadata.name}' "
                f"version='{self._metadata.version}'>")