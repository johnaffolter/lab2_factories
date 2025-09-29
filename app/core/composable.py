"""
Composable Component System
Enables building ML pipelines from reusable, discoverable components
"""

from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import json


class ComponentType(Enum):
    """Types of composable components"""
    FEATURE_GENERATOR = "feature_generator"
    MODEL = "model"
    VALIDATOR = "validator"
    STORAGE = "storage"
    TRANSFORMER = "transformer"


@dataclass
class ComponentMetadata:
    """Metadata for component discovery and UI display"""
    name: str
    version: str
    type: ComponentType
    description: str
    author: str
    tags: List[str]
    icon: str  # Emoji for UI
    color: str  # Hex color

    # I/O specification
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    config_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result["type"] = self.type.value
        return result


class ComposableComponent(ABC):
    """Base class for all composable components"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metadata = self.get_metadata()

    @abstractmethod
    def get_metadata(self) -> ComponentMetadata:
        """Return component metadata for registration"""
        pass

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute component logic"""
        pass

    def __or__(self, other: 'ComposableComponent') -> 'ComposablePipeline':
        """Enable pipeline composition with | operator"""
        if isinstance(other, ComposablePipeline):
            return ComposablePipeline([self] + other.components)
        return ComposablePipeline([self, other])


class ComponentRegistry:
    """Registry for discovering and managing components"""

    def __init__(self):
        self.components: Dict[str, Type[ComposableComponent]] = {}
        self.metadata_cache: Dict[str, ComponentMetadata] = {}

    def register(self, component_class: Type[ComposableComponent]) -> Type[ComposableComponent]:
        """Register a component"""
        # Instantiate to get metadata
        temp = component_class()
        metadata = temp.get_metadata()

        self.components[metadata.name] = component_class
        self.metadata_cache[metadata.name] = metadata

        return component_class

    def get(self, name: str) -> Optional[Type[ComposableComponent]]:
        """Get component class by name"""
        return self.components.get(name)

    def list_all(self, type_filter: Optional[ComponentType] = None) -> List[ComponentMetadata]:
        """List all registered components"""
        if type_filter:
            return [m for m in self.metadata_cache.values() if m.type == type_filter]
        return list(self.metadata_cache.values())

    def search(self, query: str) -> List[ComponentMetadata]:
        """Search components by name, description, or tags"""
        query = query.lower()
        results = []

        for metadata in self.metadata_cache.values():
            if (query in metadata.name.lower() or
                query in metadata.description.lower() or
                any(query in tag for tag in metadata.tags)):
                results.append(metadata)

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary"""
        return {
            "components": [m.to_dict() for m in self.metadata_cache.values()],
            "total": len(self.components)
        }


class ComposablePipeline:
    """Chain multiple components into an executable pipeline"""

    def __init__(self, components: List[ComposableComponent]):
        self.components = components
        self.name = "Custom Pipeline"

    def execute(self, input_data: Any) -> Any:
        """Execute entire pipeline"""
        result = input_data

        for component in self.components:
            result = component.execute(result)

        return result

    def __or__(self, other: ComposableComponent) -> 'ComposablePipeline':
        """Add another component to pipeline"""
        return ComposablePipeline(self.components + [other])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline"""
        return {
            "name": self.name,
            "components": [
                {
                    "name": c.metadata.name,
                    "version": c.metadata.version,
                    "config": c.config
                }
                for c in self.components
            ]
        }

    def to_json(self) -> str:
        """Export as JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry: ComponentRegistry) -> 'ComposablePipeline':
        """Deserialize pipeline"""
        components = []

        for comp_data in data["components"]:
            ComponentClass = registry.get(comp_data["name"])
            if ComponentClass:
                component = ComponentClass(config=comp_data.get("config", {}))
                components.append(component)

        pipeline = cls(components)
        pipeline.name = data.get("name", "Custom Pipeline")
        return pipeline


# Global registry instance
global_registry = ComponentRegistry()