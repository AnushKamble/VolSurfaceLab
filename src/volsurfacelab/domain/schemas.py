"""Serialization, schema transformation, and export utilities for domain objects."""

from typing import Any, Dict
import json
from pydantic import BaseModel


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic domain model to a Python dict."""
    return model.model_dump(mode="json")


def model_to_json(model: BaseModel, indent: int = 2) -> str:
    """Serialize a Pydantic domain model to a formatted JSON string."""
    return json.dumps(model_to_dict(model), indent=indent, default=str)


def model_from_dict(cls: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """Instantiate a Pydantic domain model from a dictionary."""
    return cls.model_validate(data)


def model_from_json(cls: type[BaseModel], json_str: str) -> BaseModel:
    """Instantiate a Pydantic domain model from a JSON string."""
    return cls.model_validate_json(json_str)
