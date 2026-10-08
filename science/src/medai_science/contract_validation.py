"""Small fail-closed JSON Schema subset for generated capability contracts."""
from __future__ import annotations

import math
from typing import Any


_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def validate_schema(schema: Any, path: str = "$", depth: int = 0) -> None:
    if depth > 32 or not isinstance(schema, dict):
        raise ValueError(f"capability schema must be a bounded object at {path}")
    allowed = {"type", "title", "description", "properties", "required", "additionalProperties", "items", "enum", "const", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"}
    unknown = set(schema) - allowed
    if unknown:
        raise ValueError(f"unsupported JSON Schema keywords at {path}: {', '.join(sorted(unknown))}")
    kind = schema.get("type")
    if kind not in _SCHEMA_TYPES:
        raise ValueError(f"capability schema requires a supported type at {path}")
    properties = schema.get("properties", {})
    if kind != "object" and properties:
        raise ValueError(f"properties require object type at {path}")
    if not isinstance(properties, dict) or len(properties) > 256 or any(not isinstance(key, str) or not key for key in properties):
        raise ValueError(f"invalid object properties at {path}")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) or key not in properties for key in required):
        raise ValueError(f"required must name declared properties at {path}")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError(f"additionalProperties must be boolean at {path}")
    if "items" in schema:
        if kind != "array":
            raise ValueError(f"items require array type at {path}")
        validate_schema(schema["items"], f"{path}[]", depth + 1)
    for key, child in properties.items():
        validate_schema(child, f"{path}.{key}", depth + 1)
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not 1 <= len(enum) <= 256):
        raise ValueError(f"enum must contain 1..256 values at {path}")
    for key in ("minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"):
        value = schema.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid {key} bound at {path}")
    if kind != "string" and any(key in schema for key in ("minLength", "maxLength")):
        raise ValueError(f"string bounds require string type at {path}")
    if kind != "array" and any(key in schema for key in ("minItems", "maxItems")):
        raise ValueError(f"array bounds require array type at {path}")
    if kind not in {"integer", "number"} and any(key in schema for key in ("minimum", "maximum")):
        raise ValueError(f"numeric bounds require number or integer type at {path}")


def validate_value(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    kind = schema["type"]
    matches = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[kind]
    if not matches:
        raise ValueError(f"value at {path} does not match contract type {kind}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"value at {path} does not match contract const")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"value at {path} is outside contract enum")
    if kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"object at {path} omits required contract fields: {', '.join(sorted(missing))}")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError(f"object at {path} contains undeclared contract fields")
        for key, item in value.items():
            if key in properties:
                validate_value(item, properties[key], f"{path}.{key}")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ValueError(f"array at {path} violates contract item bounds")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_value(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ValueError(f"string at {path} violates contract length bounds")
    elif kind in {"integer", "number"}:
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError(f"number at {path} violates contract bounds")


def capability_io_schemas(contract: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read either MedAI's input/output or AMCP's inputs/outputs schema names."""
    if not isinstance(contract, dict):
        raise ValueError("capability contract must be an object")
    input_schema = contract.get("inputs", contract.get("input"))
    output_schema = contract.get("outputs", contract.get("output"))
    validate_schema(input_schema, "$.inputs")
    validate_schema(output_schema, "$.outputs")
    if input_schema["type"] != "object" or output_schema["type"] != "object":
        raise ValueError("capability input and output schemas must describe JSON objects")
    return input_schema, output_schema
