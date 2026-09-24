"""Minimal JSON-Schema validator + coercer (stdlib only).

Why NEXUS ships its own instead of using ``jsonschema``:

* Zero dependencies is a hard project requirement.
* We need **coercion**, not just validation. Weaker models routinely send
  ``{"limit": "20"}`` or ``{"replace_all": "true"}``; rejecting those costs a
  full retry round-trip, while coercing them is safe and obvious.
* Errors must be actionable for the model: each failure is reported as a
  JSON-pointer style path plus the expected type, which the agent loop feeds
  straight back into the next turn.

Supported keywords: type, properties, required, items, enum, const, minimum,
maximum, exclusiveMinimum/Maximum, minLength, maxLength, pattern, minItems,
maxItems, additionalProperties (bool), anyOf/oneOf/allOf, nullable, default.
Unknown keywords are ignored (forward compatible).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


class SchemaError(ValueError):
    """Raised when the *schema itself* is malformed (a bug in a tool definition)."""


def validate(instance: Any, schema: Optional[Dict[str, Any]], path: str = "") -> List[str]:
    """Return a list of human/model readable validation errors (empty == valid)."""
    if not isinstance(schema, dict):
        return []
    errors: List[str] = []

    types = schema.get("type")
    nullable = bool(schema.get("nullable"))
    if types:
        allowed = types if isinstance(types, list) else [types]
        if nullable and "null" not in allowed:
            allowed = list(allowed) + ["null"]
        if not _type_ok(instance, allowed):
            errors.append(f"{path or '(root)'}: expected {('/'.join(allowed))}, got {_type_name(instance)}")
            return errors  # further checks are meaningless
    if instance is None:
        return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path or '(root)'}: must equal {schema['const']!r}")
    if "enum" in schema and isinstance(schema["enum"], list) and instance not in schema["enum"]:
        errors.append(f"{path or '(root)'}: must be one of {schema['enum']}")

    if isinstance(instance, dict):
        for req in schema.get("required") or []:
            if req not in instance or instance[req] is None:
                errors.append(f"{path or '(root)'}: missing required property '{req}'")
        props = schema.get("properties") or {}
        addl = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], f"{path}.{key}" if path else key))
            elif addl is False:
                errors.append(f"{path or '(root)'}: unexpected property '{key}'")
        for key, sub in props.items():
            if isinstance(sub, dict) and "default" in sub and key not in instance:
                pass  # defaults are applied by apply_defaults()
    elif isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, value in enumerate(instance):
                errors.extend(validate(value, items, f"{path}[{i}]"))
        elif isinstance(items, list):
            for i, (value, sub) in enumerate(zip(instance, items)):
                errors.extend(validate(value, sub, f"{path}[{i}]"))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path or '(root)'}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path or '(root)'}: allows at most {schema['maxItems']} items")
    elif isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path or '(root)'}: must be at least {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path or '(root)'}: must be at most {schema['maxLength']} characters")
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], instance) is None:
                    errors.append(f"{path or '(root)'}: must match /{schema['pattern']}/")
            except re.error as exc:
                raise SchemaError(f"invalid pattern in schema: {exc}") from exc
    elif isinstance(instance, bool):
        pass
    elif isinstance(instance, (int, float)):
        for key, op in (("minimum", lambda a, b: a >= b), ("exclusiveMinimum", lambda a, b: a > b),
                        ("maximum", lambda a, b: a <= b), ("exclusiveMaximum", lambda a, b: a < b)):
            if key in schema and isinstance(schema[key], (int, float)) and not op(instance, schema[key]):
                errors.append(f"{path or '(root)'}: {key} is {schema[key]}, got {instance}")

    for combiner in ("allOf", "anyOf", "oneOf"):
        subs = schema.get(combiner)
        if not isinstance(subs, list) or not subs:
            continue
        results = [validate(instance, s, path) for s in subs if isinstance(s, dict)]
        if combiner == "allOf":
            for r in results:
                errors.extend(r)
        elif combiner == "anyOf":
            if all(r for r in results):
                errors.append(f"{path or '(root)'}: does not match any allowed variant")
        else:  # oneOf
            matches = sum(1 for r in results if not r)
            if matches != 1:
                errors.append(f"{path or '(root)'}: must match exactly one variant (matched {matches})")
    return errors


def coerce(instance: Any, schema: Optional[Dict[str, Any]]) -> Any:
    """Best-effort type coercion driven by the schema. Never raises."""
    if not isinstance(schema, dict):
        return instance
    types = schema.get("type")
    allowed = (types if isinstance(types, list) else [types]) if types else []
    if instance is None:
        return None
    if "object" in allowed and isinstance(instance, str):
        parsed = _try_json(instance)
        if isinstance(parsed, dict):
            instance = parsed
    if "array" in allowed and isinstance(instance, str):
        parsed = _try_json(instance)
        if isinstance(parsed, list):
            instance = parsed
        elif parsed is not None and not isinstance(parsed, (dict,)):
            instance = [parsed]
    if "integer" in allowed and not isinstance(instance, bool):
        if isinstance(instance, str):
            v = _try_number(instance)
            if v is not None:
                instance = int(v) if float(v).is_integer() else v
        elif isinstance(instance, float) and instance.is_integer():
            instance = int(instance)
    if "number" in allowed and isinstance(instance, str):
        v = _try_number(instance)
        if v is not None:
            instance = v
    if "boolean" in allowed:
        if isinstance(instance, str):
            low = instance.strip().lower()
            if low in ("true", "yes", "y", "1", "on"):
                instance = True
            elif low in ("false", "no", "n", "0", "off"):
                instance = False
        elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
            instance = bool(instance)
    if "string" in allowed and not isinstance(instance, str) and isinstance(instance, (int, float, bool)):
        instance = json.dumps(instance) if isinstance(instance, bool) else str(instance)

    if isinstance(instance, dict):
        props = schema.get("properties") or {}
        out = {}
        for k, v in instance.items():
            out[k] = coerce(v, props.get(k)) if isinstance(props.get(k), dict) else v
        instance = out
    elif isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            instance = [coerce(v, items) for v in instance]
    return instance


def apply_defaults(instance: Any, schema: Optional[Dict[str, Any]]) -> Any:
    """Fill in schema ``default`` values for missing object properties."""
    if not isinstance(schema, dict) or not isinstance(instance, dict):
        return instance
    props = schema.get("properties") or {}
    out = dict(instance)
    for key, sub in props.items():
        if isinstance(sub, dict) and "default" in sub and key not in out:
            out[key] = sub["default"]
        elif isinstance(sub, dict) and key in out:
            out[key] = apply_defaults(out[key], sub)
    return out


def normalize_args(args: Any, schema: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce -> defaults -> validate. The single entry point used by the agent loop."""
    if args is None:
        args = {}
    if isinstance(args, str):
        parsed = _try_json(args)
        args = parsed if isinstance(parsed, dict) else {"_raw": args}
    if not isinstance(args, dict):
        return {}, [f"arguments must be a JSON object, got {_type_name(args)}"]
    coerced = coerce(args, schema)
    if not isinstance(coerced, dict):
        coerced = args
    with_defaults = apply_defaults(coerced, schema)
    errors = validate(with_defaults, schema)
    return with_defaults, errors


def _type_ok(instance: Any, allowed: List[str]) -> bool:
    for t in allowed:
        py = _TYPES.get(t)
        if py is None:
            continue
        if t == "integer":
            if isinstance(instance, bool):
                continue
            if isinstance(instance, int):
                return True
            if isinstance(instance, float) and instance.is_integer():
                return True
            continue
        if t == "number":
            if isinstance(instance, bool):
                continue
            if isinstance(instance, (int, float)):
                return True
            continue
        if t == "string" and isinstance(instance, str):
            return True
        if t == "boolean" and isinstance(instance, bool):
            return True
        if t == "array" and isinstance(instance, list):
            return True
        if t == "object" and isinstance(instance, dict):
            return True
        if t == "null" and instance is None:
            return True
    return False


def _type_name(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, dict):
        return "object"
    return type(instance).__name__


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _try_number(text: str) -> Optional[float]:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


__all__ = ["validate", "coerce", "apply_defaults", "normalize_args", "SchemaError"]
