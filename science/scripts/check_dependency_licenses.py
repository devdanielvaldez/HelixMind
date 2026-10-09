#!/usr/bin/env python3
"""Check pip-licenses JSON against MedAI's fail-closed license label policy."""
from __future__ import annotations

import json
import sys

from medai_science.dependency_license_policy import validate_license_inventory


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def main() -> int:
    try:
        inventory = json.load(
            sys.stdin,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {value}")),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        print(f"license inventory is invalid JSON: {error}", file=sys.stderr)
        return 2

    violations = validate_license_inventory(inventory)
    if violations:
        for violation in violations:
            print(f"license policy violation: {violation}", file=sys.stderr)
        return 1
    print(f"license policy passed for {len(inventory)} installed packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
