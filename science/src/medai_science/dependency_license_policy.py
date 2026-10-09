"""Fail-closed policy for license labels reported by pip-licenses."""
from __future__ import annotations

from typing import Any

ALLOWED_LICENSE_LABELS = frozenset(
    label.casefold()
    for label in (
        "Apache Software License",
        "Apache License 2.0",
        "Apache-2.0",
        "MIT License",
        "MIT",
        "BSD License",
        "BSD",
        "BSD 2-Clause License",
        "BSD 3-Clause License",
        "BSD 3-Clause 'New' or 'Revised' License",
        "BSD 2-Clause 'Simplified' License",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC License",
        "ISC",
        "Python Software Foundation License",
        "PSF License",
        "Mozilla Public License 2.0 (MPL 2.0)",
        "MPL-2.0",
        "Unlicense",
        "CC0 1.0 Universal",
        "CC0-1.0",
    )
)


def validate_license_inventory(packages: Any) -> list[str]:
    """Return policy violations; every declared label must be explicitly allowed."""
    if not isinstance(packages, list) or not packages:
        return ["license inventory must be a non-empty JSON array"]

    violations: list[str] = []
    seen: set[str] = set()
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            violations.append(f"package row {index} must be a JSON object")
            continue
        name = package.get("Name")
        license_value = package.get("License")
        if not isinstance(name, str) or not name.strip():
            violations.append(f"package row {index} has no valid Name")
            name = f"<row {index}>"
        if name in seen:
            violations.append(f"duplicate package entry: {name}")
        seen.add(name)
        if not isinstance(license_value, str) or not license_value.strip():
            violations.append(f"{name}: license metadata is missing")
            continue

        labels = [label.strip() for label in license_value.split(";")]
        if any(not label for label in labels):
            violations.append(f"{name}: license metadata contains an empty label")
            continue
        for label in labels:
            if label.casefold() not in ALLOWED_LICENSE_LABELS:
                violations.append(f"{name}: unapproved or unrecognized license label: {label}")
    return violations
