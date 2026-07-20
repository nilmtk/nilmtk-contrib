#!/usr/bin/env python3
"""Check nilmtk-contrib's public onboarding and model index."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

REQUIRED_TEXT = (
    "Dataset conversion, meter access, preprocessing, and metrics",
    "Appliance taxonomy, synonyms, meter relationships, and dataset schema",
    "Disaggregation model implementation and testing",
    "Fixed T1/T2/T3 evaluation and published result bundles",
    "https://nilmtk.github.io/",
    "https://github.com/nilmtk/nilmtk",
    "https://github.com/nilmtk/nilm_metadata",
    "https://github.com/nilmtk/nilmbench",
    "https://doi.org/10.1145/3360322.3360844",
    "https://doi.org/10.1145/3744256.3812587",
    "UV_TORCH_BACKEND=cpu",
    '"nilmtk-contrib[torch] @ git+https://github.com/nilmtk/nilmtk-contrib.git"',
    "This repository owns the one general NILMTK development Dockerfile.",
    "Anonymous pulls from the GHCR package are not yet part of the supported path",
)

FORBIDDEN_TEXT = (
    "# NILMTK-Contrib with NILMBench2026",
    "NILMBench2026 benchmark contribution",
    "docker pull ghcr.io/nilmtk/nilmtk-contrib",
    "https://github.com/sustainability-lab/nilmbench",
)

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def exported_models(package: str) -> set[str]:
    path = ROOT / "nilmtk_contrib" / package / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_EXPORTS" for target in node.targets):
            exports = ast.literal_eval(node.value)
            return set(exports)
    raise RuntimeError(f"{path.relative_to(ROOT)} does not define _EXPORTS")


def check_local_links(source: str, errors: list[str]) -> int:
    count = 0
    for raw_target in MARKDOWN_LINK.findall(source):
        target = raw_target.split(maxsplit=1)[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:", "tel:")):
            continue
        path = (ROOT / unquote(parsed.path)).resolve()
        count += 1
        if not path.exists():
            errors.append(f"README.md has a missing local link: {target}")
    return count


def main() -> int:
    source = README.read_text(encoding="utf-8")
    normalized = " ".join(source.split())
    errors: list[str] = []

    if source.count("```") % 2:
        errors.append("README.md has an unclosed code fence")
    if "http://" in source:
        errors.append("README.md contains an insecure URL")

    for required in REQUIRED_TEXT:
        if required not in normalized:
            errors.append(f"README.md is missing required text: {required}")
    for forbidden in FORBIDDEN_TEXT:
        if forbidden in source:
            errors.append(f"README.md contains retired text: {forbidden}")

    for package in ("disaggregate", "torch"):
        for model in exported_models(package):
            public_path = f"`nilmtk_contrib.{package}.{model}`"
            if public_path not in source:
                errors.append(f"model table is missing public import: {public_path}")
    if "`nilmtk_contrib.torch.msdc_without_crf.MSDC`" not in source:
        errors.append("model table is missing the public MSDC no-CRF module")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for extra in ("torch", "tensorflow", "classical", "all", "nilm"):
        if f"{extra} = [" not in pyproject:
            errors.append(f"documented dependency extra is not defined: {extra}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    if 'org.opencontainers.image.source="https://github.com/nilmtk/nilmtk-contrib"' not in dockerfile:
        errors.append("Dockerfile does not identify the canonical source repository")
    if "sustainability-lab/nilmbench" in dockerfile:
        errors.append("Dockerfile still identifies the old benchmark repository")

    checked_links = check_local_links(source, errors)
    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Documentation checks passed: "
        f"{len(REQUIRED_TEXT)} contract clauses, {checked_links} local links, "
        f"{len(exported_models('disaggregate')) + len(exported_models('torch'))} "
        "exported model imports."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
