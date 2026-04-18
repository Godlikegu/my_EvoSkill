"""Auxiliary compile-audit adapters for README and public-contract extraction."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


class CompileAuditAdapter(ABC):
    """Optional audit layer that suggests public-facing compile metadata."""

    @abstractmethod
    def audit(
        self,
        readme_text: str,
        manifest: Mapping[str, Any],
        compile_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return structured audit suggestions without mutating compiler output."""


class NullCompileAuditAdapter(CompileAuditAdapter):
    """No-op audit adapter."""

    def audit(
        self,
        readme_text: str,
        manifest: Mapping[str, Any],
        compile_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "suspicious_sections": [],
            "suggested_public_contract": {},
            "suggested_method_hints": [],
            "sanitization_warnings": [],
        }


class HeuristicCompileAuditAdapter(CompileAuditAdapter):
    """Deterministic placeholder for future LLM-backed compile auditing."""

    SUSPICIOUS_PATTERNS = (
        r"(?i)ground_truth",
        r"(?i)evaluation/reference_outputs",
        r"(?i)evaluation/tests",
        r"(?i)\bsrc/",
        r"(?i)\bmain\.py\b",
        r"(?i)\bnotebooks/",
    )

    def audit(
        self,
        readme_text: str,
        manifest: Mapping[str, Any],
        compile_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        suspicious_sections = []
        sanitization_warnings = []
        for line in readme_text.splitlines():
            if any(re.search(pattern, line) for pattern in self.SUSPICIOUS_PATTERNS):
                stripped = line.strip()
                suspicious_sections.append(stripped)
                sanitization_warnings.append(f"suspicious_readme_line:{stripped}")

        method_hints = self._extract_method_hints(readme_text)
        suggested_contract: Dict[str, Any] = {}
        output_name = manifest.get("proxy_output_name")
        if output_name:
            suggested_contract["required_outputs"] = [
                {
                    "path": f"output/{output_name}",
                    "format": output_name.split(".")[-1],
                }
            ]
        judge_metrics = manifest.get("judge_metrics")
        if judge_metrics:
            suggested_contract["judge_metrics"] = list(judge_metrics)
        if method_hints:
            suggested_contract["method_hints"] = method_hints

        return {
            "suspicious_sections": suspicious_sections,
            "suggested_public_contract": suggested_contract,
            "suggested_method_hints": method_hints,
            "sanitization_warnings": sanitization_warnings,
        }

    def _extract_method_hints(self, readme_text: str) -> list[str]:
        lines = readme_text.splitlines()
        in_method_hints = False
        hints: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip().lower()
                in_method_hints = heading == "method hints"
                continue
            if not in_method_hints:
                continue
            if not stripped:
                continue
            if stripped.startswith("- "):
                hints.append(stripped[2:].strip())
            else:
                hints.append(stripped)
        return hints
