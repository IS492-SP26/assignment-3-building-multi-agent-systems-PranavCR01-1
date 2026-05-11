"""
Output Guardrail
Checks system outputs for PII, harmful content, and other safety issues.
"""

from typing import Dict, Any, List, Optional
import re


class OutputGuardrail:
    """Guardrail for checking output safety before returning to the user."""

    # Regex patterns for PII detection
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "phone_us": r"\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    # Phrases that indicate highly harmful output
    HARMFUL_PHRASES = [
        "step-by-step guide to",
        "instructions for making",
        "how to synthesize",
        "detailed instructions for",
        "recipe for",
        "exploit vulnerability",
        "crack the password",
        "bypass authentication",
    ]

    # Harmful topics that should trigger warnings in outputs
    HARMFUL_TOPICS = [
        "weapons of mass destruction",
        "biological weapon",
        "chemical weapon",
        "child pornography",
        "child sexual abuse",
        "self-harm instructions",
        "suicide instructions",
    ]

    def __init__(self, config: Dict[str, Any]):
        """Initialize output guardrail from config."""
        self.config = config
        safety_cfg = config.get("safety", {})
        self.on_violation_action = safety_cfg.get("on_violation", {}).get("action", "sanitize")

    def validate(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate an output response.

        Returns a dict with:
          - valid: bool
          - violations: list of violation dicts
          - sanitized_output: safe version to show the user
          - action_taken: "none" | "redacted" | "refused"
        """
        violations: List[Dict[str, Any]] = []

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))

        if sources:
            violations.extend(self._check_factual_consistency(response, sources))

        has_high = any(v["severity"] == "high" for v in violations)
        is_valid = len(violations) == 0

        if is_valid:
            return {
                "valid": True,
                "violations": [],
                "sanitized_output": response,
                "action_taken": "none",
            }

        if has_high and self.on_violation_action == "refuse":
            return {
                "valid": False,
                "violations": violations,
                "sanitized_output": (
                    "This response has been blocked due to safety policy violations. "
                    "Please rephrase your query."
                ),
                "action_taken": "refused",
            }

        # Default: redact PII and flag other issues
        sanitized = self._sanitize(response, violations)
        return {
            "valid": False,
            "violations": violations,
            "sanitized_output": sanitized,
            "action_taken": "redacted",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect personally identifiable information via regex."""
        violations = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append(
                    {
                        "validator": "pii",
                        "pii_type": pii_type,
                        "reason": f"Output contains {pii_type.replace('_', ' ')} (PII)",
                        "severity": "high",
                        "matches": matches,
                        "pattern": pattern,
                    }
                )
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """Detect harmful instructions or dangerous topic coverage."""
        violations = []
        text_lower = text.lower()

        for phrase in self.HARMFUL_PHRASES:
            if phrase in text_lower:
                violations.append(
                    {
                        "validator": "harmful_content",
                        "reason": f"Output may contain harmful instructional content",
                        "severity": "high",
                        "matched": phrase,
                    }
                )
                return violations  # One high-severity violation is sufficient

        for topic in self.HARMFUL_TOPICS:
            if topic in text_lower:
                violations.append(
                    {
                        "validator": "harmful_content",
                        "reason": f"Output references sensitive harmful topic: {topic}",
                        "severity": "medium",
                        "matched": topic,
                    }
                )

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Basic check that the response actually references sources that were retrieved.
        A simple heuristic: if many sources exist but none are cited, flag it.
        """
        violations = []
        if len(sources) >= 3:
            has_citation = any(
                marker in response
                for marker in ["[Source:", "[1]", "[2]", "http", "according to"]
            )
            if not has_citation:
                violations.append(
                    {
                        "validator": "citation_missing",
                        "reason": "Response uses multiple sources but contains no citations",
                        "severity": "low",
                    }
                )
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Redact PII matches; leave other content with a warning header."""
        sanitized = text
        for v in violations:
            if v.get("validator") == "pii":
                pii_type = v.get("pii_type", "PII")
                for match in v.get("matches", []):
                    sanitized = sanitized.replace(match, f"[REDACTED-{pii_type.upper()}]")
        return sanitized
