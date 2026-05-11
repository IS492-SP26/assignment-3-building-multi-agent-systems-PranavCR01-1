"""
Safety Manager
Coordinates input and output guardrails and logs all safety events.
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os
from datetime import datetime
from pathlib import Path

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Central coordinator for all safety guardrails.

    Responsibilities:
      - Run InputGuardrail on every user query
      - Run OutputGuardrail on every agent response
      - Log every safety event to an in-memory list (and optionally to disk)
      - Expose statistics for the UI sidebar
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: The 'safety' section from config.yaml
                    (i.e. config["safety"], not the whole config).
        """
        self.config = config
        self.enabled: bool = config.get("enabled", True)
        self.log_events: bool = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # In-memory safety event log — shown in the Streamlit sidebar
        self.safety_events: List[Dict[str, Any]] = []

        self.prohibited_categories: List[str] = config.get(
            "prohibited_categories",
            ["harmful_content", "personal_attacks", "misinformation", "off_topic_queries"],
        )
        self.on_violation: Dict[str, Any] = config.get("on_violation", {})
        self.violation_action: str = self.on_violation.get("action", "sanitize")
        self.violation_message: str = self.on_violation.get(
            "message", "I cannot process this request due to safety policies."
        )

        # Sub-guardrails — pass the full safety config so each can read thresholds
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        # Optional file-based logging
        self._safety_log_path: Optional[str] = config.get("safety_log_file")
        if self._safety_log_path:
            Path(self._safety_log_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Validate a user query before sending it to the agent pipeline.

        Returns:
            {
                "safe": bool,
                "query": str (possibly sanitized),
                "violations": [...],
                "action": "allow" | "block",
            }
        """
        if not self.enabled:
            return {"safe": True, "query": query, "violations": [], "action": "allow"}

        validation = self.input_guardrail.validate(query)
        is_valid = validation["valid"]
        violations = validation["violations"]

        # Determine action
        has_high = any(v["severity"] == "high" for v in violations)
        if has_high or not is_valid:
            action = "block"
        else:
            action = "allow"

        is_safe = action == "allow"

        # Log every check (not just violations)
        self._log_safety_event("input", query, violations, is_safe)

        return {
            "safe": is_safe,
            "query": validation.get("sanitized_input", query),
            "violations": violations,
            "action": action,
        }

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate agent output before returning it to the user.

        Returns:
            {
                "safe": bool,
                "response": str (possibly sanitized or refused),
                "violations": [...],
                "action_taken": "none" | "redacted" | "refused",
            }
        """
        if not self.enabled:
            return {
                "safe": True,
                "response": response,
                "violations": [],
                "action_taken": "none",
            }

        validation = self.output_guardrail.validate(response, sources)
        is_valid = validation["valid"]
        violations = validation["violations"]
        action_taken = validation.get("action_taken", "none")

        is_safe = is_valid

        self._log_safety_event("output", response, violations, is_safe)

        # Respect the global on_violation config if the guardrail didn't already refuse
        final_response = validation.get("sanitized_output", response)
        if not is_safe and self.violation_action == "refuse" and action_taken != "refused":
            final_response = self.violation_message
            action_taken = "refused"

        return {
            "safe": is_safe,
            "response": final_response,
            "violations": violations,
            "action_taken": action_taken,
        }

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Return a copy of the full safety event log."""
        return list(self.safety_events)

    def get_safety_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the UI sidebar."""
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0.0,
        }

    def clear_events(self):
        """Clear the in-memory event log."""
        self.safety_events.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
    ):
        """Append a safety event to the in-memory list and optionally the log file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:120] + "..." if len(content) > 120 else content,
        }
        self.safety_events.append(event)

        if not is_safe:
            self.logger.warning(
                "Safety violation | type=%s | violations=%d", event_type, len(violations)
            )

        if self._safety_log_path and self.log_events:
            try:
                with open(self._safety_log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event) + "\n")
            except Exception as exc:
                self.logger.error("Failed to write safety log: %s", exc)
