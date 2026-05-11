"""
Input Guardrail
Checks user inputs for safety violations before they reach the agent pipeline.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """Guardrail for checking input safety."""

    # Harmful content keywords that should block processing
    TOXIC_KEYWORDS = [
        "how to kill", "how to murder", "bomb making", "make a bomb",
        "drug synthesis", "make drugs", "child abuse", "child exploitation",
        "self-harm method", "suicide method", "how to hurt",
        "illegal weapon", "human trafficking", "sex trafficking",
        "hack into", "steal credentials", "ddos attack",
    ]

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard your instructions",
        "forget your instructions",
        "forget everything above",
        "you are now",
        "pretend you are",
        "act as if you have no",
        "new system prompt",
        "system prompt:",
        "[system]",
        "jailbreak",
        "bypass safety",
        "ignore safety",
        "do anything now",
        "dan mode",
        "developer mode",
        "unrestricted mode",
        "print your system prompt",
        "reveal your instructions",
        "disregard previous",
        "override your",
    ]

    # Keywords that indicate on-topic queries (Ethical AI in Education / HCI)
    TOPIC_KEYWORDS = [
        # AI and ML
        "ai", "artificial intelligence", "machine learning", "deep learning",
        "neural network", "algorithm", "model", "chatbot", "llm", "generative",
        # Education
        "education", "learning", "student", "teacher", "classroom", "school",
        "university", "college", "curriculum", "assessment", "grading",
        "tutoring", "pedagogy", "academic", "course", "campus",
        # Ethics and fairness
        "ethical", "ethics", "bias", "fairness", "equity", "accountability",
        "transparency", "explainability", "trust", "responsible", "privacy",
        "data protection", "consent", "rights", "discrimination", "inclusion",
        # HCI and design
        "hci", "user experience", "ux", "interface", "usability", "design",
        "interaction", "accessibility", "tool", "platform", "software",
        # Research
        "research", "study", "paper", "framework", "methodology", "review",
        "analysis", "survey", "evaluation", "policy", "governance",
        # Related topics
        "technology", "digital", "online", "automation", "data", "system",
        "recommendation", "personalized", "adaptive", "proctoring",
        "detection", "surveillance", "monitoring",
    ]

    def __init__(self, config: Dict[str, Any]):
        """Initialize input guardrail from config."""
        self.config = config
        self.min_length = 10
        self.max_length = 1000

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query for safety and relevance.

        Returns a dict with:
          - valid: bool (False if any medium/high severity violation)
          - violations: list of violation dicts
          - sanitized_input: cleaned version of the query
        """
        violations: List[Dict[str, Any]] = []

        # 1. Length checks
        stripped = query.strip()
        if len(stripped) < self.min_length:
            violations.append({
                "validator": "length",
                "reason": f"Query too short (minimum {self.min_length} characters required)",
                "severity": "medium",
            })

        if len(query) > self.max_length:
            violations.append({
                "validator": "length",
                "reason": f"Query too long (maximum {self.max_length} characters allowed)",
                "severity": "medium",
            })

        # 2. Harmful content
        violations.extend(self._check_toxic_language(query))

        # 3. Prompt injection
        violations.extend(self._check_prompt_injection(query))

        # 4. Off-topic relevance (medium severity — blocks query)
        violations.extend(self._check_relevance(query))

        # Block on high or medium violations
        blocking_severities = {"high", "medium"}
        is_valid = not any(v["severity"] in blocking_severities for v in violations)

        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_input": stripped[: self.max_length],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """Detect toxic or harmful language using a keyword list."""
        text_lower = text.lower()
        for phrase in self.TOXIC_KEYWORDS:
            if phrase in text_lower:
                return [
                    {
                        "validator": "harmful_content",
                        "reason": "Query contains potentially harmful or dangerous language",
                        "severity": "high",
                    }
                ]
        return []

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect attempts to override or hijack the system prompt."""
        text_lower = text.lower()
        matched = [p for p in self.INJECTION_PATTERNS if p in text_lower]
        if matched:
            return [
                {
                    "validator": "prompt_injection",
                    "reason": "Query contains prompt injection patterns that attempt to override system instructions",
                    "severity": "high",
                    "matched_patterns": matched[:3],
                }
            ]
        return []

    # Core keywords that must appear for a query to be considered on-topic.
    # Intentionally short so broad/unrelated queries (e.g. recipes) are caught.
    CORE_TOPIC_KEYWORDS = [
        "ai", "artificial intelligence", "education", "student", "school",
        "learning", "teacher", "algorithm", "bias", "ethics", "data", "privacy",
    ]

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """Check whether the query is related to Ethical AI in Education or HCI research."""
        query_lower = query.lower()
        if not any(kw in query_lower for kw in self.CORE_TOPIC_KEYWORDS):
            return [
                {
                    "validator": "off_topic",
                    "reason": (
                        "Query appears off-topic. This system focuses on Ethical AI in Education "
                        "and Human-Computer Interaction research."
                    ),
                    "severity": "medium",
                }
            ]
        return []
