"""Workflow analyzer: framework-hint parser + observation-based AEG inference.

Three observability tiers, in order of preference:

* **Explicit hints**: agent framework callback metadata (LangChain / AutoGen /
  CrewAI). The :class:`FrameworkHintParser` converts these into an
  :class:`~saga.core.aeg.AgentExecutionGraph` directly.
* **Implicit traces**: when only the request stream is observable, the
  :class:`PatternInferenceEngine` extracts tool-type transitions, computes
  empirical transition probabilities, and keeps edges that pass the
  confidence threshold ``theta_conf = 0.7``.
* **Cold start**: a freshly-seen agent type is served as a request-level
  workload until ``cold_start_tasks = 30`` samples are observed, at which
  point pattern inference activates.
"""

from saga.workflow.analyzer import FrameworkHint, FrameworkHintParser
from saga.workflow.pattern import PatternInferenceEngine, PatternState


__all__ = [
    "FrameworkHint",
    "FrameworkHintParser",
    "PatternInferenceEngine",
    "PatternState",
]
