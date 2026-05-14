"""Agent-framework integrations.

Three integrations ship out-of-the-box for the major agent frameworks:

* :class:`LangChainAdapter` --- attaches as a LangChain callback handler so
  SAGA observes ``on_chain_start`` / ``on_tool_end`` events and constructs
  :class:`~saga.core.aeg.AgentExecutionGraph` instances on the fly.
* :class:`AutoGenAdapter`   --- consumes AutoGen message logs (``role``,
  ``content``, ``tool_calls``).
* :class:`CrewAIAdapter`    --- consumes CrewAI per-step traces.

Each adapter has *no hard dependency* on its target framework. The adapters
accept duck-typed inputs so SAGA can be exercised without LangChain /
AutoGen / CrewAI installed; importing the adapter only fails if you call
the framework-specific ``attach()`` method while the framework is missing.
"""

from saga.integrations.autogen import AutoGenAdapter
from saga.integrations.crewai import CrewAIAdapter
from saga.integrations.langchain import LangChainAdapter


__all__ = ["AutoGenAdapter", "CrewAIAdapter", "LangChainAdapter"]
