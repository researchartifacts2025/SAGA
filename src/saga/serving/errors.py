"""Error types used by the serving stack."""

from __future__ import annotations


class MissingRuntimeError(RuntimeError):
    """Raised when a serving feature is invoked but its runtime is absent.

    Examples:

    * Calling :class:`SagaVLLMEngine` without ``vllm`` installed.
    * Calling the Ray coordinator without ``ray`` installed.
    * Calling the CUDA prefetch path without a CUDA-capable build of torch.

    The string carries a hint about the install command needed to fix the
    situation; messages are stable so callers can match on them in tests.
    """

    def __init__(self, runtime: str, install_hint: str | None = None) -> None:
        msg = f"required runtime not available: {runtime}"
        if install_hint:
            msg = f"{msg}; install with: {install_hint}"
        super().__init__(msg)
        self.runtime = runtime
        self.install_hint = install_hint
