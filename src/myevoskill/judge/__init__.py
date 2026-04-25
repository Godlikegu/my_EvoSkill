"""Hidden judge bridge.

The judge is *not* visible to the agent. The harness invokes it after each
round and translates its output into the minimum signal needed by the agent
(PASS / FAIL / INVALID + optional per-metric pass mask).

We always invoke the judge in a separate Python subprocess so the judge's
imports (numpy, scipy, the task's own deps) cannot crash the harness.
"""

from .bridge import JudgeFeedback, JudgeRunner

__all__ = ["JudgeFeedback", "JudgeRunner"]
