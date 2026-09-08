"""Request-scoped context shared by agents and their tools.

The model may suggest a ``user_id`` in a tool call, but it must not be able
to choose which user a request reads or writes.  The authenticated API layer
sets this context before entering the existing agent workflow; tools use the
bound value whenever it is available.  ContextVars also keep concurrent
requests isolated in the same process.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class RequestContext:
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None


_request_context: ContextVar[RequestContext] = ContextVar(
    "fitcoach_request_context", default=RequestContext()
)


def get_request_context() -> RequestContext:
    """Return the current request context, or an empty context for scripts/tests."""

    return _request_context.get()


def get_effective_user_id(requested_user_id: Optional[int] = None) -> Optional[int]:
    """Return the server-bound user id, falling back for direct tool tests."""

    bound_user_id = _request_context.get().user_id
    return bound_user_id if bound_user_id is not None else requested_user_id


@contextmanager
def request_context(
    user_id: Optional[int],
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Iterator[RequestContext]:
    """Bind request identity for the duration of an existing workflow call."""

    context = RequestContext(
        user_id=int(user_id) if user_id is not None else None,
        session_id=session_id,
        request_id=request_id,
    )
    token = _request_context.set(context)
    try:
        yield context
    finally:
        _request_context.reset(token)
