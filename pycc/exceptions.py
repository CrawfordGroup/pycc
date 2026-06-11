"""
exceptions.py: PyCC-specific exception types.

Defining a small hierarchy lets callers catch PyCC input errors specifically
(``except PyCCError`` / ``except InvalidKeywordError``) instead of a bare
``Exception``, and standardizes the "not an allowed value" messages that were
previously hand-formatted at each validation site.
"""

from __future__ import annotations


class PyCCError(Exception):
    """Base class for all PyCC-specific errors."""


class PyCCWarning(UserWarning):
    """Base class for all PyCC-specific warnings.

    Subclassing :class:`UserWarning` lets callers silence or escalate PyCC
    warnings selectively, e.g. ``warnings.filterwarnings('error',
    category=PyCCWarning)``.
    """


class InvalidKeywordError(PyCCError, ValueError):
    """A keyword argument was given an unrecognized or invalid value.

    Also subclasses :class:`ValueError`, so existing ``except ValueError``
    handlers keep working.

    Parameters
    ----------
    keyword : str
        Name of the offending keyword (e.g. ``'model'``).
    value : object
        The value that was supplied.
    allowed : iterable
        The permitted values, used to build the message.
    """

    def __init__(self, keyword, value, allowed):
        self.keyword = keyword
        self.value = value
        self.allowed = list(allowed)
        allowed_str = ", ".join(repr(a) for a in self.allowed)
        super().__init__(
            "%r is not an allowed value for '%s'. Allowed values: %s."
            % (value, keyword, allowed_str)
        )
