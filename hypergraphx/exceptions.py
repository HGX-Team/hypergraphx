class HypergraphxError(Exception):
    """Base exception for hypergraphx errors."""


class MissingNodeError(ValueError, HypergraphxError):
    """Raised when a node is missing from a hypergraph."""


class MissingEdgeError(ValueError, HypergraphxError):
    """Raised when an edge is missing from a hypergraph."""


class InvalidFormatError(ValueError, HypergraphxError):
    """Raised when an invalid format is provided."""


class InvalidFileTypeError(ValueError, HypergraphxError):
    """Raised when an unsupported file extension is provided."""


class ReadwriteError(RuntimeError, HypergraphxError):
    """Raised when read/write operations fail."""


class InvalidParameterError(ValueError, HypergraphxError):
    """Raised when invalid or conflicting parameters are provided."""
