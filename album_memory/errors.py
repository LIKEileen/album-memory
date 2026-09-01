class AlbumMemoryError(Exception):
    """Base domain error."""


class ConfigurationError(AlbumMemoryError):
    pass


class ConsentRequiredError(AlbumMemoryError):
    pass


class IdempotencyConflictError(AlbumMemoryError):
    pass


class ObservationConflictError(AlbumMemoryError):
    pass


class ProcessingError(AlbumMemoryError):
    pass


class ClaimTransitionError(AlbumMemoryError):
    pass
