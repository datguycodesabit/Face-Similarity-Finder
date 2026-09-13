class FaceMatchError(Exception):
    """Expected user-facing application error."""


class InvalidImageError(FaceMatchError):
    pass


class FaceCountError(FaceMatchError):
    def __init__(self, count: int) -> None:
        self.count = count
        if count == 0:
            message = "No face was found. Use a clear, front-facing photo with one visible face."
        else:
            message = f"Found {count} faces. Crop the image so exactly one face is visible."
        super().__init__(message)


class IndexNotReadyError(FaceMatchError):
    pass
