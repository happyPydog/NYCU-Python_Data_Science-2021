import datetime
from typing import Tuple


class ActivityDateTimeError(Exception):
    """End time grater than start time error."""

    def __init__(
        self, value: Tuple[datetime.datetime, datetime.datetime], message: str
    ) -> None:
        self.value = value
        self.message = message
        super().__init__(message)


class ActivityOverlappingError(Exception):
    """Activity Overlapping error."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
