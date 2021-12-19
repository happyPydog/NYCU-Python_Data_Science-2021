from typing import Optional, List, Tuple
import datetime
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass


class ActivityTimeError(Exception):
    """End time must be grater than start time."""

    def __init__(
        self, value: Tuple[datetime.datetime, datetime.datetime], message: str
    ) -> None:
        self.value = value
        self.message = message
        super().__init__(message)


class ActivityDateError(Exception):
    """Activity date must be grater than or equal to today today."""

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


class Activity(BaseModel):
    """Activity unit."""

    id: int
    date: Tuple[datetime.datetime, datetime.datetime]
    text: str
    tags: Optional[List[str]]

    @validator("date")
    @classmethod
    def end_must_grater_start(cls, value) -> None:
        start, end = value
        if end <= start:
            raise ActivityTimeError(
                value=value,
                message=f"End time({end}) must be grater than start time({start}).",
            )
        return value

    # @validator("date")
    # @classmethod
    # def date_must_grater_than_today(cls, value) -> None:
    #     start, _ = value
    #     if start.date() < datetime.date.today():
    #         raise ActivityDateError(
    #             value=value,
    #             message=f"Activity date({start.date()}) must be grater than or equal to today({datetime.date.today()}).",
    #         )
    #     return value


@dataclass
class ActivityList:
    """A class to store all activity."""

    calevent: List[Activity] = None

    def check_act_overlapping(self, event: Activity) -> None:
        """Check whether the event is overlapping."""
        start, end = event.date
        for evn in self.calevent:
            start_evn, end_evn = evn.date
            if (start == start_evn) or ((start > start_evn) and (start < end_evn)):
                raise ActivityOverlappingError(message="Activity time is overlapping.")
            elif (end == end_evn) or ((end > start_evn) and (end < end_evn)):
                raise ActivityOverlappingError(message="Activity time is overlapping.")

    def sort_act(self) -> None:
        """Sorted activity base on the start time."""
        self.calevent = sorted(self.calevent, key=lambda x: x.date[0])

    def add_event(self, event: Activity) -> None:
        if self.calevent is None:
            self.calevent = []
        self.check_act_overlapping(event)
        self.calevent.append(event)

    def remove_event(self, id: int) -> None:
        for idx, event in enumerate(self.calevent):
            if event.id == id:
                del self.calevent[idx]

    def modify_event(self, id: int, date: Activity, text: str) -> None:
        for idx, event in enumerate(self.calevent):
            if event.id == id:
                self.calevent[idx].date = date
                self.calevent[idx].text = text

    def search_event(self, text: str) -> List[int]:
        indices = []
        for idx, event in enumerate(self.calevent):
            if event.text == text:
                indices.append(idx)
        return indices
