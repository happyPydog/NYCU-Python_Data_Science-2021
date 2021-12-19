from typing import Tuple, Optional, List
import datetime
from functools import partial
import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar, DateEntry

from src.base_activity import (
    Activity,
    ActivityList,
    ActivityDateError,
    ActivityTimeError,
    ActivityOverlappingError,
)


def get_id(
    cal: Calendar,
    start_date: datetime.datetime,
    text: str,
    tags: Optional[List[str]] = "",
) -> Tuple[int, int]:
    start_id = cal.calevent_create(date=start_date, text=text, tags=tags)
    return start_id


def warning_empty_name():
    msg = f"Activity name can not be empty."
    messagebox.showwarning(title="Empty name", message=msg)


def get_date(date: datetime.datetime, hour: str, minute: str) -> datetime.datetime:
    year, month, day = tuple(str(date).split("-"))
    year, month, day, hour, minute = (
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
    )
    return datetime.datetime(year, month, day, hour, minute)


def activity_adder(
    cal,
    act_list,
    text_entry,
    start_date,
    start_hours,
    start_minutes,
    end_date,
    end_hours,
    end_minutes,
    window,
):
    text = text_entry.get()
    if text == "":
        warning_empty_name()
    else:
        start = get_date(start_date.get_date(), start_hours.get(), start_minutes.get())
        end = get_date(end_date.get_date(), end_hours.get(), end_minutes.get())
        id_ = get_id(cal, start, text)

        try:
            act = Activity(id=id_, date=(start, end), text=text)
            act_list.add_event(act)

        except ActivityDateError:
            msg = f"End time must be greater than start time."
            messagebox.showwarning(title="Time Error", message=msg)
            cal.calevent_remove(id_)

        except ActivityTimeError:
            msg = f"Activity date must be greater than or equal to today."
            messagebox.showwarning(title="Date Error", message=msg)
            cal.calevent_remove(id_)

        except ActivityOverlappingError:
            msg = f"Activity time can not be Overlappping."
            messagebox.showwarning(title="Activity Overlappping", message=msg)
            cal.calevent_remove(id_)

    # clear activity name entry
    text_entry.delete(0, "end")

    # close window
    window.destroy()


class AddActivity:
    def __call__(self, cal: Calendar, act_list: ActivityList) -> None:
        window = tk.Toplevel(
            width=300,
            height=70,
            bg="white",
            highlightthickness=3,
            highlightbackground="black",
        )
        add_event_entry = tk.Entry(window, width=20, borderwidth=3)
        start_date_entry = DateEntry(window)
        end_date_entry = DateEntry(window)
        activate_label = tk.Label(window, text="name:")
        start_label = tk.Label(window, text="start:")
        end_label = tk.Label(window, text="end:")

        # add hours and minutes
        start_h = tk.StringVar(value=12)
        start_m = tk.StringVar(value=0)
        end_h = tk.StringVar(value=12)
        end_m = tk.StringVar(value=0)
        start_hours_box = tk.Spinbox(
            window, width=2, from_=0, to=24, textvariable=start_h, wrap=True
        )
        start_minutes_box = tk.Spinbox(
            window, width=2, from_=0, to=59, textvariable=start_m, wrap=True
        )
        end_hours_box = tk.Spinbox(
            window, width=2, from_=0, to=23, textvariable=end_h, wrap=True
        )
        end_minutes_box = tk.Spinbox(
            window, width=2, from_=0, to=59, textvariable=end_m, wrap=True
        )

        # add button
        add_event_button = tk.Button(
            window,
            text="add",
            pady=5,
            padx=5,
            command=partial(
                activity_adder,
                cal,
                act_list,
                add_event_entry,
                start_date_entry,
                start_hours_box,
                start_minutes_box,
                end_date_entry,
                end_hours_box,
                end_minutes_box,
                window,
            ),
        )

        # layout
        activate_label.grid(row=0, sticky="w")
        add_event_entry.grid(row=0, column=1, columnspan=3)
        start_label.grid(row=1, sticky="w")
        end_label.grid(row=2, sticky="w")
        start_date_entry.grid(row=1, column=1)
        end_date_entry.grid(row=2, column=1)
        start_hours_box.grid(row=1, column=2)
        start_minutes_box.grid(row=1, column=3)
        end_hours_box.grid(row=2, column=2)
        end_minutes_box.grid(row=2, column=3)
        add_event_button.grid(row=3, columnspan=4)
