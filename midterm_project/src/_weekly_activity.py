from typing import Tuple, Optional, List
import datetime
from functools import partial
import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar, DateEntry
from src.base_activity import (
    ActivityList,
    Activity,
    ActivityDateError,
    ActivityOverlappingError,
    ActivityTimeError,
)


def get_id(
    cal: Calendar, start_date: datetime.datetime, text: str, tags: Optional[List[str]]
) -> Tuple[int, int]:
    id = cal.calevent_create(date=start_date, text=text, tags=tags)
    return id


def warning_empty_name():
    msg = f"Activity name can not be empty."
    messagebox.showwarning(title="Empty name", message=msg)


def get_date(date: datetime.datetime, hour: str) -> datetime.datetime:
    year, month, day = tuple(str(date).split("-"))
    year, month, day, hour = (
        int(year),
        int(month),
        int(day),
        int(hour),
    )
    return datetime.datetime(year, month, day, hour)


def weekly_adder(
    cal: Calendar,
    act_list: ActivityList,
    routine_entry: tk.Entry,
    routine_data_entry: DateEntry,
    routine_start_time_box: tk.Spinbox,
    routine_end_time_box: tk.Spinbox,
    routine_weeks_box: tk.Spinbox,
    window: tk.Toplevel,
) -> None:

    weeks = routine_weeks_box.get()
    for week in range(int(weeks)):

        text = routine_entry.get()

        if text == "":
            warning_empty_name()

        date = routine_data_entry.get_date() + datetime.timedelta(days=week * 7)

        start = get_date(date, routine_start_time_box.get())
        end = get_date(date, routine_end_time_box.get())
        tags = ["weekly"]
        id = get_id(cal, start, text, tags)

        try:
            act = Activity(id=id, date=(start, end), text=text, tags=tags)
            act_list.add_event(act)

        except ActivityDateError:
            msg = f"End time must be grater than start time."
            messagebox.showwarning(title="Time Error", message=msg)
            cal.calevent_remove(id)

        except ActivityTimeError:
            msg = f"Activity date must be grater than or equal to today today."
            messagebox.showwarning(title="Date Error", message=msg)
            cal.calevent_remove(id)

        except ActivityOverlappingError:
            msg = f"Activity time can not be Overlappping."
            messagebox.showwarning(title="Activity Overlappping", message=msg)
            cal.calevent_remove(id)

    # close window
    window.destroy()


class WeeklyActivity:
    def __call__(self, cal: Calendar, act_list: ActivityList) -> None:
        window = tk.Toplevel()
        routine_start_h = tk.StringVar(value=12)
        routine_end_h = tk.StringVar(value=12)
        routine_weeks = tk.StringVar(value=4)
        routine_name_label = tk.Label(window, text="name")
        routine_period_date_label = tk.Label(window, text="date")
        routine_period_time_label = tk.Label(window, text="time")
        routine_period_times_label = tk.Label(window, text="reapeat")
        routine_label_ = tk.Label(window, text="~")
        routine_label_weeks = tk.Label(window, text="maintain")
        routine_entry = tk.Entry(window, width=10, borderwidth=3)
        routine_data_entry = DateEntry(window, width=8)
        routine_start_time_box = tk.Spinbox(
            window, width=2, from_=0, to=24, textvariable=routine_start_h, wrap=True
        )
        routine_end_time_box = tk.Spinbox(
            window, width=2, from_=0, to=24, textvariable=routine_end_h, wrap=True
        )
        routine_weeks_box = tk.Spinbox(
            window, width=2, from_=1, to=53, textvariable=routine_weeks, wrap=True
        )
        routine_button = tk.Button(
            window,
            text="add",
            command=partial(
                weekly_adder,
                cal,
                act_list,
                routine_entry,
                routine_data_entry,
                routine_start_time_box,
                routine_end_time_box,
                routine_weeks_box,
                window,
            ),
        )
        routine_name_label.grid(row=0, sticky="w", padx=5, pady=2)
        routine_period_date_label.grid(row=1, sticky="w", padx=5, pady=2)
        routine_period_time_label.grid(row=2, sticky="w", padx=5, pady=2)
        routine_period_times_label.grid(row=3, sticky="w", padx=5, pady=2)
        routine_entry.grid(row=0, column=1, padx=5, pady=5)
        routine_data_entry.grid(row=1, column=1, padx=5, pady=2)
        routine_start_time_box.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        routine_end_time_box.grid(row=2, column=1, padx=5, pady=2, sticky="e")
        routine_label_.grid(row=2, column=1, padx=5, pady=2)
        routine_label_weeks.grid(row=3, column=1, padx=5, pady=2, sticky="e")
        routine_weeks_box.grid(row=3, column=1, padx=5, pady=2, sticky="w")
        routine_button.grid(row=4, columnspan=4, padx=5, pady=2)
