from typing import Tuple
import datetime
import tkinter as tk
from tkinter import messagebox
from functools import partial

from tkcalendar import Calendar, DateEntry
from src.base_activity import (
    ActivityList,
    Activity,
    ActivityDateError,
    ActivityTimeError,
    ActivityOverlappingError,
)


def get_id(cal: Calendar, start_date: datetime.datetime, text: str) -> Tuple[int, int]:
    start_id = cal.calevent_create(date=start_date, text=text, tags=["start"])
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


class ActivityMenu:
    def __call__(
        self, cal: Calendar, act_list: ActivityList, button: tk.Button
    ) -> None:
        window = tk.Toplevel()
        label_name = tk.Label(window, text="name")
        label_start = tk.Label(window, text="Start")
        label_end = tk.Label(window, text="end")
        label_modify = tk.Label(window, text="modify")
        label_delete = tk.Label(window, text="delet")
        label_name.grid(row=0, column=0)
        label_start.grid(row=0, column=1)
        label_end.grid(row=0, column=2)
        label_modify.grid(row=0, column=3)
        label_delete.grid(row=0, column=4)

        # act_list = [], then do nothing
        if act_list.calevent:
            # sort activity list
            act_list.sort_act()

            for idx, event in enumerate(act_list.calevent):
                id = event.id
                text = event.text
                start, end = event.date[0], event.date[1]
                label_name = tk.Label(window, text=text)
                label_start = tk.Label(window, text=start)
                label_end = tk.Label(window, text=end)
                globals()[f"label_modify_{idx}"] = tk.Button(
                    window,
                    text="mod",
                    command=partial(
                        modify_activity, event, cal, act_list, window, button
                    ),
                )
                globals()[f"label_delete_{idx}"] = tk.Button(
                    window,
                    text="del",
                    command=partial(del_activity, id, cal, act_list, window, button),
                )
                label_name.grid(row=idx + 1, column=0)
                label_start.grid(row=idx + 1, column=1)
                label_end.grid(row=idx + 1, column=2)
                globals()[f"label_modify_{idx}"].grid(row=idx + 1, column=3)
                globals()[f"label_delete_{idx}"].grid(row=idx + 1, column=4)


def modify_activity_(
    event: Activity,
    cal: Calendar,
    act_list: ActivityList,
    window: tk.Toplevel,
    modify_window: tk.Toplevel,
    button: tk.Button,
) -> None:

    ori_id = event.id
    ori_date = event.date
    ori_text = event.text

    cal.calevent_remove(ori_id)
    act_list.remove_event(ori_id)

    start = get_date(
        globals()[f"modify_start_date_entry"].get_date(),
        globals()[f"modify_start_hours_box"].get(),
        globals()[f"modify_start_minutes_box"].get(),
    )

    end = get_date(
        globals()[f"modify_end_date_entry"].get_date(),
        globals()[f"modify_end_hours_box"].get(),
        globals()[f"modify_end_minutes_box"].get(),
    )
    text = globals()[f"modify_add_event_entry"].get()
    id = get_id(cal, start, text)

    if text == "":
        warning_empty_name()
    else:
        try:
            act = Activity(id=id, date=(start, end), text=text)
            act_list.add_event(act)

        except ActivityDateError:
            msg = f"End time must be grater than start time."
            messagebox.showwarning(title="Time Error", message=msg)
            id = cal.calevent_create(date=ori_date[0], text=ori_text)
            act = Activity(id=id, date=ori_date, text=ori_text)
            act_list.calevent.append(act)

        except ActivityTimeError:
            msg = f"Activity date must be grater than or equal to today today."
            messagebox.showwarning(title="Date Error", message=msg)
            id = cal.calevent_create(date=ori_date[0], text=ori_text)
            act = Activity(id=id, date=ori_date, text=ori_text)
            act_list.calevent.append(act)

        except ActivityOverlappingError:
            msg = f"Activity time can not be Overlappping."
            messagebox.showwarning(title="Activity Overlappping", message=msg)
            id = cal.calevent_create(date=ori_date[0], text=ori_text)
            act = Activity(id=id, date=ori_date, text=ori_text)
            act_list.calevent.append(act)

    # close window and make new one
    window.destroy()
    modify_window.destroy()
    button.invoke()


def modify_activity(
    event: Activity,
    cal: Calendar,
    act_list: ActivityList,
    window: tk.Toplevel,
    button: tk.Button,
) -> None:

    modify_window = tk.Toplevel()
    globals()[f"modify_add_event_entry"] = tk.Entry(modify_window, width=10)
    globals()[f"modify_start_date_entry"] = DateEntry(modify_window)
    globals()[f"modify_end_date_entry"] = DateEntry(modify_window)
    modify_activate_label = tk.Label(modify_window, text="name:")
    modify_start_label = tk.Label(modify_window, text="start:")
    modify_end_label = tk.Label(modify_window, text="end:")

    start_h = tk.StringVar(value=12)
    start_m = tk.StringVar(value=0)
    end_h = tk.StringVar(value=12)
    end_m = tk.StringVar(value=0)
    globals()[f"modify_start_hours_box"] = tk.Spinbox(
        modify_window, width=2, from_=0, to=24, textvariable=start_h, wrap=True
    )
    globals()[f"modify_start_minutes_box"] = tk.Spinbox(
        modify_window, width=2, from_=0, to=59, textvariable=start_m, wrap=True
    )
    globals()[f"modify_end_hours_box"] = tk.Spinbox(
        modify_window, width=2, from_=0, to=23, textvariable=end_h, wrap=True
    )
    globals()[f"modify_end_minutes_box"] = tk.Spinbox(
        modify_window, width=2, from_=0, to=59, textvariable=end_m, wrap=True
    )
    modify_add_event_button = tk.Button(
        modify_window,
        text="mod",
        pady=5,
        padx=5,
        command=partial(
            modify_activity_, event, cal, act_list, window, modify_window, button
        ),
    )

    modify_activate_label.grid(row=0, sticky="w")
    globals()[f"modify_add_event_entry"].grid(row=0, column=1)
    modify_start_label.grid(row=1, sticky="w")
    modify_end_label.grid(row=2, sticky="w")
    globals()[f"modify_start_date_entry"].grid(row=1, column=1)
    globals()[f"modify_end_date_entry"].grid(row=2, column=1)
    globals()[f"modify_start_hours_box"].grid(row=1, column=2)
    globals()[f"modify_start_minutes_box"].grid(row=1, column=3)
    globals()[f"modify_end_hours_box"].grid(row=2, column=2)
    globals()[f"modify_end_minutes_box"].grid(row=2, column=3)
    modify_add_event_button.grid(row=3, columnspan=4)


def del_activity(
    id: int,
    cal: Calendar,
    act_list: ActivityList,
    window: tk.Toplevel,
    button: tk.Button,
) -> None:
    """Delet activity."""
    cal.calevent_remove(id)
    act_list.remove_event(id)
    window.destroy()
    button.invoke()
