import math
import tkinter as tk
import datetime
from src.base_activity import ActivityList


class CountDown:
    def __call__(self, act_list: ActivityList) -> None:
        window = tk.Toplevel()
        label_rest_time = tk.Label(window, text="countdown_table")
        label_name = tk.Label(window, text="name")
        label_rest_days = tk.Label(window, text="day")
        label_rest_hours = tk.Label(window, text="hour")
        label_rest_time.grid(row=0, column=0)
        label_name.grid(row=1, column=0)
        label_rest_days.grid(row=1, column=1)
        label_rest_hours.grid(row=1, column=2)

        # today
        current_datetime = datetime.datetime.now()

        # sort activity
        act_list.sort_act()

        # countdown
        for i, event in enumerate(act_list.calevent):

            # date
            start, end = event.date

            # text
            text = event.text

            diff_time = start - current_datetime
            day_text, hour_text = diff_time.days, math.floor(diff_time.seconds / 3600)

            # activity passed
            if diff_time.days < 0:
                # day_text, hour_text = "Passed", "Passed"
                continue

            # end in an hour
            elif hour_text == 0:
                hour_text = "<1"

            text_label = tk.Label(window, text=text)
            day_label = tk.Label(window, text=day_text)
            hour_label = tk.Label(window, text=hour_text)
            text_label.grid(row=2 + i, column=0)
            day_label.grid(row=2 + i, column=1)
            hour_label.grid(row=2 + i, column=2)
