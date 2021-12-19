import datetime
import tkinter as tk
from tkcalendar import Calendar
from src.base_activity import ActivityList


class DailyView:
    def get_current_date(self, date: str) -> datetime.datetime:
        year, month, day = tuple(str(date).split("-"))
        year, month, day = int(year), int(month), int(day)
        return year, month, day

    def __call__(self, cal: Calendar, act_list: ActivityList) -> None:

        window = tk.Toplevel()
        current_date = cal.get_date()
        title = tk.Label(window, text=f"日期: {current_date}")
        activate_name = tk.Label(window, text="name")
        start_label = tk.Label(window, text="start")
        end_label = tk.Label(window, text="end")
        year, month, day = self.get_current_date(current_date)
        current_date = datetime.date(year, month, day)
        title.grid(row=0, column=0, columnspan=3, sticky="e")
        activate_name.grid(row=1, column=0, sticky="w", pady=5)
        start_label.grid(row=1, column=1, sticky="w", pady=5)
        end_label.grid(row=1, column=2, sticky="w", pady=5)

        # get current date id
        indices = cal.get_calevents(current_date)

        if indices:
            for i, event in enumerate(act_list.calevent):
                if event.id in indices:
                    text = event.text
                    start, end = event.date
                    lable_text = tk.Label(window, text=text)
                    label_start_time = tk.Label(
                        window, text=str(start.time())[:5]
                    )  # only hour:minutes
                    label_end_time = tk.Label(window, text=str(end.time())[:5])
                    lable_text.grid(row=i + 2, column=0, sticky="w", pady=5)
                    label_start_time.grid(row=i + 2, column=1, sticky="w", pady=5)
                    label_end_time.grid(row=i + 2, column=2, sticky="w", pady=5)
