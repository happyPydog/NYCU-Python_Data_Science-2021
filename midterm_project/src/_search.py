from functools import partial
import tkinter as tk
from src.base_activity import ActivityList


def search(act_list: ActivityList, search_entry: tk.Entry, window: tk.Toplevel):

    search_window = tk.Toplevel()
    search_label = tk.Label(search_window, text="search").grid(
        row=0, column=0, sticky="w", pady=5
    )
    activate_name = tk.Label(search_window, text="name").grid(
        row=1, column=0, sticky="w", pady=5
    )
    start_label = tk.Label(search_window, text="start").grid(
        row=1, column=1, sticky="e", pady=5
    )
    end_label = tk.Label(search_window, text="end").grid(
        row=1, column=2, sticky="e", pady=5
    )

    text = search_entry.get()

    for i, event in enumerate(act_list.calevent):
        if event.text == text:
            name = event.text
            start, end = event.date
            name_label = tk.Label(search_window, text=name)
            start_time_label = tk.Label(search_window, text=str(start)[:16])
            end_time_label = tk.Label(search_window, text=str(end)[:16])
            name_label.grid(row=2 + i, column=0, sticky="w", pady=5)
            start_time_label.grid(row=2 + i, column=1, sticky="e", pady=5)
            end_time_label.grid(row=2 + i, column=2, sticky="e", pady=5)

    window.destroy()


class SearchAcivity:
    def __call__(self, act_list: ActivityList) -> None:

        window = tk.Toplevel()
        search_label = tk.Label(window, text="search")
        search_entry = tk.Entry(window, width=10)
        search_button = tk.Button(
            window,
            text="search",
            command=partial(search, act_list, search_entry, window),
        )

        search_label.grid(row=0, column=0, sticky="w")
        search_entry.grid(row=0, column=1, sticky="w")
        search_button.grid(row=0, column=2, sticky="w")
