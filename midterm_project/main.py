"""Main."""


from functools import partial
import tkinter as tk
from tkcalendar import Calendar

from src.base_activity import ActivityList
from src._all_activity import ActivityMenu
from src._add_activity import AddActivity
from src._daily_view import DailyView
from src._countdown import CountDown
from src._search import SearchAcivity
from src._weekly_activity import WeeklyActivity
from src._national_holiday import make_holiday


def root_exit(root: tk.Tk):
    root.destroy()


# root
root = tk.Tk()
root.title("2021 NYCU Python Data Science Midterm Project")
root.geometry("1000x810")
root.resizable(0, 0)


# calendar
cal = Calendar(root, borderwidth=5, selectmode="day", date_pattern="y-mm-dd")
cal.config(
    font=("Times New Roman", 20, "bold"),
    selectforeground="black",
    selectbackground="yellow",
)
cal.config(weekenddays=[1, 7], weekendforeground="red")
cal.config(firstweekday="sunday", showothermonthdays=False, showweeknumbers=False)
cal.config(foreground="black", othermonthforeground="gray")

# holiday
holiday_list = make_holiday()
for date, text, tags in holiday_list:
    id_ = cal.calevent_create(date=date, text=text, tags=tags)

# tag foreground and background color
# Tags: "weekly", "national_holiday"
cal.tag_config("weekly", foreground="white", background="blue")
cal.tag_config("national_holiday", foreground="white", background="green")
cal.tag_config("", foreground="white", background="Plum")

# activity list
act_list = ActivityList()

# add activity


# functional
act_menu = ActivityMenu()
act_adder = AddActivity()
daily_menu = DailyView()
countdown_ = CountDown()
search_act = SearchAcivity()
weekly_act_ = WeeklyActivity()

# button
BUTTON_HEIGHT = 3
BUTTON_WIDTH = 30
button_font = ("Times New Roman", 20, "bold")
daily_view = tk.Button(
    text="Daily view",
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    font=button_font,
    command=partial(daily_menu, cal, act_list),
)
all_act = tk.Button(
    text="All activities", height=BUTTON_HEIGHT, width=BUTTON_WIDTH, font=button_font
)
all_act.config(command=partial(act_menu, cal, act_list, all_act))
countdown = tk.Button(
    text="Countdown",
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    font=button_font,
    command=partial(countdown_, act_list),
)
add_act = tk.Button(
    text="Add",
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    font=button_font,
    command=partial(act_adder, cal, act_list),
)
weekly_act = tk.Button(
    text="Weekly activities",
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    font=button_font,
    command=partial(weekly_act_, cal, act_list),
)
search = tk.Button(
    text="Search",
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    font=button_font,
    command=partial(search_act, act_list),
)
exit = tk.Button(
    text="Exit",
    height=BUTTON_HEIGHT,
    width=BUTTON_WIDTH,
    font=button_font,
    command=partial(root_exit, root),
)

# layout
cal.pack(side=tk.LEFT, fill=tk.BOTH)
daily_view.pack(fill=tk.BOTH)
all_act.pack(fill=tk.BOTH)
countdown.pack(fill=tk.BOTH)
add_act.pack(fill=tk.BOTH)
weekly_act.pack(fill=tk.BOTH)
search.pack(fill=tk.BOTH)
exit.pack(fill=tk.BOTH)


root.mainloop()
