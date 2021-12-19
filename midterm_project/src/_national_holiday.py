import datetime
from dataclasses import dataclass, field


@dataclass(order=True, frozen=True)
class MakeHoliday:
    _year_sort_idx: int = field(init=False, repr=False)
    _month_sort_idx: int = field(init=False, repr=False)
    _day_sort_idx: int = field(init=False, repr=False)
    year: int
    month: int
    day: int
    name: str = ""
    tag: str = "national_holiday"

    def __post_init__(self):
        object.__setattr__(self, "_year_sort_idx", self.year)
        object.__setattr__(self, "_month_sort_idx", self.month)
        object.__setattr__(self, "_day_sort_idx", self.day)

    def __str__(self):
        return f"{datetime.date(self.year, self.month, self.day)} {self.name}"

    def __call__(self):
        return [datetime.date(self.year, self.month, self.day), self.name, self.tag]


def make_national_holiday(
    year: int, month: int, day: int, *, tags: str = None, n_day: int = 1
):
    """Make a day or list to indicate the national holiday."""
    assert n_day >= 1, "'n_day' must grater than 1."
    if isinstance(tags, type(None)):
        tags = "national_holiday"

    day_list = []
    for d in range(n_day):
        day_list.append([datetime.date(year=year, month=month, day=day + d), tags])
    return day_list


def make_holiday():
    holiday_ = []
    holiday_.append(
        MakeHoliday(year=2021, month=1, day=1, name="2021元旦", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=10, name="小年夜", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=11, name="除夕", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=12, name="春節", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=13, name="初二", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=14, name="初三", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=15, name="初四", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=16, name="初五", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=16, name="初五", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=2, day=28, name="228", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=3, day=1, name="228", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=4, day=2, name="清明節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=4, day=5, name="清明節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=6, day=14, name="端午節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=9, day=20, name="中秋節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=9, day=21, name="中秋節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=10, day=10, name="雙十節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(year=2021, month=10, day=11, name="雙十節連假", tag="national_holiday")()
    )
    holiday_.append(
        MakeHoliday(
            year=2021, month=12, day=31, name="2022元旦", tag="national_holiday"
        )()
    )
    return holiday_
