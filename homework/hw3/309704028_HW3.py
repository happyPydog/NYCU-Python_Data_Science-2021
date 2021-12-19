"""Homework 3."""


# 1.
try:
    print(X)  # get NameError
except NameError as e:
    print("you should define the variable.")

# 2.
import csv

with open("file.csv", newline="") as f:
    sentances = csv.reader(f)
    for sen in sentances:
        print(sen)

# 3.


class MyIter:
    def __init__(self, value=5, stop=50):
        self.value = value
        self.stop = stop

    def __iter__(self):
        return self

    def __next__(self):
        if self.value >= self.stop:
            raise StopIteration
        current = self.value
        self.value *= 2
        self.stop += 1
        return current


myiter = MyIter()
for value in myiter:
    print(value)

# 4.


class Midterm:
    def __init__(self, name: str) -> None:
        self.name = name

    def grade(self, english: float, math: float) -> None:
        print(f"{self.name} english grade is {english}, math grade is {math}")


mid = Midterm(name="Tony")
mid.grade(english=100, math=90)

# 5.


class Bank:
    def __init__(self) -> None:
        self._password = 1234

    @property
    def password(self) -> int:
        return self._password

    @password.setter
    def password(self, value: int) -> None:
        self._password = value


bank = Bank()
bank.password = 5678
print(f"lisa password: {bank.password}")
