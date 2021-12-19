""" Python Data Science HW2.
    Student Name: 張育誠
    Student Id: 309704028
"""

# 1.
def f1(leader, hp_player, *args, **kwargs):
    print(f"Points Leader: {leader}")
    print(f"Hightest Paid Player: {hp_player}")
    print(f"Other Player: {args}")
    print(f"Stat Learders: {kwargs}", end="\n" * 2)


# 1. result
print(f'{"="*10} 1. result{"="*10}')
f1("Curry", "Curry", "James", Blocks_Leader="Turner", Assists_Leader="Westbrook")
f1(
    "Curry",
    "Curry",
    "James",
    "Lillard",
    "Harden",
    Rebounds_Leader="Capela",
    Blocks_Leader="Turner",
    Assists_Leader="Westbrook",
)

# 2.
def factorial(n: int):
    if not isinstance(n, int):
        raise TypeError("'n' must be int.")
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


# 2. result
print(f'{"="*10} 2. result{"="*10}')
print(f"{factorial(n=6)}", end="\n" * 2)

# 3.
def get_prime(n: int):
    """Get prime numbers that small than n."""
    assert n > 1, "n must grate than 1."

    prime_list = []
    for num in range(2, n):
        for number in range(2, num):
            if num % number == 0:
                break
        else:
            prime_list.append(num)
    return prime_list


# 3. result
print(f'{"="*10} 3. result{"="*10}')
print(f"{get_prime(n=72)}", end="\n" * 2)

# 4.
a = [2, -3, 3.3, 23, 78, 111, 0]


def is_odd_and_positive_int(x):
    return x % 2 == 1 and x > 0 and isinstance(x, int)


# 4.1 result
print(f'{"="*10} 4.1 result{"="*10}')
print(list(filter(is_odd_and_positive_int, a)), end="\n" * 2)
# 4.2 result
print(f'{"="*10} 4.2 result{"="*10}')
print([a_ for a_ in a if is_odd_and_positive_int(a_)], end="\n" * 2)

# 5.
list1 = ["Asia", "Alabama", "Arizona", "Aloha", "Colorado", "Montana", "Nevada"]
# 5. result
print(f'{"="*10} 5. result{"="*10}')
print(
    list(
        map(
            lambda name: len(
                [string for string in name if string == "A" or string == "a"]
            ),
            list1,
        )
    ),
    end="\n" * 2,
)

# 6.
def multiplier(num: int):
    _number = num

    def multiplication(number):
        nonlocal _number
        _number *= number
        return _number

    return multiplication


# 6. result
print(f'{"="*10} 6. result{"="*10}')
mutiply_with_5 = multiplier(5)
print(mutiply_with_5(9))
mutiply_with_7 = multiplier(7)
print(mutiply_with_7(5))
