"""Python資料科學 Python Data Science Homework 4."""

#%%
# 1.
import numpy as np


def replace_odd(array):
    out = [num if num % 2 == 0 else -1 for num in array]
    return np.array(out).reshape(-1, 2)


print(f"{'='*20} 1. {'='*20}")
ary = np.arange(30, 0, -1)
print(repr(ary))
print(repr(replace_odd(ary)))

#%%
# 2.
import numpy as np


def l1_distance(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    return np.sum(np.absolute(x1 - x2))


def l2_distance(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    return np.sqrt(np.sum((x1 - x2) ** 2))


a = [1, 2, 3, 4, 5]
b = [4, 5, 6, 7, 8]
print(f"{'='*20} 2. {'='*20}")
print(f"L1 distance = {l1_distance(a, b)}")
print(f"L2 distance = {l2_distance(a, b)}")


# %%
# 3.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "First_name": ["allen", "johnny", "Chloe", "John", "Alice", "Bob"],
    "Last_name": ["lin", "Lin", "Huang", "Chen", "Chang", "Wang"],
    "Gender": ["M", "M", "F", "M", "F", "M"],
    "Height(inch)": [72, 69, 63, 62, 57, 69],
    "Weight(lbs)": [130, 205, 180, 125, 89, 160],
}
df = pd.DataFrame(data)

# 3-1
# Combine ‘First_name’and ‘Last_name’into ‘Name
print(f"{'='*20} 3-1 {'='*20}")
df["Name"] = df["First_name"].combine(
    df["Last_name"], lambda s1, s2: s1.capitalize() + " " + s2.capitalize()
)
# Set ‘Name’as index
df.set_index("Name", inplace=True)
# Drop ‘First_name’and ‘Last_name’
df.drop(["First_name", "Last_name"], axis=1, inplace=True)
print(df)

# 3-2
# Calculate the BMIand addit as a new column.
print(f"{'='*20} 3-2 {'='*20}")
df["Height(inch)"] = df["Height(inch)"].values * 0.0254
df["Weight(lbs)"] = df["Weight(lbs)"].values * 0.45359237
df = df.rename(
    columns={"Height(inch)": "Height(m)", "Weight(lbs)": "Weight(kg)"}
).round(2)
df["BMI"] = np.round(df["Weight(kg)"] / df["Height(m)"] ** 2, 2)
print(df)

# 3-3
# Create a new featureto display physical condition based on BMI
def bmi_filter(bmi):
    if bmi < 18.5:
        return "Light"
    elif 18.5 <= bmi < 24:
        return "Normal"
    else:
        return "Heavy"


df["State"] = [bmi_filter(bmi) for bmi in df["BMI"].values]
print(f"{'='*20} 3-3 {'='*20}")
print(df)

# 3-4
# Draw the scatter plot group by gender.
male, female = df[df["Gender"] == "M"], df[df["Gender"] == "F"]

rel = sns.relplot(
    data=df,
    x="Weight(kg)",
    y="Height(m)",
    col="Gender",
    hue="BMI",
    kind="scatter",
    s=200,
    alpha=0.5,
)
ax1, ax2 = rel.axes[0, 0], rel.axes[0, 1]
for i, text in enumerate(male["BMI"]):
    ax1.text(
        male["Weight(kg)"][i],
        male["Height(m)"][i],
        text,
        horizontalalignment="center",
    )
for i, text in enumerate(female["BMI"]):
    ax2.text(
        female["Weight(kg)"][i],
        female["Height(m)"][i],
        text,
        horizontalalignment="center",
    )
rel.fig.subplots_adjust(top=0.8)
rel.fig.suptitle("BMI (group by Gender)")
plt.show()

# 3-5
# Draw the bar chart of BMI.
df.sort_values(by=["BMI"], inplace=True, ascending=False)
paletter = {"Light": "tab:green", "Normal": "tab:blue", "Heavy": "tab:red"}
sns.barplot(
    x="BMI", y=df.index.values, data=df, hue="State", palette=paletter, alpha=0.5
)
plt.axvline(x=18.5, c="k")
plt.axvline(x=24, c="k")
plt.title("BMI")
plt.tight_layout()
plt.show()
