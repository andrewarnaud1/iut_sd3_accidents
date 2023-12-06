import pandas as pd
import matplotlib.pyplot as plt


victime = pd.read_csv("step1/merged_data.csv", low_memory=False)

nan_values = victime.isna().sum()

nan_values = nan_values.sort_values(ascending=True) * 100 / 127951

ax = nan_values.plot(
    kind="barh", figsize=(8, 10), color="#AF7AC5", zorder=2, width=0.85
)

ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

ax.tick_params(
    axis="both",
    which="both",
    bottom="off",
    top="off",
    labelbottom="on",
    left="off",
    right="off",
    labelleft="on",
)

vals = ax.get_xticks()

for tick in vals:
    ax.axvline(x=tick, linestyle="dashed", alpha=0.4, color="#eeeeee", zorder=1)


nans = [
    "v1",
    "v2",
    "lartpc",
    "larrout",
    "locp",
    "etatp",
    "actp",
    "voie",
    "pr1",
    "pr",
    "place",
]

victime = victime.drop(columns=nans)

victime = victime.dropna()

victime.corr()
victime.var()

victime = victime.drop(columns=["an"])

# Merger les dataframes dans un dataframe merged_data.csv
victime.to_csv("step2/missing_values_deleted.csv", index=False)

# Exporter les graphiques dans un dossier graphs
ax.figure.savefig("step2/missing_values.png", bbox_inches="tight")
