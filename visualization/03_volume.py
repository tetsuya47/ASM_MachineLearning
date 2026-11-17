import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd

from common import load_data, OUTPUT_DIR, add_chart_note

# Load source data
df = load_data()

# Calculate statistics for each stock
summary = (
    df.groupby("StockCode")
    .agg(
        Average_Volume=("Volume", "mean"),
        Median_Volume=("Volume", "median"),
        Maximum_Volume=("Volume", "max"),
        Minimum_Volume=("Volume", "min"),
        Trading_Sessions=("Volume", "count")
    )
    .sort_values("Average_Volume")
    .reset_index()
)

# --------------------------------------------------
# 1. Keep the original static PNG chart
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=summary,
    x="Average_Volume",
    y="StockCode",
    ax=ax
)

ax.set_title(
    "AVERAGE TRADING VOLUME\nMean daily traded shares by stock"
)
ax.set_xlabel("Average daily trading volume (shares)")
ax.set_ylabel("Stock code")
ax.grid(axis="x", alpha=0.25)

for container in ax.containers:
    ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)

add_chart_note(
    fig,
    df,
    "Mean calculated from available daily Volume observations"
)

fig.tight_layout(rect=(0, 0.04, 1, 1))

png_output = OUTPUT_DIR / "03_volume.png"
fig.savefig(png_output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved PNG:", png_output)

# --------------------------------------------------
# 2. Create interactive HTML chart with hover details
# --------------------------------------------------
custom_data = summary[
    [
        "Median_Volume",
        "Maximum_Volume",
        "Minimum_Volume",
        "Trading_Sessions"
    ]
].to_numpy()

fig_interactive = go.Figure()

fig_interactive.add_trace(
    go.Bar(
        x=summary["Average_Volume"],
        y=summary["StockCode"],
        orientation="h",
        text=summary["Average_Volume"].round(0),
        texttemplate="%{text:,.0f}",
        textposition="outside",
        customdata=custom_data,
        marker=dict(
            line=dict(width=0.5, color="white")
        ),
        name="Average volume",
        hovertemplate=(
            "<b>Stock: %{y}</b><br>"
            "Average volume: %{x:,.0f} shares<br>"
            "Median volume: %{customdata[0]:,.0f} shares<br>"
            "Maximum volume: %{customdata[1]:,.0f} shares<br>"
            "Minimum volume: %{customdata[2]:,.0f} shares<br>"
            "Trading sessions: %{customdata[3]:,.0f}<br>"
            "<extra>Trading volume statistics</extra>"
        )
    )
)

start_date = df["Date"].min().strftime("%d/%m/%Y")
end_date = df["Date"].max().strftime("%d/%m/%Y")

fig_interactive.update_layout(
    title=dict(
        text=(
            "AVERAGE TRADING VOLUME BY STOCK"
            "<br><sup>Move your mouse over each bar to inspect detailed statistics</sup>"
        ),
        x=0.5
    ),
    xaxis_title="Average daily trading volume (shares)",
    yaxis_title="Stock code",
    template="plotly_white",
    hovermode="closest",
    height=650,
    margin=dict(l=80, r=100, t=110, b=100),
    annotations=[
        dict(
            text=(
                f"Source: all_stocks_20241113_20261113.csv"
                f" | Period: {start_date} - {end_date}"
                f" | Stocks: {df['StockCode'].nunique()}"
                "<br>Average = mean daily volume | Median = middle observation"
                " | Max/Min = largest/smallest daily volume"
            ),
            xref="paper",
            yref="paper",
            x=0,
            y=-0.22,
            showarrow=False,
            align="left",
            font=dict(size=10, color="gray")
        )
    ]
)

fig_interactive.update_xaxes(
    separatethousands=True,
    showgrid=True
)

html_output = OUTPUT_DIR / "03_volume_interactive.html"

fig_interactive.write_html(
    str(html_output),
    include_plotlyjs="cdn",
    full_html=True
)

print("Saved interactive HTML:", html_output)
print("\nCOMPLETED!")
print("Hover over each bar in the HTML file to view detailed data.")
