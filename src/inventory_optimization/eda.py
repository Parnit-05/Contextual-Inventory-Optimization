"""Descriptive analyses and saved figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df, processed_dir, figures_dir):
    processed_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    df.groupby(["product_id", "product_name", "category"], as_index=False).demand_units.agg(["count", "sum", "mean", "std", "median", "min", "max"]).to_csv(processed_dir / "product_demand_summary.csv")
    df.groupby("category", as_index=False).demand_units.agg(["count", "sum", "mean", "std", "median"]).to_csv(processed_dir / "category_demand_summary.csv")
    df.groupby("date", as_index=False).demand_units.sum().to_csv(processed_dir / "daily_total_demand.csv", index=False)
    df.groupby("month", as_index=False).demand_units.agg(["mean", "median", "sum"]).to_csv(processed_dir / "monthly_demand_summary.csv")
    df.groupby("day_of_week", as_index=False).demand_units.agg(["mean", "median", "sum"]).to_csv(processed_dir / "weekday_demand_summary.csv")
    df.groupby("festival_name", as_index=False).demand_units.agg(["count", "mean", "median", "sum"]).to_csv(processed_dir / "festival_demand_summary.csv")
    df.groupby("promotion_indicator", as_index=False).agg(mean_demand=("demand_units", "mean"), median_demand=("demand_units", "median"), mean_discount=("discount_percentage", "mean"), rows=("demand_units", "size")).to_csv(processed_dir / "promotion_demand_summary.csv", index=False)
    df.groupby("supplier_id", as_index=False).agg(products=("product_id", "nunique"), mean_lead_time_days=("supplier_average_lead_time_days", "mean"), lead_time_std_days=("supplier_lead_time_std_days", "mean"), reliability=("supplier_reliability", "mean"), max_capacity_units=("supplier_max_capacity_units", "mean")).to_csv(processed_dir / "supplier_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5)); df.groupby("date").demand_units.sum().plot(ax=ax); ax.set(title="Total daily demand", ylabel="Units"); fig.tight_layout(); fig.savefig(figures_dir / "daily_demand_trend.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); sns.histplot(data=df, x="demand_units", bins=50, kde=True, ax=ax); ax.set(title="Demand distribution", ylabel="Observations"); fig.tight_layout(); fig.savefig(figures_dir / "demand_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); sns.boxplot(data=df, x="category", y="demand_units", ax=ax, showfliers=False); ax.tick_params(axis="x", rotation=20); ax.set(title="Demand by category", ylabel="Units"); fig.tight_layout(); fig.savefig(figures_dir / "demand_by_category.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); sns.boxplot(data=df, x="month", y="demand_units", ax=ax, showfliers=False); ax.set(title="Monthly demand distribution", ylabel="Units"); fig.tight_layout(); fig.savefig(figures_dir / "monthly_demand.png", dpi=150); plt.close(fig)
    seasonal = df.assign(season=df.date.dt.month.map(lambda m: "winter" if m in (12, 1, 2) else "summer" if m in (3, 4, 5) else "monsoon" if m in (6, 7, 8) else "autumn"))
    seasonal.groupby("season", as_index=False).demand_units.agg(["mean", "median", "sum"]).to_csv(processed_dir / "seasonal_demand_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.boxplot(data=seasonal, x="season", y="demand_units", order=["winter", "summer", "monsoon", "autumn"], showfliers=False, ax=ax); ax.set(title="Seasonal demand distribution", ylabel="Units"); fig.tight_layout(); fig.savefig(figures_dir / "seasonal_demand.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); sns.boxplot(data=df, x="day_of_week", y="demand_units", ax=ax, showfliers=False); ax.set(title="Weekly demand pattern", xlabel="Day of week (Monday=0)", ylabel="Units"); fig.tight_layout(); fig.savefig(figures_dir / "weekly_demand.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(data=df.groupby("festival_name", as_index=False).demand_units.mean(), x="festival_name", y="demand_units", ax=ax); ax.tick_params(axis="x", rotation=45); ax.set(title="Mean demand by festival/event", ylabel="Units"); fig.tight_layout(); fig.savefig(figures_dir / "festival_demand.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df.drop_duplicates("supplier_id"), x="supplier_average_lead_time_days", y="supplier_reliability", size="supplier_max_capacity_units", hue="supplier_id", ax=ax); ax.set(title="Supplier lead time, reliability and capacity"); fig.tight_layout(); fig.savefig(figures_dir / "supplier_profile.png", dpi=150); plt.close(fig)
