"""Summarise the no-storage and RBC baseline runs into small CSV tables and
the figures used in the README.

Run scripts/run_baselines.py first.

Usage: python scripts/analyze_baselines.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = REPO_ROOT / "results" / "simulations"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"
CONFIG_PATH = REPO_ROOT / "data" / "config" / "config.json"

STRATEGIES = {"no_storage": "No storage", "rbc": "Rule-based battery control"}
COLORS = {"no_storage": "#2a78d6", "rbc": "#eb6834"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

MEMBER_LABELS = {
    "dwelling1": "Dwelling 1",
    "dwelling2": "Dwelling 2 (no PV)",
    "apartment_block": "Apartment block",
    "school": "School",
    "town_hall": "Town hall",
    "factory": "Factory",
    "retail_shop": "Retail shop",
    "wholesale_biz": "Wholesale business",
    "farm": "Farm",
}

DAY_MEMBER = "apartment_block"
DAY_MONTH = "2019-07"


def load_run(name):
    """Community table and a dict of per-member tables for one run."""
    run_dir = SIM_DIR / name
    cer = pd.read_csv(run_dir / "cer_results.csv", parse_dates=["datetime"], index_col="datetime")
    members = {}
    for sub in ("prosumers", "consumers", "producers"):
        for path in sorted((run_dir / sub).glob("*_results.csv")):
            member_id = path.stem.removesuffix("_results")
            members[member_id] = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return cer, members


def n_years(index):
    return len(index) / 8760


def community_summary(runs):
    rows = []
    for name, (cer, members) in runs.items():
        years = n_years(cer.index)
        consumption = sum(m["energy_consumed"].sum() for m in members.values())
        bill = cer["energy_expense"].sum() - cer["energy_revenue"].sum()
        tip = cer["tip_actual"].sum()
        rows.append({
            "strategy": name,
            "consumption_mwh_per_year": consumption / years / 1000,
            "grid_import_mwh_per_year": cer["imported_energy"].sum() / years / 1000,
            "grid_export_mwh_per_year": cer["exported_energy"].sum() / years / 1000,
            "shared_energy_mwh_per_year": cer["selfconsumption"].sum() / years / 1000,
            "onsite_self_sufficiency_pct": 100 * (1 - cer["imported_energy"].sum() / consumption),
            "energy_bill_eur_per_year": bill / years,
            "incentive_eur_per_year": tip / years,
            "net_cost_eur_per_year": (bill - tip) / years,
        })
    return pd.DataFrame(rows).round(2)


def member_summary(runs):
    rows = []
    for name, (cer, members) in runs.items():
        years = n_years(cer.index)
        for member_id, m in members.items():
            rows.append({
                "strategy": name,
                "member": member_id,
                "consumption_kwh_per_year": m["energy_consumed"].sum() / years,
                "grid_import_kwh_per_year": m["energy_imported"].sum() / years,
                "energy_bill_eur_per_year": (m["energy_expense"].sum() - m["energy_revenue"].sum()) / years,
            })
    df = pd.DataFrame(rows)
    wide = df.pivot(index="member", columns="strategy", values="energy_bill_eur_per_year")
    wide["saving_eur_per_year"] = wide["no_storage"] - wide["rbc"]
    wide["saving_pct"] = 100 * wide["saving_eur_per_year"] / wide["no_storage"]
    return df.round(2), wide.round(2)


def eur(x):
    return f"€{x:,.0f}" if x >= 0 else f"−€{-x:,.0f}"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig, text, subtitle):
    fig.text(0.01, 0.98, text, ha="left", va="top", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def legend(ax, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK)


def plot_monthly_self_sufficiency(runs):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    months = np.arange(1, 13)
    for name, (cer, members) in runs.items():
        consumption = sum(m["energy_consumed"] for m in members.values())
        monthly = pd.DataFrame({"imp": cer["imported_energy"], "cons": consumption}).groupby(cer.index.month).sum()
        share = 100 * (1 - monthly["imp"] / monthly["cons"])
        ax.plot(months, share.values, color=COLORS[name], linewidth=2, marker="o", markersize=5, label=STRATEGIES[name])
        ax.annotate(f"{share.loc[7]:.0f}%", (7, share.loc[7]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color=INK)
    style_axes(ax)
    ax.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Demand covered on site (%)", color=MUTED, fontsize=9)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left")
    title(fig, "Share of demand met by members' own PV, by month",
          "Community total, average over Jan 2015 – Sep 2020. 100% minus grid imports as a share of consumption.")
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.08)
    fig.savefig(FIG_DIR / "monthly_self_sufficiency.png", dpi=160)
    plt.close(fig)


def plot_member_savings(wide):
    df = wide.sort_values("saving_eur_per_year")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y, df["saving_eur_per_year"], color=COLORS["rbc"], height=0.6)
    for i, (member_id, row) in enumerate(df.iterrows()):
        if row["saving_eur_per_year"] > 0:
            label = f"€{row['saving_eur_per_year']:,.0f}   (bill {eur(row['no_storage'])} → {eur(row['rbc'])})"
        else:
            label = "no battery"
        ax.text(row["saving_eur_per_year"] + df["saving_eur_per_year"].max() * 0.01, i, label,
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [MEMBER_LABELS.get(m, m) for m in df.index])
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Yearly saving on the electricity bill (€)", color=MUTED, fontsize=9)
    ax.set_xlim(0, df["saving_eur_per_year"].max() * 1.45)
    title(fig, "What rule-based battery control saves each member per year",
          "Bill = grid purchases minus sales, before → after adding the battery.\n"
          "The shared-energy incentive is paid to the community and isn't included here.")
    fig.subplots_adjust(top=0.8, left=0.2, right=0.98, bottom=0.1)
    fig.savefig(FIG_DIR / "member_savings.png", dpi=160)
    plt.close(fig)


def plot_shared_energy_and_incentive(runs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    width = 0.38
    for k, (name, (cer, _)) in enumerate(runs.items()):
        yearly = cer[["selfconsumption", "tip_actual"]].groupby(cer.index.year).sum()
        x = np.arange(len(yearly)) + (k - 0.5) * width
        axes[0].bar(x, yearly["selfconsumption"] / 1000, width, color=COLORS[name], label=STRATEGIES[name])
        axes[1].bar(x, yearly["tip_actual"] / 1000, width, color=COLORS[name], label=STRATEGIES[name])
    years = sorted(runs["no_storage"][0].index.year.unique())
    labels = [str(y) if y < 2020 else "2020*" for y in years]
    for ax, ylabel in zip(axes, ["Shared energy (MWh)", "Incentive earned (k€)"]):
        style_axes(ax)
        ax.set_xticks(np.arange(len(years)), labels)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.01, 0.87), ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK)
    title(fig, "Batteries cut the energy shared inside the community, and with it the incentive",
          "Shared energy = hourly overlap of members' exports and imports, which earns the CER incentive. "
          "*2020 covers Jan–Sep only.")
    fig.subplots_adjust(top=0.78, left=0.07, right=0.99, bottom=0.08, wspace=0.25)
    fig.savefig(FIG_DIR / "shared_energy_and_incentive.png", dpi=160)
    plt.close(fig)


def plot_summer_day(runs):
    m = runs["rbc"][1][DAY_MEMBER]
    month = m.loc[DAY_MONTH]
    daily_pv = month["energy_produced"].resample("D").sum()
    weekdays = daily_pv[daily_pv.index.dayofweek < 5]
    day = weekdays.idxmax().strftime("%Y-%m-%d")
    d = m.loc[day]
    hours = d.index.hour

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(hours, d["energy_produced"], color="#eda100", linewidth=2, label="PV production")
    ax1.plot(hours, d["energy_consumed"], color=INK, linewidth=2, label="Consumption")
    ax1.bar(hours, d["charge"], 0.7, color="#1baf7a", label="Battery charging")
    ax1.bar(hours, -d["discharge"], 0.7, color="#4a3aa7", label="Battery discharging")
    ax1.axhline(0, color=MUTED, linewidth=0.8)
    ax1.set_ylabel("kWh in the hour", color=MUTED, fontsize=9)
    legend(ax1, loc="upper left", ncol=2)

    with open(CONFIG_PATH) as f:
        capacity = json.load(f)["Prosumers"][DAY_MEMBER]["Storage"]["capacity"]
    soc = 100 * d["soc_final"] / capacity
    ax2.plot(hours, soc, color="#1baf7a", linewidth=2)
    ax2.fill_between(hours, soc, color="#1baf7a", alpha=0.15)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Battery charge (%)", color=MUTED, fontsize=9)
    ax2.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    for ax in (ax1, ax2):
        style_axes(ax)

    date_label = pd.Timestamp(day).strftime("%A %d %B %Y")
    title(fig, "A summer day for the apartment block's battery",
          f"{date_label}, rule-based control. The battery stores the midday PV surplus and covers the evening load.")
    fig.subplots_adjust(top=0.87, left=0.09, right=0.98, bottom=0.07, hspace=0.12)
    fig.savefig(FIG_DIR / "summer_day_battery.png", dpi=160)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    runs = {name: load_run(name) for name in STRATEGIES}

    community = community_summary(runs)
    members, member_savings = member_summary(runs)
    community.to_csv(RESULTS_DIR / "summary_community.csv", index=False)
    members.to_csv(RESULTS_DIR / "summary_members.csv", index=False)
    print(community.to_string(index=False))
    print(member_savings.to_string())

    plot_monthly_self_sufficiency(runs)
    plot_member_savings(member_savings)
    plot_shared_energy_and_incentive(runs)
    plot_summer_day(runs)
    print(f"figures saved to {FIG_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
