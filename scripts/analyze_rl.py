"""Compare no storage, rule-based control and the SAC agent on the 2019-2020
test years, for every community mix: summary tables and the README figures.

Run scripts/train_sac.py for each mix and seed first.

Usage: python scripts/analyze_rl.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_baselines import (FIG_DIR, GRID, INK, MEMBER_LABELS, MUTED, RESULTS_DIR, eur, legend,
                               load_run, n_years, style_axes, title)
from build_mixes import MIXES

SEEDS = [42, 7, 123]
STRATEGIES = {
    "no_storage": "No storage",
    "rbc": "Rule-based control",
    "sac": "RL agent (SAC)",
}
COLORS = {"no_storage": "#2a78d6", "rbc": "#eb6834", "sac": "#1baf7a"}
MIX_LABELS = {
    "full": "Full mix",
    "no_industry": "No industry",
    "homes_and_public": "Homes + public",
    "households_only": "Households only",
}
TEST_DIR = "test_2019_2020"
PERIOD = "Jan 2019 – Sep 2020, years the agent never saw in training"


def run_names(strategy):
    return [f"sac_seed{s}" for s in SEEDS] if strategy == "sac" else [strategy]


def community_row(cer, members):
    years = n_years(cer.index)
    consumption = sum(m["energy_consumed"].sum() for m in members.values())
    bill = cer["energy_expense"].sum() - cer["energy_revenue"].sum()
    tip = cer["tip_actual"].sum()
    return {
        "grid_import_mwh_per_year": cer["imported_energy"].sum() / years / 1000,
        "shared_energy_mwh_per_year": cer["selfconsumption"].sum() / years / 1000,
        "onsite_self_sufficiency_pct": 100 * (1 - cer["imported_energy"].sum() / consumption),
        "battery_discharge_mwh_per_year": sum(m["discharge"].sum() for m in members.values() if "discharge" in m) / years / 1000,
        "energy_bill_eur_per_year": bill / years,
        "incentive_eur_per_year": tip / years,
        "net_cost_eur_per_year": (bill - tip) / years,
    }


def load_all():
    """Community and per-member tables for every mix, strategy and seed."""
    community, members = [], []
    for mix in MIXES:
        for strategy in STRATEGIES:
            for name in run_names(strategy):
                cer, mem = load_run(f"{TEST_DIR}/{mix}/{name}")
                community.append({"mix": mix, "strategy": strategy, **community_row(cer, mem)})
                years = n_years(cer.index)
                for member_id, m in mem.items():
                    members.append({
                        "mix": mix, "strategy": strategy, "member": member_id,
                        "energy_bill_eur_per_year": (m["energy_expense"].sum() - m["energy_revenue"].sum()) / years,
                    })
    return pd.DataFrame(community), pd.DataFrame(members)


def plot_saving_breakdown(summary):
    """Bill saving and incentive change vs no storage, full mix."""
    full = summary[summary["mix"] == "full"].set_index("strategy")
    base = full.loc["no_storage"]
    rows = full.drop("no_storage")
    bill_saving = base["energy_bill_eur_per_year"] - rows["energy_bill_eur_per_year"]
    incentive_change = rows["incentive_eur_per_year"] - base["incentive_eur_per_year"]
    net = base["net_cost_eur_per_year"] - rows["net_cost_eur_per_year"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.18, bill_saving / 1000, 0.32, color="#4a3aa7", label="Lower electricity bill")
    ax.barh(y - 0.18, incentive_change / 1000, 0.32, color="#eda100", label="Change in incentive earned")
    for yi, name in zip(y, rows.index):
        ax.text(max(bill_saving[name], incentive_change[name], 0) / 1000 + 0.2, yi,
                f"net saving {eur(net[name])}/yr", va="center", fontsize=10, color=INK, weight="bold")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [STRATEGIES[n] for n in rows.index])
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("k€ per year vs. no storage", color=MUTED, fontsize=9)
    ax.set_xlim(min(incentive_change.min() / 1000, 0) - 0.6, max(bill_saving.max(), incentive_change.max()) / 1000 + 3.5)
    legend(ax, loc="lower right")
    title(fig, "Two routes to a lower net cost",
          f"Full mix, {PERIOD}. RL agent: mean of {len(SEEDS)} training seeds.\n"
          "Rule-based control cuts bills but gives up incentive; the RL agent keeps the incentive and adds to it.")
    fig.subplots_adjust(top=0.78, left=0.2, right=0.98, bottom=0.13)
    fig.savefig(FIG_DIR / "rl_saving_breakdown.png", dpi=160)
    plt.close(fig)


def plot_shared_by_hour():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    hours = np.arange(24)
    for strategy in STRATEGIES:
        cer, _ = load_run(f"{TEST_DIR}/full/{run_names(strategy)[0]}")
        days = cer.index.normalize().nunique()
        shared = cer["selfconsumption"].groupby(cer.index.hour).sum() / days
        ax.plot(hours, shared.reindex(hours, fill_value=0).values, color=COLORS[strategy], linewidth=2,
                marker="o", markersize=4, label=STRATEGIES[strategy])
    style_axes(ax)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("Shared energy (kWh, average day)", color=MUTED, fontsize=9)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper right")
    title(fig, "When the community shares energy",
          f"Full mix, {PERIOD}. Rule-based batteries soak up the morning surplus;\n"
          "the RL agent keeps the daytime sharing and stores PV to share again in the evening.")
    fig.subplots_adjust(top=0.8, left=0.08, right=0.98, bottom=0.08)
    fig.savefig(FIG_DIR / "rl_shared_by_hour.png", dpi=160)
    plt.close(fig)


def plot_diversity(community):
    """Net cost per mix and strategy, with the range over SAC seeds."""
    stats = community.groupby(["mix", "strategy"])["net_cost_eur_per_year"].agg(["mean", "min", "max"]) / 1000
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(MIXES))
    for strategy in STRATEGIES:
        s = stats.xs(strategy, level="strategy").reindex(list(MIXES))
        ax.plot(x, s["mean"], color=COLORS[strategy], linewidth=2, marker="o", markersize=7, label=STRATEGIES[strategy])
        if strategy == "sac":
            ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]],
                        fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for xi, v in zip(x, s["mean"]):
            ax.text(xi + 0.06, v, f"{v:.1f}", va="center", fontsize=8, color=INK)
    style_axes(ax)
    ax.set_xticks(x, [MIX_LABELS[m] for m in MIXES])
    ax.set_ylabel("Net cost (k€ per year)", color=MUTED, fontsize=9)
    ax.set_xlim(-0.3, len(MIXES) - 0.6)
    legend(ax, loc="upper left")
    title(fig, "Smart batteries make up for a less varied community",
          f"{PERIOD}. Mixes, left to right, have more members using electricity like households.\n"
          f"Net cost = bills minus incentive. Black whiskers: range over {len(SEEDS)} RL training seeds.")
    fig.subplots_adjust(top=0.8, left=0.08, right=0.98, bottom=0.08)
    fig.savefig(FIG_DIR / "diversity_net_cost.png", dpi=160)
    plt.close(fig)


def plot_member_bills(members):
    """Yearly bill change per member vs no storage, full mix."""
    full = members[members["mix"] == "full"]
    bills = full.groupby(["member", "strategy"])["energy_bill_eur_per_year"].mean().unstack()
    change = pd.DataFrame({s: bills["no_storage"] - bills[s] for s in ("rbc", "sac")})
    change = change.sort_values("rbc")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(change))
    ax.barh(y + 0.19, change["rbc"], 0.36, color=COLORS["rbc"], label=STRATEGIES["rbc"])
    ax.barh(y - 0.19, change["sac"], 0.36, color=COLORS["sac"], label=STRATEGIES["sac"])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, [MEMBER_LABELS.get(m, m) for m in change.index])
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Yearly saving on the member's own electricity bill (€)", color=MUTED, fontsize=9)
    legend(ax, loc="lower right")
    title(fig, "Who saves on their own bill",
          f"Full mix, {PERIOD}.\nThe RL agent's extra value is mostly incentive, paid to the community and not shown here.")
    fig.subplots_adjust(top=0.82, left=0.2, right=0.98, bottom=0.1)
    fig.savefig(FIG_DIR / "member_bills_rl.png", dpi=160)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    community, members = load_all()

    summary = community.groupby(["mix", "strategy"], sort=False).mean().reset_index()
    summary.round(1).to_csv(RESULTS_DIR / "summary_test_2019_2020.csv", index=False)

    member_summary = members.groupby(["mix", "strategy", "member"], sort=False).mean().reset_index()
    member_summary.round(1).to_csv(RESULTS_DIR / "summary_members_test_2019_2020.csv", index=False)

    print(summary.round(0).to_string(index=False))
    plot_saving_breakdown(summary)
    plot_shared_by_hour()
    plot_diversity(community)
    plot_member_bills(members)
    print("figures saved")


if __name__ == "__main__":
    main()
