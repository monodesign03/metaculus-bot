"""
Decides how much model power to spend per question so the credits last the
season.

On every run the bot reads the remaining OpenRouter credit, estimates how many
questions are still to come, and picks the best affordable plan from
bot_config.PLANS: a tier for the main tournament and a tier (or skip) for
MiniBench. The main tournament comes first because its prize pool is far
larger per question, and published results show that winning bots spend more
per question than the rest.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date

import requests

import bot_config as cfg

logger = logging.getLogger(__name__)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
FORCE_PLAN_ENV = "BOT_FORCE_TIERS"  # e.g. "standard,minimal" or "lean,skip"


@dataclass(frozen=True)
class Plan:
    main: cfg.Tier
    minibench: cfg.Tier | None  # None: MiniBench is skipped this run
    reason: str

    def describe(self) -> str:
        minibench = self.minibench.name if self.minibench else "skip"
        return f"main tournament '{self.main.name}', MiniBench '{minibench}'"


def estimated_cost_per_forecast(family: str) -> float:
    spec = cfg.MODELS[family]
    return (
        cfg.EST_INPUT_TOKENS_PER_FORECAST * spec.usd_per_m_input / 1_000_000
        + spec.est_output_tokens * spec.usd_per_m_output / 1_000_000
        + cfg.EST_HELPER_USD_PER_FORECAST
    )


def estimated_cost_per_question(tier: cfg.Tier | None) -> float:
    if tier is None:
        return 0.0
    cost = sum(estimated_cost_per_forecast(family) for family in tier.forecasters)
    return cost + cfg.EST_RESEARCH_USD[tier.research]


def _read_json(url: str, api_key: str, timeout: float) -> dict:
    response = requests.get(
        url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload) if isinstance(payload, dict) else {}


def remaining_openrouter_credit(
    api_key: str | None = None, timeout: float = 15
) -> float | None:
    """USD left to spend through this OpenRouter key, or None if unknown."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        key_info = _read_json(OPENROUTER_KEY_URL, key, timeout)
        if key_info.get("limit_remaining") is not None:
            return float(key_info["limit_remaining"])
    except Exception as error:  # network trouble must never stop forecasting
        logger.warning(f"Could not read OpenRouter key limit: {error}")
    try:
        credits = _read_json(OPENROUTER_CREDITS_URL, key, timeout)
        return float(credits["total_credits"]) - float(credits["total_usage"])
    except Exception as error:
        logger.warning(f"Could not read OpenRouter account credits: {error}")
    return None


def _tier_or_skip(name: str | None) -> cfg.Tier | None:
    if name is None or name.lower() in {"skip", "none", "off"}:
        return None
    return cfg.tier_by_name(name)


def _forced_plan() -> Plan | None:
    forced = os.getenv(FORCE_PLAN_ENV, "").strip()
    if not forced:
        return None
    names = [part.strip() for part in forced.split(",") if part.strip()]
    main = cfg.tier_by_name(names[0])
    minibench = _tier_or_skip(names[1] if len(names) > 1 else names[0])
    return Plan(main, minibench, f"forced by {FORCE_PLAN_ENV}={forced}")


def choose_plan(remaining_usd: float | None, today: date) -> Plan | None:
    """
    Returns the plan to use, or None when there is not even enough credit for
    one cheap forecast (in that case the run should stop cleanly).
    """
    forced = _forced_plan()
    if forced:
        return forced

    if remaining_usd is None:
        main_name, minibench_name = cfg.DEFAULT_PLAN_WHEN_CREDIT_UNKNOWN
        return Plan(
            cfg.tier_by_name(main_name),
            _tier_or_skip(minibench_name),
            "remaining credit unknown, using the conservative default",
        )

    days_left = max(cfg.MIN_PLANNING_DAYS, (cfg.SEASON_LAST_QUESTION_DAY - today).days)
    expected_main = days_left * cfg.MAIN_QUESTIONS_PER_DAY
    expected_minibench = days_left * cfg.MINIBENCH_QUESTIONS_PER_DAY
    spendable = remaining_usd - cfg.CREDIT_RESERVE_USD

    for main_name, minibench_name in cfg.PLANS:
        main = cfg.tier_by_name(main_name)
        minibench = _tier_or_skip(minibench_name)
        projected = expected_main * estimated_cost_per_question(
            main
        ) + expected_minibench * estimated_cost_per_question(minibench)
        if projected <= spendable:
            return Plan(
                main,
                minibench,
                f"${remaining_usd:.2f} left, ~{expected_main:.0f} main + "
                f"~{expected_minibench:.0f} MiniBench questions expected in "
                f"{days_left} days, projected spend ${projected:.2f}",
            )

    cheapest = cfg.TIERS[-1]
    if spendable >= estimated_cost_per_question(cheapest):
        return Plan(
            cheapest,
            None,
            f"only ${remaining_usd:.2f} left: cheapest tier, main tournament only",
        )
    return None
