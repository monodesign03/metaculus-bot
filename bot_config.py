"""
Settings for the ensemble bot. Everything worth tuning lives in this file.

Choices below follow the published evidence from past FutureEval seasons
(sources in the README): high reasoning effort beat standard effort in 8 of 8
model pairs; GPT-5.4 as the final model was the strongest single marker of a
good bot; scraping resolution links, checking similar questions/markets and
capping extreme probabilities were associated with better scores, while
extremizing was associated with worse ones.

Prices are USD per million tokens as listed on OpenRouter on 6 Oct 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# ---------------------------------------------------------------- season ----
# Tournaments are named by slug on purpose: the forecasting-tools version
# pinned in poetry.lock (0.2.92) still points its CURRENT_* constants at the
# Summer 2026 tournament. Update these three lines when a new season starts.
MAIN_TOURNAMENT = "fall-futureeval-2026"  # Fall 2026 FutureEval, $50k
MINIBENCH_TOURNAMENT = "minibench"  # rolling two-week rounds, $1k each
METACULUS_CUP_TOURNAMENT = "metaculus-cup-fall-2026"  # bots not prize-eligible

# Fall 2026 FutureEval: questions open from 28 Sep 2026 until "a few weeks
# before 1 Jan 2027". MiniBench runs back-to-back two-week rounds.
SEASON_LAST_QUESTION_DAY = date(2026, 12, 12)
MAIN_QUESTIONS_PER_DAY = 4.5  # 300-400 questions over ~11 weeks
MINIBENCH_QUESTIONS_PER_DAY = 4.3  # ~60 questions per two-week round
MIN_PLANNING_DAYS = 7  # after the season ends, still plan for a week ahead
CREDIT_RESERVE_USD = 2.0  # never plan to spend the last couple of dollars

# Bots are not prize-eligible in the Metaculus Cup, so by default the bot
# refuses to spend credits there. Set ENABLE_METACULUS_CUP=true to allow it.
METACULUS_CUP_ENV_FLAG = "ENABLE_METACULUS_CUP"

# ---------------------------------------------------------------- timing ----
# Main-tournament questions stay open for about 1.5 hours, so they always go
# first. MiniBench is processed in small batches, re-checking the main
# tournament between batches, and only for a limited time per run so a big
# MiniBench round can never block a main question.
MAX_QUESTIONS_IN_FLIGHT = 3
MINIBENCH_BATCH_SIZE = 3
MINIBENCH_SECONDS_PER_RUN = 15 * 60
RUN_DEADLINE_SECONDS = 42 * 60  # hard stop; unfinished questions retry next run
TEST_MODE_MAX_QUESTIONS = 4  # the Test Bot workflow forecasts at most this many


# ---------------------------------------------------------------- models ----
@dataclass(frozen=True)
class ModelSpec:
    openrouter_id: str  # used when OPENROUTER_API_KEY is set (preferred)
    direct_id: str  # litellm id when calling the provider directly
    direct_key_env: str  # env var with the provider's own key
    usd_per_m_input: float
    usd_per_m_output: float
    max_tokens: int  # caps reasoning + answer (Claude: ~80% thinking at high effort)
    est_output_tokens: int  # typical reasoning + answer at high effort


MODELS: dict[str, ModelSpec] = {
    "gpt": ModelSpec(
        openrouter_id="openrouter/openai/gpt-5.4",
        direct_id="openai/gpt-5.4",
        direct_key_env="OPENAI_API_KEY",
        usd_per_m_input=2.5,
        usd_per_m_output=15.0,
        max_tokens=24_000,
        est_output_tokens=9_000,
    ),
    "opus": ModelSpec(
        openrouter_id="openrouter/anthropic/claude-opus-5.5",
        direct_id="anthropic/claude-opus-5-5",
        direct_key_env="ANTHROPIC_API_KEY",
        usd_per_m_input=4.0,
        usd_per_m_output=20.0,
        max_tokens=20_000,
        est_output_tokens=8_000,
    ),
    "sonnet": ModelSpec(
        openrouter_id="openrouter/anthropic/claude-sonnet-5.5",
        direct_id="anthropic/claude-sonnet-5-5",
        direct_key_env="ANTHROPIC_API_KEY",
        usd_per_m_input=2.0,
        usd_per_m_output=10.0,
        max_tokens=20_000,
        est_output_tokens=8_000,
    ),
}
# When a family has no usable key, the first available family in this order
# takes its seat in the ensemble.
FAMILY_FALLBACK_ORDER = ("gpt", "opus", "sonnet")
REASONING_EFFORT = "high"
FALLBACK_REASONING_EFFORT = "medium"  # last-resort forecaster when all fail
FORECAST_TIMEOUT_SECONDS = 420

# Research models (OpenRouter web plugin with the provider's native search).
WEB_SCOUT_MODEL = "openrouter/anthropic/claude-sonnet-5.5"  # first broad pass
GAP_FILLER_MODEL = "openrouter/openai/gpt-5.4"  # second, targeted pass
WEB_RESEARCH_MAX_RESULTS = 8
WEB_RESEARCH_TIMEOUT_SECONDS = 240
MAX_WEB_RESEARCH_IN_FLIGHT = 3

# Cheap helper that turns free-text forecasts into numbers and summarises
# research for the posted comment.
HELPER_MODEL_OPENROUTER = "openrouter/openai/gpt-4o-mini"
HELPER_MODEL_OPENAI = "openai/gpt-4o-mini"
HELPER_MODEL_ANTHROPIC = "anthropic/claude-haiku-4-5-20251001"

# Size caps (characters) per research section, to keep input costs bounded.
RESEARCH_SECTION_CHAR_LIMITS = {
    "news": 10_000,
    "web": 6_000,
    "gaps": 6_000,
    "sources": 9_000,  # all resolution-source pages together
    "similar": 2_500,
    "markets": 1_500,
}
MAX_SOURCE_PAGES = 3
SOURCE_PAGE_CHAR_LIMIT = 3_000

# --------------------------------------------------------- post-processing ----
# Capping extremes protects against a single confident mistake (hallucination,
# misread criteria, stale news) wiping out many good forecasts.
BINARY_PROBABILITY_FLOOR = 0.03
BINARY_PROBABILITY_CEILING = 0.97
MULTIPLE_CHOICE_OPTION_FLOOR = 0.02  # 0.01 when there are more than 10 options

# ------------------------------------------------------- cost estimates ----
# Rough numbers used only for budgeting. Real usage is read back from
# OpenRouter on every run, so errors here self-correct over time.
EST_INPUT_TOKENS_PER_FORECAST = 11_000
EST_HELPER_USD_PER_FORECAST = 0.003
EST_RESEARCH_USD = {"full": 0.13, "light": 0.05}


# ----------------------------------------------------------------- tiers ----
@dataclass(frozen=True)
class Tier:
    name: str
    forecasters: tuple[str, ...]  # one high-effort prediction per entry
    research: str  # "full": scout + gap-filling search; "light": scout only


TIERS: tuple[Tier, ...] = (
    Tier("max", ("gpt", "opus", "gpt", "opus", "sonnet"), research="full"),
    Tier("strong", ("gpt", "opus", "gpt"), research="full"),
    Tier("standard", ("gpt", "opus"), research="full"),
    Tier("lean", ("gpt",), research="full"),
    Tier("minimal", ("gpt",), research="light"),
)

# Allowed (main tournament, MiniBench) combinations, best first. The planner
# takes the first one that the remaining credit can sustain for the rest of
# the season. The main tournament pays roughly 8x more per question, so
# MiniBench ("None" = skip) only gets credit once the main tournament is
# well funded.
PLANS: tuple[tuple[str, str | None], ...] = (
    ("max", "standard"),
    ("max", "minimal"),
    ("strong", "minimal"),
    ("standard", "minimal"),
    ("standard", None),
    ("lean", None),
    ("minimal", None),
)
# Used when the remaining credit cannot be read (e.g. direct provider keys).
DEFAULT_PLAN_WHEN_CREDIT_UNKNOWN = ("lean", None)
# Tier used by the "Test Bot" workflow: touches both main model families.
TEST_TIER = "standard"


def tier_by_name(name: str) -> Tier:
    for tier in TIERS:
        if tier.name == name:
            return tier
    raise ValueError(
        f"Unknown tier '{name}'. Known tiers: {[t.name for t in TIERS]}"
    )
