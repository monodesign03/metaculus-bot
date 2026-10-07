"""
Offline checks for the ensemble bot: no Metaculus, no real LLM, no money.

A fake OpenAI-compatible server stands in for OpenRouter and a fake
`requests.get` stands in for web pages, the Metaculus API and Manifold, so
the whole pipeline runs for real: research, ensemble predictions, the
template's parsing, aggregation, and the posted explanation.

Run from the repository root:
    poetry run python tests/test_offline.py
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORECAST_BY_MODEL = {  # binary probability (%) each fake model answers with
    "openai/gpt-5.4": 30,
    "anthropic/claude-opus-5.5": 20,
    "anthropic/claude-sonnet-5.5": 26,
}
FAILING_MODELS: set[str] = set()  # forecast calls to these models return HTTP 400
REQUESTS: list[dict] = []
HTTP_GETS: list[str] = []


# ------------------------------------------------------------ fake models ----
def _flatten(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(block.get("text", "") for block in content if isinstance(block, dict))
    return "\n".join(parts)


def _parser_answer(prompt: str) -> str:
    text = prompt.split("START TEXT", 1)[1].split("END TEXT", 1)[0]
    wanted = prompt.rsplit("Please convert the text to the following type:", 1)[1][:300]
    if "BinaryPrediction" in wanted:
        value = float(re.findall(r"Probability:\s*([\d.]+)%", text)[-1]) / 100
        return json.dumps({"prediction_in_decimal": value})
    if "PredictedOptionList" in wanted:
        options = [
            {"option_name": name.strip(), "probability": float(pct) / 100}
            for name, pct in re.findall(r"^\s*([^:\n]+):\s*([\d.]+)%\s*$", text, re.M)
        ]
        return json.dumps({"predicted_options": options})
    if "DatePercentile" in wanted:
        rows = re.findall(r"Percentile (\d+):\s*(\d{4}-\d{2}-\d{2})", text)
        return json.dumps([{"percentile": int(p) / 100, "value": f"{d}T00:00:00Z"} for p, d in rows])
    if "Percentile" in wanted:
        rows = re.findall(r"Percentile (\d+):\s*([\d.]+)", text)
        return json.dumps([{"percentile": int(p) / 100, "value": float(v)} for p, v in rows])
    raise AssertionError(f"Unexpected parse target: {wanted!r}")


def _answer(body: dict) -> str:
    model = body["model"]
    prompt = _flatten(body["messages"])
    if "You are a data analyst helping to convert text into structured data" in prompt:
        return _parser_answer(prompt)
    if "research analyst supporting a professional forecaster" in prompt:
        assert "Research so far" in prompt
        return "Open point: latest value. Found: 41 units on 2026-10-05 ([source](https://example.com/latest))."
    if "research assistant of a professional forecaster" in prompt:
        return "1. Current status: nothing decisive yet ([source](https://example.com/news), 2026-10-01)."
    if "Please summarize the following research" in prompt:
        return "The research finds nothing decisive yet."
    if model in FAILING_MODELS:
        raise RuntimeError(f"{model} is down")
    if 'Probability: ZZ%' in prompt:
        return f"(a) Two weeks left. (c) Status quo is No.\nProbability: {FORECAST_BY_MODEL[model]}%"
    if "Option_A: Probability_A" in prompt:
        options = ast.literal_eval(re.search(r"The options are: (\[.*?\])", prompt).group(1))
        shares = [50] + [round(50 / (len(options) - 1), 4)] * (len(options) - 1)
        return "\n".join(f"{name}: {share}%" for name, share in zip(options, shares))
    percentiles = [5, 10, 20, 40, 60, 80, 90, 95]
    if "Percentile 5: YYYY-MM-DD" in prompt:
        days = ["2026-10-25", "2026-11-01", "2026-11-15", "2026-12-01", "2026-12-20",
                "2027-01-15", "2027-02-10", "2027-03-01"]
        return "\n".join(f"Percentile {p}: {d}" for p, d in zip(percentiles, days))
    if "Percentile 5: XX" in prompt:
        offset = {"openai/gpt-5.4": 0, "anthropic/claude-opus-5.5": 15}.get(model, 5)
        values = [v + offset for v in (15, 20, 30, 40, 50, 60, 70, 75)]
        return "\n".join(f"Percentile {p}: {v}" for p, v in zip(percentiles, values))
    raise AssertionError(f"Fake server got an unexpected prompt: {prompt[:200]!r}")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        REQUESTS.append(body)
        try:
            content = _answer(body)
            status = 200
            payload = {
                "id": "fake",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }
        except Exception as error:  # surfaces as an API error to the bot
            status, payload = 400, {"error": {"message": f"fake server: {error}"}}
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start_fake_openrouter() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    os.environ["OPENROUTER_API_BASE"] = f"http://127.0.0.1:{server.server_port}/api/v1"


# -------------------------------------------------------------- fake web ----
DATA_PAGE = """<html><head><title>Stats</title><script>var secret = "IGNORE ME";</script>
<style>.x{color:red}</style></head><body><nav>Home | About</nav>
<h1>Weekly widget count</h1><p>Updated 2026-10-04.</p>
<table><tr><th>Week</th><th>Widgets</th></tr><tr><td>2026-09-27</td><td>38</td></tr>
<tr><td>2026-10-04</td><td>41</td></tr></table><footer>Copyright</footer></body></html>"""

SIMILAR_POSTS = {
    "results": [
        {"title": "Will widgets exceed 40 by Sep 1?", "question": {
            "type": "binary", "resolution": "no", "actual_resolve_time": "2026-09-02T10:00:00Z"}},
        {"title": "Will widgets exceed 35 by Aug 1?", "question": {
            "type": "binary", "resolution": "yes", "actual_resolve_time": "2026-08-02T10:00:00Z"}},
        {"title": "Widget count in July?", "question": {
            "type": "numeric", "resolution": "36.40000001", "actual_resolve_time": "2026-08-01T00:00:00Z"}},
        {"title": "Annulled one", "question": {"type": "binary", "resolution": "annulled"}},
        {"title": "A group", "group_of_questions": {"questions": []}},
    ]
}

MANIFOLD_MARKETS = [
    {"question": "Will widgets exceed 40 this year?", "outcomeType": "BINARY", "probability": 0.31,
     "uniqueBettorCount": 25, "closeTime": 1798761600000, "url": "https://manifold.markets/x/widgets-40"},
    {"question": "Tiny market", "outcomeType": "BINARY", "probability": 0.9,
     "uniqueBettorCount": 3, "closeTime": 1798761600000, "url": "https://manifold.markets/x/tiny"},
    {"question": "Which widget?", "outcomeType": "MULTIPLE_CHOICE", "probability": 0.5,
     "uniqueBettorCount": 50, "closeTime": 1798761600000, "url": "https://manifold.markets/x/mc"},
]


class _Raw:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, amount=None, decode_content=True) -> bytes:
        return self.data[:amount] if amount else self.data


class FakeResponse:
    def __init__(self, status: int, body, content_type: str) -> None:
        self.status_code = status
        self._raw_bytes = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.text = self._raw_bytes.decode()
        self.headers = {"content-type": content_type}
        self.encoding = "utf-8"
        self.raw = _Raw(self._raw_bytes)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


def fake_get(url, params=None, headers=None, timeout=None, stream=False, **kwargs):
    HTTP_GETS.append(url if not params else f"{url}?{sorted(params.items())}")
    if url == "https://www.metaculus.com/api/posts/":
        assert params["statuses"] == "resolved"
        return FakeResponse(200, SIMILAR_POSTS, "application/json")
    if url == "https://api.manifold.markets/v0/search-markets":
        assert params["term"] and params["filter"] == "open"
        return FakeResponse(200, MANIFOLD_MARKETS, "application/json")
    if url == "https://example.com/data":
        return FakeResponse(200, DATA_PAGE, "text/html; charset=utf-8")
    if url == "https://example.com/js-app":
        return FakeResponse(200, "<html><body><div id='root'></div></body></html>", "text/html")
    if url == "https://r.jina.ai/https://example.com/js-app":
        return FakeResponse(200, "Title: Widget app\n\nMarkdown Content:\nThe app reports 41 widgets as of October 4, up from 38 a week earlier.", "text/plain")
    raise requests.ConnectionError(f"offline test: no route to {url}")


def set_keys(**keys: str) -> None:
    for name in [
        "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY", "BOT_FORCE_TIERS",
    ]:
        os.environ.pop(name, None)
    os.environ.update(keys)


def _questions():
    from forecasting_tools import BinaryQuestion, DateQuestion, MultipleChoiceQuestion, NumericQuestion

    now = datetime.now(timezone.utc)

    def common(n: int) -> dict:
        return dict(
            id_of_post=n,
            id_of_question=n,
            page_url=f"https://example.com/q/{n}",
            background_info="Background text.",
            resolution_criteria=(
                "Resolves Yes if the [official tally](https://example.com/data) shows more than 40 "
                "widgets. Backup: https://example.com/js-app."
            ),
            fine_print="Fine print.",
            close_time=now + timedelta(hours=1, minutes=n),
            scheduled_resolution_time=now + timedelta(days=14),
            already_forecasted=False,
        )

    return [
        BinaryQuestion(question_text="Will the widget count exceed 40 in October?", **common(1)),
        MultipleChoiceQuestion(question_text="Which option?", options=["Alpha", "Beta", "Gamma"], **common(2)),
        NumericQuestion(
            question_text="How many widgets?", upper_bound=100, lower_bound=0,
            open_upper_bound=True, open_lower_bound=True, unit_of_measure="widgets", **common(3),
        ),
        DateQuestion(
            question_text="When?", upper_bound=datetime(2027, 6, 1, tzinfo=timezone.utc),
            lower_bound=datetime(2026, 10, 1, tzinfo=timezone.utc),
            open_upper_bound=True, open_lower_bound=True, **common(4),
        ),
    ]


def _bot(tier_name: str):
    import bot_config as cfg
    import main as bot

    return bot.EnsembleBot(
        tier=cfg.tier_by_name(tier_name),
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


# ------------------------------------------------------------------ tests ----
def test_research_tools() -> None:
    import research_tools as rt

    urls = rt.extract_urls(
        ["See [tally](https://example.com/data), https://en.wikipedia.org/wiki/Foo_(bar). "
         "Image https://x.org/a.png and again https://example.com/data."],
        limit=5,
    )
    assert urls == ["https://example.com/data", "https://en.wikipedia.org/wiki/Foo_(bar)"], urls

    text = rt.html_to_text(DATA_PAGE)
    assert "IGNORE ME" not in text and "color:red" not in text and "Home | About" not in text
    assert "41" in text and "Weekly widget count" in text, text

    long_text = "\n".join([f"Filler paragraph number {i} about nothing." for i in range(400)]
                          + ["The widget count reached 41 on 2026-10-04."])
    excerpt = rt.select_relevant_passages(long_text, "Will the widget count exceed 40?", 1_500)
    assert len(excerpt) <= 1_600 and "reached 41" in excerpt, excerpt

    assert rt.top_keywords("Will Apple release the iPhone 18 before October 2026?", 4) == [
        "Apple", "iPhone", "18", "release"]
    assert rt._format_resolution("36.40000001") == "36.4"
    assert rt._format_resolution("2026-10-04T00:00:00Z") == "2026-10-04"

    similar = rt.similar_resolved_questions(1, "Will widgets exceed 40?", "token")
    assert "→ no" in similar and "→ yes" in similar and "→ 36.4" in similar
    assert "Annulled" not in similar and "A group" not in similar

    markets = rt.manifold_markets("Will widgets exceed 40 this year?", binary_only=True)
    assert "31%" in markets and "Tiny market" not in markets and "Which widget" not in markets

    digest = rt.resolution_source_digest(
        ["https://example.com/data", "https://example.com/js-app", "https://down.example"],
        "widget count exceed 40", 3_000,
    )
    assert "### https://example.com/data" in digest and "41" in digest
    assert "The app reports 41 widgets" in digest, "reader fallback for JS pages"
    assert "down.example" not in digest


def test_aggregation_rules() -> None:
    from forecasting_tools import NumericQuestion, Percentile
    from forecasting_tools import NumericDistribution, PredictedOptionList
    from forecasting_tools.data_models.multiple_choice_report import PredictedOption

    import main as bot

    assert bot.aggregate_binary([0.005, 0.01, 0.02]) == 0.03
    assert bot.aggregate_binary([0.995]) == 0.97
    assert abs(bot.aggregate_binary([0.2, 0.3, 0.9]) - 0.3) < 1e-12, "median, no extremizing"

    options = PredictedOptionList(predicted_options=[
        PredictedOption(option_name="A", probability=0.97),
        PredictedOption(option_name="B", probability=0.02),
        PredictedOption(option_name="C", probability=0.01),
    ])
    floored = bot.floor_options(options, bot.multiple_choice_floor(3))
    values = {o.option_name: o.probability for o in floored.predicted_options}
    assert abs(sum(values.values()) - 1) < 1e-9 and values["C"] > 0.019, values
    averaged = bot.average_options([options, floored])
    assert abs(sum(o.probability for o in averaged.predicted_options) - 1) < 1e-9

    question = NumericQuestion(
        question_text="How many?", id_of_post=9, upper_bound=100, lower_bound=0,
        open_upper_bound=True, open_lower_bound=True,
    )

    def dist(center: float) -> NumericDistribution:
        points = [(0.05, center - 6), (0.1, center - 4), (0.5, center), (0.9, center + 4), (0.95, center + 6)]
        return NumericDistribution.from_question(
            [Percentile(percentile=p, value=v) for p, v in points], question)

    low, high = dist(30), dist(70)
    mixed = bot.mixture_of_distributions([low, high], question)

    def height(distribution, value: float) -> float:
        return min(distribution.get_cdf(), key=lambda point: abs(point.value - value)).percentile

    # A mixture keeps both modes: about half the mass sits below 50.
    assert 0.4 < height(mixed, 50) < 0.6, height(mixed, 50)
    assert height(mixed, 40) > 0.4 and height(mixed, 60) < 0.6


def test_budget_planner() -> None:
    import bot_config as cfg
    import budget

    set_keys()
    costs = [budget.estimated_cost_per_question(t) for t in cfg.TIERS]
    assert costs == sorted(costs, reverse=True), f"tiers must get cheaper: {costs}"

    day = date(2026, 10, 7)
    expectations = {
        100: ("lean", "skip"),
        160: ("standard", "skip"),
        250: ("standard", "minimal"),
        300: ("strong", "minimal"),
        5_000: ("max", "standard"),
    }
    for credit, (main_name, minibench_name) in expectations.items():
        plan = budget.choose_plan(credit, day)
        got = (plan.main.name, plan.minibench.name if plan.minibench else "skip")
        assert got == (main_name, minibench_name), (credit, got)
        print(f"    ${credit:>5} -> {plan.describe()}  [{plan.reason}]")

    unknown = budget.choose_plan(None, day)
    assert unknown.main.name == "lean" and unknown.minibench is None
    low = budget.choose_plan(20, day)
    assert low.main.name == "minimal" and low.minibench is None
    assert budget.choose_plan(1.0, day) is None, "below the reserve the run must stop"

    os.environ["BOT_FORCE_TIERS"] = "standard,skip"
    forced = budget.choose_plan(5_000, day)
    assert forced.main.name == "standard" and forced.minibench is None
    os.environ.pop("BOT_FORCE_TIERS")


def test_model_resolution() -> None:
    import bot_config as cfg
    import main as bot

    standard = cfg.tier_by_name("standard")

    set_keys(OPENROUTER_API_KEY="sk-test")
    seats = bot.resolve_forecasters(standard)
    assert [llm.model for _, llm in seats] == [
        "openrouter/openai/gpt-5.4", "openrouter/anthropic/claude-opus-5.5"]
    assert seats[0][1].litellm_kwargs["extra_body"] == {"reasoning": {"effort": "high"}}
    family, fallback = bot.build_fallback_forecaster(["gpt", "opus"])
    assert family == "sonnet" and fallback.litellm_kwargs["extra_body"]["reasoning"]["effort"] == "medium"

    set_keys(ANTHROPIC_API_KEY="sk-ant-test")  # no OpenAI key: GPT seat goes to Opus
    seats = bot.resolve_forecasters(standard)
    assert [family for family, _ in seats] == ["opus", "opus"], seats
    assert seats[0][1].model == "anthropic/claude-opus-5-5"
    assert bot.build_web_search_llms(cfg.WEB_SCOUT_MODEL, 1000) == []

    set_keys()
    try:
        bot.resolve_forecasters(standard)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing keys must fail loudly")


def test_end_to_end_with_fake_models() -> None:
    from forecasting_tools import ForecastReport

    import bot_config as cfg

    set_keys(OPENROUTER_API_KEY="sk-test")
    REQUESTS.clear()
    reports = asyncio.run(_bot("standard").forecast_questions(_questions(), return_exceptions=True))

    failures = [r for r in reports if not isinstance(r, ForecastReport)]
    assert not failures, failures
    binary, multiple_choice, numeric, date_q = reports

    assert abs(binary.prediction - 0.25) < 1e-9, f"median of 30% and 20% expected, got {binary.prediction}"
    probabilities = {o.option_name: o.probability for o in multiple_choice.prediction.predicted_options}
    assert abs(sum(probabilities.values()) - 1) < 1e-6 and probabilities["Alpha"] > probabilities["Gamma"]
    assert numeric.prediction.cdf and date_q.prediction.cdf
    for report in reports:
        assert "Ensemble member: gpt" in report.explanation
        assert "Ensemble member: opus" in report.explanation

    research = binary.explanation
    for title in ["Resolution sources", "Similar questions already resolved", "Prediction markets",
                  "Web research", "Follow-up research"]:
        assert title in research, f"missing research section: {title}"
    assert "IGNORE ME" not in research and "31%" in research and "→ no" in research

    forecast_calls = [r for r in REQUESTS if "plugins" not in r and r.get("reasoning")]
    assert {r["model"] for r in forecast_calls} == {"openai/gpt-5.4", "anthropic/claude-opus-5.5"}
    assert all(r["reasoning"] == {"effort": "high"} for r in forecast_calls)
    assert all("temperature" not in r for r in forecast_calls), "no temperature for reasoning models"
    assert {r["max_tokens"] for r in forecast_calls} == {24_000, 20_000}

    research_calls = [r for r in REQUESTS if "plugins" in r]
    scouts = [r for r in research_calls if r["model"] == "anthropic/claude-sonnet-5.5"]
    gap_fills = [r for r in research_calls if r["model"] == "openai/gpt-5.4"]
    assert len(scouts) == 4 and len(gap_fills) == 4, "two search passes per question"
    assert scouts[0]["plugins"][0]["engine"] == "native" and scouts[0]["reasoning"] == {"effort": "low"}
    assert scouts[0]["plugins"][0]["max_results"] == cfg.WEB_RESEARCH_MAX_RESULTS
    print(f"    {len(REQUESTS)} fake LLM calls, binary forecast {binary.prediction:.0%}")


def test_failures_and_fallback() -> None:
    from forecasting_tools import ForecastReport

    set_keys(OPENROUTER_API_KEY="sk-test")
    binary = _questions()[:1]

    FAILING_MODELS.update({"openai/gpt-5.4"})
    try:  # one member down: the other one carries the forecast
        report = asyncio.run(_bot("standard").forecast_questions(binary, return_exceptions=True))[0]
        assert isinstance(report, ForecastReport), report
        assert abs(report.prediction - 0.20) < 1e-9, report.prediction

        FAILING_MODELS.update({"anthropic/claude-opus-5.5"})  # both down: fallback
        report = asyncio.run(_bot("standard").forecast_questions(binary, return_exceptions=True))[0]
        assert isinstance(report, ForecastReport), report
        assert abs(report.prediction - 0.26) < 1e-9 and "sonnet (fallback)" in report.explanation

        FAILING_MODELS.update({"anthropic/claude-sonnet-5.5"})  # everything down
        report = asyncio.run(_bot("standard").forecast_questions(binary, return_exceptions=True))[0]
        assert isinstance(report, BaseException), report
    finally:
        FAILING_MODELS.clear()


def test_tournament_runner() -> None:
    from forecasting_tools import ForecastReport, MetaculusClient

    import bot_config as cfg
    import budget
    import main as bot

    set_keys(OPENROUTER_API_KEY="sk-test")
    asked: list = []
    question_sets = {
        cfg.MAIN_TOURNAMENT: _questions()[:1],
        cfg.MINIBENCH_TOURNAMENT: [
            q.model_copy(update={"id_of_post": 100 + i, "id_of_question": 100 + i,
                                 "page_url": f"https://example.com/m/{i}"})
            for i, q in enumerate(_questions()[:1] * 5)
        ],
        "bot-testing-area": _questions(),
    }

    def fake_open_questions(self, tournament_id):
        asked.append(tournament_id)
        return [q.model_copy() for q in question_sets[tournament_id]]

    original_fetch = MetaculusClient.get_all_open_questions_from_tournament
    original_credit = budget.remaining_openrouter_credit
    original_box = cfg.MINIBENCH_SECONDS_PER_RUN
    MetaculusClient.get_all_open_questions_from_tournament = fake_open_questions
    try:
        budget.remaining_openrouter_credit = lambda *args, **kwargs: 100.0  # lean, MiniBench off
        reports = asyncio.run(bot.forecast("tournament", publish=False))
        assert asked == [cfg.MAIN_TOURNAMENT], asked
        assert len(reports) == 1 and isinstance(reports[0], ForecastReport), reports

        asked.clear()
        budget.remaining_openrouter_credit = lambda *args, **kwargs: 250.0  # MiniBench on
        reports = asyncio.run(bot.forecast("tournament", publish=False))
        # main first, then MiniBench batches of 3 with a main re-check after each
        assert asked == [cfg.MAIN_TOURNAMENT, cfg.MINIBENCH_TOURNAMENT, cfg.MAIN_TOURNAMENT,
                         cfg.MAIN_TOURNAMENT], asked
        assert len(reports) == 6 and all(isinstance(r, ForecastReport) for r in reports), reports

        asked.clear()
        cfg.MINIBENCH_SECONDS_PER_RUN = 0  # time box already used up
        reports = asyncio.run(bot.forecast("tournament", publish=False))
        assert asked == [cfg.MAIN_TOURNAMENT, cfg.MINIBENCH_TOURNAMENT] and len(reports) == 1
        cfg.MINIBENCH_SECONDS_PER_RUN = original_box

        asked.clear()
        assert asyncio.run(bot.forecast("metaculus_cup", publish=False)) == [] and asked == []

        reports = asyncio.run(bot.forecast("test_questions", publish=False))
        assert asked == ["bot-testing-area"] and len(reports) == 4, (asked, reports)
        assert all(isinstance(r, ForecastReport) for r in reports), reports

        asked.clear()
        budget.remaining_openrouter_credit = lambda *args, **kwargs: 0.5
        assert asyncio.run(bot.forecast("tournament", publish=False)) == [] and asked == []
    finally:
        MetaculusClient.get_all_open_questions_from_tournament = original_fetch
        budget.remaining_openrouter_credit = original_credit
        cfg.MINIBENCH_SECONDS_PER_RUN = original_box


def test_run_deadline() -> None:
    import bot_config as cfg
    import main as bot

    original_deadline, original_inner = cfg.RUN_DEADLINE_SECONDS, bot._forecast

    async def slow(mode, publish, reports, started):
        reports.append("partial result")
        await asyncio.sleep(30)

    cfg.RUN_DEADLINE_SECONDS, bot._forecast = 1, slow
    try:
        assert asyncio.run(bot.forecast("tournament", publish=False)) == ["partial result"]
    finally:
        cfg.RUN_DEADLINE_SECONDS, bot._forecast = original_deadline, original_inner


def test_test_question_sampling() -> None:
    import main as bot

    questions = _questions()
    picked = bot.pick_test_questions(questions + questions, 3)
    assert len(picked) == 3 and len({type(q) for q in picked}) == 3


if __name__ == "__main__":
    start_fake_openrouter()
    requests.get = fake_get  # every HTTP GET in the bot goes through the fake web
    os.environ.setdefault("METACULUS_TOKEN", "offline-test")
    tests = [
        test_research_tools,
        test_aggregation_rules,
        test_budget_planner,
        test_model_resolution,
        test_test_question_sampling,
        test_end_to_end_with_fake_models,
        test_failures_and_fallback,
        test_tournament_runner,
        test_run_deadline,
    ]
    for test in tests:
        print(f"• {test.__name__}")
        test()
        print("  ok")
    print("All offline checks passed.")
