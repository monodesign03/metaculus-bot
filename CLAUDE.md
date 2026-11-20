# Metaculus forecasting bot

The owner writes in Italian and is not a programmer: answer in short, simple
Italian with step-by-step instructions. Never ask him to paste API keys or
tokens into the chat.

## What this is

An AI forecasting bot competing in the Metaculus FutureEval bot tournaments
for prize money. It runs for free on GitHub Actions and pays for LLM calls
through OpenRouter credit that the owner tops up himself.

- Tournaments now: `fall-futureeval-2026` (main, ~$50k pool, questions until
  mid-December 2026) and `minibench` (~$1k per 2-week round).
- Results so far (17 Nov 2026): rank 93, peer score +7.55, projected ~$160.
- Next goal: the Spring 2027 seasonal tournament (starts January). Update the
  tournament slugs and `SEASON_LAST_QUESTION_DAY` in `bot_config.py` then, and
  apply early to Metaculus for free LLM credits citing the Fall results.
- Market Pulse is skipped for now: it needs forecasts updated continuously.

## Files

- `bot_config.py`: every setting (tournaments, models, tiers, plans, costs).
- `budget.py`: reads the remaining OpenRouter key credit each run and picks
  the strongest plan that lasts to the end of the season. MiniBench stays on
  at the `minimal` tier while credit is at least $20.
- `main.py`: the ensemble bot (research, several models, aggregation:
  binary median capped 3-97%, multiple choice average with 2% floor,
  numeric CDF mixture plus 10% even tail).
- `research_tools.py`: resolution pages, similar resolved Metaculus
  questions, Manifold markets.
- `.github/workflows/run_bot_on_tournament.yaml`: runs every ~10 minutes
  (cron plus a self-restart job using `gh workflow run`).
  `keepalive.yaml` re-enables workflows weekly.
- `tests/test_offline.py`: offline test suite with a fake OpenRouter server.
  Run it with `python tests/test_offline.py` before every change. Some budget
  expectations depend on today's date.
- `.claude/skills/review-bot`: how to review performance on resolved
  questions.

## Real costs

OpenRouter actually charges about half of litellm's estimate in the logs:
roughly $0.50 per question at the `standard` tier and $0.17 at `minimal`.

## Open items

- AskNews: the owner has a single API key (`ank_...`), not OAuth. The
  workflow now passes `ASKNEWS_API_KEY`; the owner must save the key as the
  GitHub secret `ASKNEWS_API_KEY` and remove `ASKNEWS_CLIENT_ID` (with both
  set, forecasting_tools would prefer OAuth). Then check that the log's 🧠
  line says "AskNews".
- End of November: check the leaderboard and analyse the worst questions.
