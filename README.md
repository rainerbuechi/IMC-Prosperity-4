# IMC Prosperity 4 — Solo Learning Journal & Strategy Research 

This repository documents my work during IMC Prosperity 4, my first quant / algorithmic trading competition. I participated solo, so the focus of this write-up is not only final performance, but also the research process: how I analyzed market data, formed hypotheses, tested strategies, handled execution problems, and approached the manual challenges.

## Repository Structure

- `docsg/` — written explanations, round summaries, manual challenge analysis
- `notebooks/` — exploratory data analysis and parameter research
- `src/` — reusable trading and analysis utilities
- `submissions/` — submitted trader files by round
- `data/` — local data folder, not fully tracked in GitHub if large

## Main Themes

- Fair value estimation from order book structure
- Passive market making and inventory control
- Execution quality and adverse selection
- Counterparty analysis once trader IDs became visible
- Option pricing, implied volatility, and practical failure modes
- Game-theoretic manual challenge reasoning

## Round Write-ups

| Round | Main Focus | Notes |
|---|---|---|
| Tutorial | Basic market making | First implementation and environment setup |
| Round 1 | Wide-spread market making | Fair value proxies, passive quoting, inventory limits |
| Round 2 | Market access + refinement | Execution quality, bid sizing, manual investment allocation |
| Round 3 | Options + delta-one products | Black-Scholes intuition, voucher mispricing, reserve-price manual challenge |
| Round 4 | Counterparty IDs + exotics | Trader behavior analysis, option oscillation failure, exotic option manual challenge |
| Round 5 | - | Didn't compete |

## Disclaimer

This was my first quant competition and I played solo. The repo is intentionally written as a learning journal and postmortem rather than a top-placement solution guide. The last round was voided due to Time issues.
