# F1 Predictor v2 🏎️

A modular Python system for Formula 1 race-weekend predictions using multi-dimensional ELO ratings, tyre-degradation models, Monte Carlo simulation, and live FastF1 data.

---

## Features

| Module | Description |
|---|---|
| **ELO Engine** | 5-dimensional ratings: overall, circuit type, wet, qualifying, race-start |
| **Tyre Model** | Power-law degradation fit per compound × circuit × temperature |
| **Monte Carlo** | 10 000-run simulation with weather weighting, DNF, SC events |
| **Undercut/Overcut** | Lap-by-lap pit-stop strategy simulation with confidence scores |
| **Track Evolution** | Q1→Q2→Q3 grip improvement correction |
| **Weather** | Free Open-Meteo API integration (no key required) |
| **Dashboard** | Streamlit multi-tab visual interface |

---

## Setup

```bash
# 1. Clone and enter the directory
cd f1_predictor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install fastf1 pandas numpy scipy matplotlib seaborn \
            pydantic streamlit requests python-dotenv tqdm pytest
```

---

## Quick Start

### Mock data (no FastF1 required)

```bash
python main.py --mock --gp "Bahrain Grand Prix"
```

This will:
1. Generate mock 2024 data (20 drivers, 10 teams, 5 circuits).
2. Run Monte Carlo qualifying and race simulations.
3. Produce a JSON + CSV report in `data/`.

### Live FastF1 data

```bash
python main.py --year 2024 --gp "Bahrain" --mode both
```

FastF1 caches sessions to `cache/` automatically.

### Streamlit dashboard

```bash
streamlit run reports/dashboard.py
```

Open `http://localhost:8501` in your browser.  Select a generated report from the sidebar.

---

## Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- `tests/test_elo.py` — ELO engine correctness, zero-sum property, order-independence.
- `tests/test_tyre_model.py` — Degradation fitting, prediction, cliff detection.
- `tests/test_undercut_sim.py` — Undercut/overcut outcomes, full-strategy generation.

---

## Project Structure

```
f1_predictor/
├── models/           # Pydantic data models (Driver, Car, Track, Weather, …)
├── data_fetch/       # FastF1 loader, Open-Meteo weather fetcher, cache manager
├── engine/           # ELO, TyreModel, MonteCarlo, Undercut, TrackEvolution
├── reports/          # JSON/CSV generator + Streamlit dashboard
├── utils/            # Lap-time normaliser, sector analyser, grid conversion
├── tests/            # pytest test suite
├── data/             # Mock 2024 season data + generated reports
├── cache/            # FastF1 session cache (auto-created)
├── config.py         # Central configuration
├── main.py           # CLI orchestrator
└── .env              # Environment variable overrides (no keys needed)
```

---

## Report Sections

1. **Qualifying Prediction** — predicted grid, theoretical bests, confidence %.
2. **Race Prediction** — expected finish, win/podium/DNF probabilities.
3. **Strategy Windows** — recommended pit laps, undercut opportunities.
4. **Risk Factors** — meteo / reliability / strategy / tyre (0–10 scores).
5. **ELO Leaderboard** — all five rating dimensions exported to CSV.
6. **Championship Projection** — available after 3+ rounds (WDC/WCC estimates).

---

## Key Bug Fixes vs Original Code

| File | Bug | Fix |
|---|---|---|
| `engine/elo.py` | Duplicate in-place and snapshot mutations applied simultaneously | Removed in-place `+=`; use only snapshot-based update |
| `engine/elo.py` | `expected` score doubled (called `get_expected_score` + `_expected_score_from_ratings` in same loop) | Use only snapshot version throughout |
| `engine/elo.py` | `delta` variable from inner loop leaked into outer apply loop | Computed and stored in `deltas` dict; read back by driver ID |
| `models/car.py` | `ReliabilityProfile()` shared mutable default | `Field(default_factory=ReliabilityProfile)` |
| `models/driver.py` | `ELOProfile()` shared mutable default | `Field(default_factory=ELOProfile)` |
| `data_fetch/fastf1_loader.py` | Duplicate `return` statements | Kept `reindex` version with safe fallback |
