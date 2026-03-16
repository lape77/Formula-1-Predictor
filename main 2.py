"""F1 Predictor v2 — main orchestrator.

Run:
    python main.py --year 2024 --gp "Bahrain" --mode qualifying
    python main.py --year 2024 --gp "Bahrain" --mode race
    python main.py --mock   # use mock data, no FastF1 required
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Project imports ────────────────────────────────────────────────────────────
from config import DATA_DIR
from data.mock_data import CIRCUITS, DRIVERS, MOCK_QUALIFYING_R1, MOCK_RACE_R1, write_mock_data
from data_fetch.weather_api import WeatherFetcher
from engine.elo import ELOEngine, QualifyingResult, RaceResult
from engine.monte_carlo import MonteCarloSimulator
from engine.tyre_model import TyreModel
from engine.undercut_sim import DriverState, PitStopSimulator
from models.driver import Driver, ELOProfile
from models.track import CircuitType, TrackProfile
from reports.generator import ReportData, ReportGenerator, RiskFactors


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_drivers_from_mock() -> List[Driver]:
    """Build :class:`Driver` objects from mock data with seeded ELO ratings."""
    drivers = []
    for d in DRIVERS:
        profile = ELOProfile(
            overall=float(d["elo_overall"]),
            qualifying=float(d["elo_overall"]) * 0.98,
            wet=float(d["elo_overall"]) * 0.95,
            race_start=float(d["elo_overall"]) * 0.97,
        )
        drivers.append(
            Driver(
                driver_id=d["driver_id"],
                full_name=d["full_name"],
                team=d["team"],
                number=d["number"],
                elo=profile,
            )
        )
    return drivers


def build_track_from_mock(circuit_name: str = "Bahrain Grand Prix") -> TrackProfile:
    """Build a :class:`TrackProfile` from mock circuit data."""
    for c in CIRCUITS:
        if circuit_name.lower() in c["name"].lower():
            return TrackProfile(
                name=c["name"],
                country=c["country"],
                circuit_type=CircuitType(c["circuit_type"]),
                lap_length_km=c["lap_length_km"],
                total_laps=c["total_laps"],
                pit_lane_loss=c["pit_lane_loss"],
                latitude=c["latitude"],
                longitude=c["longitude"],
                safety_car_probability=c["safety_car_probability"],
            )
    # Default fallback
    return TrackProfile(name=circuit_name, country="Unknown")


def seed_elo_from_mock(engine: ELOEngine, drivers: List[Driver]) -> None:
    """Seed the ELO engine from driver ELO profiles."""
    from models.driver import ELOProfile as EP

    for d in drivers:
        engine.ratings[d.driver_id] = EP(
            overall=d.elo.overall,
            qualifying=d.elo.qualifying,
            wet=d.elo.wet,
            race_start=d.elo.race_start,
        )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_mock_pipeline(gp_name: str = "Bahrain Grand Prix") -> None:
    """Run the full prediction pipeline on mock data without FastF1.

    Args:
        gp_name: Circuit name to use from mock dataset.
    """
    logger.info("=== F1 Predictor v2 — Mock Pipeline ===")

    # 1. Write mock CSV/JSON files.
    write_mock_data()

    # 2. Load entities.
    drivers = build_drivers_from_mock()
    track = build_track_from_mock(gp_name)
    logger.info("Circuit: %s (%d laps)", track.name, track.total_laps)

    # 3. Initialise engines.
    elo_engine = ELOEngine()
    seed_elo_from_mock(elo_engine, drivers)
    tyre_model = TyreModel()  # default coefficients (no historical stint data)

    # 4. Fetch weather (best effort).
    try:
        weather = WeatherFetcher().fetch_race_weekend_forecast(
            lat=track.latitude,
            lon=track.longitude,
            race_date=date.today(),
        )
        logger.info("Weather condition: %s  (peak rain %.0f%%)", weather.condition, weather.peak_rain_probability() * 100)
    except Exception as exc:
        logger.warning("Weather fetch failed (%s); using dry defaults.", exc)
        from models.weather import WeatherForecast
        weather = WeatherForecast()

    # 5. Monte Carlo qualifying simulation.
    logger.info("Running Monte Carlo qualifying simulation (N=10 000)…")
    mc = MonteCarloSimulator(elo_engine=elo_engine, tyre_model=tyre_model, weather=weather)
    qual_dist = mc.simulate_qualifying(drivers, track)

    # 6. Derive predicted grid from MC output.
    grid: Dict[str, int] = {drv.driver_id: drv.most_likely_position for drv in qual_dist.drivers}

    # 7. Update ELO from mock qualifying results.
    elo_engine.update_after_qualifying([
        QualifyingResult(driver_id=r["driver_id"], position=r["position"])
        for r in MOCK_QUALIFYING_R1
    ])

    # 8. Monte Carlo race simulation.
    logger.info("Running Monte Carlo race simulation (N=10 000)…")
    race_dist = mc.simulate_race(drivers, grid, track)

    # 9. Strategy simulation.
    driver_states = [
        DriverState(
            driver_id=d.driver_id,
            position=grid.get(d.driver_id, 20),
            gap_to_driver_ahead_s=1.5,
            current_lap=1,
            compound="MEDIUM",
            tyre_age=0,
            base_lap_time_s=90.5,
        )
        for d in drivers
    ]
    pit_sim = PitStopSimulator(track=track)
    strategy_results = pit_sim.simulate_full_strategy(
        driver_states, race_laps=track.total_laps, tyre_model=tyre_model,
        sc_probability=track.safety_car_probability,
    )

    # 10. Update ELO from mock race results.
    elo_engine.update_after_race([
        RaceResult(driver_id=r["driver_id"], position=r["position"], track_profile=track)
        for r in MOCK_RACE_R1
    ])

    # 11. Build report.
    from reports.generator import ReportGenerator, ReportData, RiskFactors

    generator = ReportGenerator()
    report = ReportData(
        grand_prix=track.name,
        year=2024,
        qualifying_prediction=ReportGenerator.build_qualifying_rows(qual_dist),
        race_prediction=ReportGenerator.build_race_rows(race_dist),
        strategy_windows=ReportGenerator.build_strategy_section(strategy_results),
        risk_factors=RiskFactors(
            meteo_risk=round(weather.peak_rain_probability() * 10, 1),
            reliability_risk=3.0,
            strategy_risk=round(track.safety_car_probability * 10, 1),
            tyre_risk=4.0,
        ),
        elo_leaderboard=elo_engine.export_ratings().to_dict(orient="records"),
    )
    out_path = generator.generate(report)
    logger.info("Report saved → %s", out_path)
    logger.info("To view the dashboard run:  streamlit run reports/dashboard.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 Predictor v2")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", type=str, default="Bahrain Grand Prix")
    parser.add_argument("--mode", choices=["qualifying", "race", "both"], default="both")
    parser.add_argument("--mock", action="store_true", help="Use mock data (no FastF1)")
    parser.add_argument("--n-sims", type=int, default=10_000, dest="n_sims")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mock:
        run_mock_pipeline(gp_name=args.gp)
    else:
        logger.info("Live FastF1 pipeline for %d %s (mode=%s)", args.year, args.gp, args.mode)
        try:
            from data_fetch.fastf1_loader import FastF1Loader
            session = FastF1Loader.load_session(args.year, args.gp, "Q")
            lap_times = FastF1Loader.get_lap_times(session)
            logger.info("Loaded %d clean laps from FastF1.", len(lap_times))
        except Exception as exc:
            logger.error("FastF1 load failed: %s", exc)
            logger.info("Falling back to mock pipeline.")
            run_mock_pipeline(gp_name=args.gp)
