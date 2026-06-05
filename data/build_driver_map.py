"""
KRONECTOR — Driver ID Mapping Builder
Generates drivers_map.json mapping FastF1 abbreviations → Jolpica slugs.

Run this ONCE at project init before any pipeline runs:
    python -m data.build_driver_map

Rule: FastF1 3-letter abbreviation is the MASTER driver_id throughout
the entire system. Jolpica slugs are only used for Jolpica API calls.

Output: drivers_map.json at project root
Format: {"VER": "max_verstappen", "HAM": "lewis_hamilton", ...}
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Path to output mapping file
DRIVER_MAP_PATH = Path(__file__).parent / "drivers_map.json"

JOLPICA_BASE_URL = os.getenv(
    "JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1"
)


# ---------------------------------------------------------------------------
# Jolpica request helper (reuse from jolpica_pipeline)
# ---------------------------------------------------------------------------

def _jolpica_get(
    url: str, retries: int = 3, base_delay: float = 0.2
) -> Optional[dict[str, Any]]:
    """Rate-limited Jolpica GET with exponential backoff."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(base_delay)
            return response.json()
        except requests.exceptions.RequestException as e:
            wait = base_delay * (2 ** attempt)
            logger.warning(
                f"Request failed (attempt {attempt + 1}): {e}. "
                f"Retrying in {wait:.1f}s"
            )
            time.sleep(wait)
    logger.error(f"Failed after {retries} attempts: {url}")
    return None


# ---------------------------------------------------------------------------
# Fetch all Jolpica drivers
# ---------------------------------------------------------------------------

def _fetch_jolpica_drivers() -> dict[str, str]:
    """
    Fetch all F1 drivers from Jolpica API.

    Returns dict: {full_name_lower: jolpica_slug}
    Example: {"max verstappen": "max_verstappen"}
    """
    # Fetch with high limit to get all drivers
    url = f"{JOLPICA_BASE_URL}/drivers.json?limit=1000"
    data = _jolpica_get(url)

    if data is None:
        return {}

    try:
        drivers = data["MRData"]["DriverTable"]["Drivers"]
        result = {}
        for driver in drivers:
            full_name = (
                f"{driver['givenName']} {driver['familyName']}"
            ).lower()
            slug = driver["driverId"]
            result[full_name] = slug

        logger.info(f"Fetched {len(result)} drivers from Jolpica")
        return result

    except (KeyError, TypeError) as e:
        logger.error(f"Failed to parse Jolpica drivers: {e}")
        return {}


# ---------------------------------------------------------------------------
# Fetch FastF1 driver abbreviations
# ---------------------------------------------------------------------------

def _fetch_fastf1_drivers(
    start_year: int = 2014, end_year: int = 2026
) -> dict[str, str]:
    """
    Fetch driver abbreviations from FastF1 across multiple seasons.

    Returns dict: {full_name_lower: abbreviation}
    Example: {"max verstappen": "VER"}
    """
    try:
        import fastf1
    except ImportError:
        logger.error("fastf1 not installed — cannot fetch driver abbreviations")
        return {}

    # Enable cache if possible
    cache_dir = os.getenv("FASTF1_CACHE_DIR", "./cache/fastf1")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        fastf1.Cache.enable_cache(cache_dir)
    except Exception:
        pass

    result = {}

    for year in range(start_year, end_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            # Pick first race of the season to get driver list
            first_round = schedule[schedule["RoundNumber"] > 0].iloc[0]
            round_num = first_round["RoundNumber"]

            session = fastf1.get_session(year, round_num, "R")
            session.load()

            if session.results is not None and not session.results.empty:
                for _, driver in session.results.iterrows():
                    abbrev = driver.get("Abbreviation", "")
                    full_name = str(driver.get("FullName", "")).lower()
                    if abbrev and full_name:
                        result[full_name] = abbrev

            logger.info(
                f"Fetched FastF1 drivers for {year}: "
                f"{len(session.results)} drivers"
            )

        except Exception as e:
            logger.warning(f"Could not fetch FastF1 drivers for {year}: {e}")
            continue

    logger.info(f"Total unique FastF1 drivers: {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Build the mapping
# ---------------------------------------------------------------------------

def build_driver_map(
    start_year: int = 2014, end_year: int = 2026
) -> dict[str, str]:
    """
    Build the master driver mapping: FastF1 abbreviation → Jolpica slug.

    Strategy:
      1. Fetch all Jolpica drivers → {full_name: slug}
      2. Fetch all FastF1 drivers → {full_name: abbreviation}
      3. Match on full_name → {abbreviation: slug}

    Returns:
        Dict mapping FastF1 abbreviation to Jolpica slug
        Example: {"VER": "max_verstappen", "HAM": "lewis_hamilton"}
    """
    logger.info("Building driver map...")

    # Fetch from both sources
    jolpica_drivers = _fetch_jolpica_drivers()
    fastf1_drivers = _fetch_fastf1_drivers(start_year, end_year)

    # Match on full name
    driver_map = {}
    unmatched_fastf1 = []

    for full_name, abbreviation in fastf1_drivers.items():
        if full_name in jolpica_drivers:
            driver_map[abbreviation] = jolpica_drivers[full_name]
        else:
            # Try fuzzy match: remove accents, extra spaces
            matched = False
            for jolpica_name, slug in jolpica_drivers.items():
                # Simple contains check for partial name matches
                name_parts = full_name.split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    if last_name in jolpica_name:
                        driver_map[abbreviation] = slug
                        matched = True
                        break
            if not matched:
                unmatched_fastf1.append((abbreviation, full_name))

    if unmatched_fastf1:
        logger.warning(
            f"Unmatched FastF1 drivers ({len(unmatched_fastf1)}): "
            f"{unmatched_fastf1}"
        )

    logger.info(
        f"Driver map built: {len(driver_map)} mapped, "
        f"{len(unmatched_fastf1)} unmatched"
    )
    return driver_map


# ---------------------------------------------------------------------------
# Known fallback map (covers common edge cases)
# ---------------------------------------------------------------------------

KNOWN_DRIVER_MAP = {
    # 2014–2026 comprehensive fallback for edge cases
    "VER": "max_verstappen",
    "HAM": "lewis_hamilton",
    "LEC": "charles_leclerc",
    "NOR": "lando_norris",
    "SAI": "carlos_sainz",
    "PER": "perez",
    "RUS": "george_russell",
    "PIA": "oscar_piastri",
    "ALO": "alonso",
    "STR": "stroll",
    "GAS": "gasly",
    "OCO": "ocon",
    "ALB": "albon",
    "TSU": "tsunoda",
    "BOT": "bottas",
    "ZHO": "zhou",
    "MAG": "kevin_magnussen",
    "HUL": "hulkenberg",
    "RIC": "ricciardo",
    "LAW": "lawson",
    "SAR": "sargeant",
    "DEV": "de_vries",
    "VET": "vettel",
    "RAI": "raikkonen",
    "GRO": "grosjean",
    "KVY": "kvyat",
    "MAL": "maldonado",
    "MAS": "massa",
    "BUT": "button",
    "ROS": "rosberg",
    "VAN": "vandoorne",
    "WEH": "wehrlein",
    "ERI": "ericsson",
    "NAK": "nakajima",
    "PAL": "palmer",
    "HAR": "hartley",
    "SIR": "sirotkin",
    "GIO": "giovinazzi",
    "KUB": "kubica",
    "LAT": "latifi",
    "AIT": "aitken",
    "FIT": "pietro_fittipaldi",
    "MSC": "mick_schumacher",
    "MAZ": "mazepin",
    "BEA": "bearman",
    "COL": "colapinto",
    "DOO": "doohan",
    "ANT": "antonelli",
    "HAD": "hadjar",
    "BOR": "bortoleto",
}


def save_driver_map(driver_map: dict[str, str]) -> None:
    """Save driver map to JSON file at project root."""
    # Merge with known fallback (API results take priority)
    merged = {**KNOWN_DRIVER_MAP, **driver_map}

    with open(DRIVER_MAP_PATH, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)

    logger.info(
        f"Saved driver map to {DRIVER_MAP_PATH} ({len(merged)} entries)"
    )


def load_driver_map() -> dict[str, str]:
    """Load driver map from JSON file. Falls back to known map if missing."""
    if DRIVER_MAP_PATH.exists():
        with open(DRIVER_MAP_PATH, "r") as f:
            return json.load(f)
    else:
        logger.warning(
            f"Driver map not found at {DRIVER_MAP_PATH} — "
            f"using KNOWN_DRIVER_MAP fallback"
        )
        return KNOWN_DRIVER_MAP


# Module-level constant: always available for import
DRIVER_MAP = load_driver_map()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KRONECTOR — Build Driver ID Mapping"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2014,
        help="Start year (default: 2014)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2026,
        help="End year (default: 2026)",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip API calls, save KNOWN_DRIVER_MAP only",
    )
    args = parser.parse_args()

    if args.fallback_only:
        save_driver_map({})
        print(f"Saved fallback map: {len(KNOWN_DRIVER_MAP)} entries")
    else:
        driver_map = build_driver_map(args.start, args.end)
        save_driver_map(driver_map)
        print(f"\nDriver map saved to: {DRIVER_MAP_PATH}")
        print(f"Total entries: {len({**KNOWN_DRIVER_MAP, **driver_map})}")

        # Print sample
        final = {**KNOWN_DRIVER_MAP, **driver_map}
        for abbrev in ["VER", "HAM", "LEC", "NOR", "SAI"]:
            print(f"  {abbrev} → {final.get(abbrev, '???')}")
