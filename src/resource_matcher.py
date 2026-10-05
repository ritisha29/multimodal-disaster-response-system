"""
Relief Resource Matcher Module.

Matches extracted location strings against relief_resources.csv mock dataset.
Returns matching relief facilities or fallback general resources.
"""

import os
import csv
from typing import List, Dict, Any


# CONFIGURATION FLAG: Set to True to enable live API integrations if available.
# NOTE: Live API mode requires an active Google Cloud Billing Account.
# When False, relief resource lookups operate cleanly via local CSV datasets.
USE_LIVE_API = False


def _resolve_csv_path(csv_path: str) -> str:
    """Resolves absolute path to relief_resources.csv."""
    if os.path.isabs(csv_path) and os.path.exists(csv_path):
        return csv_path

    # Try relative to current script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    candidates = [
        csv_path,
        os.path.join(project_root, csv_path),
        os.path.join(project_root, "data", "relief_resources.csv"),
        os.path.join(script_dir, "..", "data", "relief_resources.csv"),
    ]

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(f"Relief resources CSV not found at '{csv_path}'.")


def match_resources(location_str: str, csv_path: str = "data/relief_resources.csv", top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Matches extracted location mentions with relief resources in relief_resources.csv.

    Args:
        location_str (str): Extracted location string (e.g. "Napa County, California" or "Location not specified").
        csv_path (str): Path to relief_resources.csv.
        top_k (int): Maximum number of resources to return.

    Returns:
        List[Dict[str, Any]]: List of matching relief resource dicts.
    """
    resolved_path = _resolve_csv_path(csv_path)
    resources = []
    general_fallback = []

    with open(resolved_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(line for line in f if not line.startswith('#'))
        for row in reader:
            resources.append(row)
            if row.get('region', '').lower() == 'general' or row.get('city', '').lower() == 'general':
                general_fallback.append(row)

    loc_lower = location_str.strip().lower()

    # If location is unspecified or empty, return fallback general resources immediately
    if not loc_lower or "location not specified" in loc_lower or loc_lower == "none":
        return general_fallback[:top_k]

    matched = []
    for res in resources:
        city = res.get('city', '').strip().lower()
        region = res.get('region', '').strip().lower()

        # Ignore general entries during location matching phase
        if city == 'general' or region == 'general':
            continue

        # Check if city or region appears in extracted location_str or vice versa
        if (city and (city in loc_lower or loc_lower in city)) or \
           (region and (region in loc_lower or loc_lower in region)):
            matched.append(res)

    if matched:
        return matched[:top_k]

    # Fallback if no specific city/region matched
    return general_fallback[:top_k]


if __name__ == '__main__':
    # Simple test execution
    print("Testing match_resources for 'Napa County, California':")
    for r in match_resources("Napa County, California"):
        print(" -", r['name'], f"({r['resource_type']})", "| City:", r['city'], "| Contact:", r['contact_info'])

    print("\nTesting match_resources for 'Location not specified':")
    for r in match_resources("Location not specified"):
        print(" -", r['name'], f"({r['resource_type']})", "| City:", r['city'], "| Contact:", r['contact_info'])
