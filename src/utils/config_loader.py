"""Shared config loading and progress reporting for pipeline scripts.

Usage in any pipeline script:
    from config_loader import load_config, progress

    cfg = load_config()  # returns {} if --config not provided
    name = cfg.get("point_cloud_name", "default_name")

    progress(1, 10, "Loading point cloud")
"""

import argparse
import json
import sys


def load_config(extra_args=None):
    """Parse --config <path> from sys.argv and return the JSON dict.

    If --config is not provided, returns an empty dict so callers can
    fall back to hardcoded defaults via cfg.get(key, default).

    Parameters
    ----------
    extra_args : list[str], optional
        Additional argparse arguments to parse (e.g. positional args
        that the script already expects). Not used by default.

    Returns
    -------
    dict
        Parsed config values, or {} if no --config flag was given.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    known, _remaining = parser.parse_known_args()

    if known.config is None:
        return {}

    with open(known.config, "r", encoding="utf-8") as f:
        return json.load(f)


def progress(current, total, message=""):
    """Print a structured progress line for the Electron app to parse.

    Format: [PROGRESS] <current> <total> <message>

    Parameters
    ----------
    current : int
        Current step number (1-based).
    total : int
        Total number of steps.
    message : str
        Human-readable description of the current step.
    """
    print(f"[PROGRESS] {current} {total} {message}", flush=True)
