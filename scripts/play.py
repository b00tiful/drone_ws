#!/usr/bin/env python3
"""Evaluate a trained AeroStrike policy."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    """Entrypoint for policy playback."""
    args = parse_args()
    raise NotImplementedError(f"Policy playback is not implemented yet: {args.checkpoint}")


if __name__ == "__main__":
    main()
