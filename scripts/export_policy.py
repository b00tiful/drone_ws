#!/usr/bin/env python3
"""Export an AeroStrike policy for ROS 2 inference."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/aerostrike_policy.onnx")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for policy export."""
    args = parse_args()
    raise NotImplementedError(
        f"Policy export is not implemented yet: {args.checkpoint} -> {args.output}"
    )


if __name__ == "__main__":
    main()
