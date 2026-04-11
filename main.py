"""Startet Label Studio mit der Labelkonfiguration für dieses Repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LABEL_CONFIG = PROJECT_ROOT / "label_config.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Studio für machinevisionessentials starten.")
    parser.add_argument(
        "-l",
        "--label-config",
        type=Path,
        default=DEFAULT_LABEL_CONFIG,
        help="Pfad zur Labeling-XML (Standard: label_config.xml).",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port des Webservers (Standard: 8080).",
    )
    parser.add_argument(
        "-b",
        "--no-browser",
        action="store_true",
        help="Browser beim Start nicht öffnen.",
    )
    parser.add_argument(
        "--username",
        default="dev",
        help="Erster Nutzer bei frischer Installation (Standard: dev).",
    )
    parser.add_argument(
        "--password",
        default="devpass",
        help="Passwort für --username (Standard: devpass; nur für lokale Entwicklung).",
    )
    parser.add_argument(
        "project_name",
        nargs="?",
        default="machine_vision",
        help='Name des Label-Studio-Projekts (Standard: "machine_vision").',
    )
    args = parser.parse_args()

    label_config = args.label_config.resolve()
    if not label_config.is_file():
        print(f"Labelkonfiguration fehlt: {label_config}", file=sys.stderr)
        sys.exit(1)

    exe = shutil.which("label-studio")
    if exe is None:
        print(
            "label-studio wurde nicht gefunden. Im Projektordner z. B.: uv sync && uv run python main.py",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd: list[str] = [
        exe,
        "start",
        args.project_name,
        "-l",
        str(label_config),
        "-p",
        str(args.port),
        "--init",
        "--username",
        args.username,
        "--password",
        args.password,
    ]
    if args.no_browser:
        cmd.append("-b")

    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
