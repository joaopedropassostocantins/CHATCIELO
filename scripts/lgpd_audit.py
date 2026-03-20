"""
LGPD PII Audit Script.

Scans log files and output directories for any PII patterns.
Exits with code 1 if PII is found (used in CI pipeline).

Usage:
    python scripts/lgpd_audit.py --target logs/ --target outputs/
    python scripts/lgpd_audit.py --target logs/ --strict
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

# Allow import from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import contains_pii


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Scan a single file for PII patterns.

    Args:
        path: File path to scan.

    Returns:
        List of (line_number, line_content) tuples where PII was detected.
    """
    violations = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                if contains_pii(line):
                    violations.append((lineno, line.rstrip()))
    except (OSError, PermissionError) as e:
        click.echo(f"  [SKIP] Cannot read {path}: {e}", err=True)
    return violations


@click.command()
@click.option(
    "--target",
    "targets",
    multiple=True,
    required=True,
    help="Directory or file to scan. Can be specified multiple times.",
)
@click.option(
    "--extensions",
    default=".log,.json,.txt,.csv",
    help="Comma-separated file extensions to scan.",
)
@click.option("--strict", is_flag=True, help="Exit 1 immediately on first PII found.")
@click.option("--report", default=None, help="Write violations to this JSON file.")
def main(
    targets: tuple[str, ...],
    extensions: str,
    strict: bool,
    report: str | None,
) -> None:
    """Scan files for LGPD-regulated PII patterns.

    Detects: CPF, CNPJ, credit card numbers, Brazilian phone numbers, emails.
    """
    allowed_exts = set(extensions.split(","))
    total_files = 0
    total_violations = 0
    all_violations: dict[str, list[dict]] = {}

    for target_str in targets:
        target = Path(target_str)
        if not target.exists():
            click.echo(f"Warning: Target does not exist: {target}", err=True)
            continue

        paths = [target] if target.is_file() else list(target.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix not in allowed_exts:
                continue

            total_files += 1
            violations = _scan_file(path)

            if violations:
                total_violations += len(violations)
                all_violations[str(path)] = [
                    {"line": lineno, "content": content[:200]}
                    for lineno, content in violations
                ]
                for lineno, content in violations:
                    click.echo(
                        click.style(f"  [PII FOUND] {path}:{lineno}: {content[:100]}", fg="red")
                    )
                if strict:
                    click.echo(click.style("STRICT mode: aborting on first violation.", fg="red"))
                    sys.exit(1)

    click.echo(f"\nAudit complete: {total_files} files scanned, {total_violations} PII violations found.")

    if report:
        import json
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w") as f:
            json.dump({
                "files_scanned": total_files,
                "total_violations": total_violations,
                "violations": all_violations,
            }, f, indent=2)
        click.echo(f"Report written to: {report}")

    if total_violations > 0:
        click.echo(click.style(f"\nFAIL: {total_violations} PII violations must be remediated.", fg="red"))
        sys.exit(1)
    else:
        click.echo(click.style("PASS: No PII detected.", fg="green"))


if __name__ == "__main__":
    main()
