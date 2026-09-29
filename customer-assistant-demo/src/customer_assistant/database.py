"""Step 1 - Validate synthetic source CSVs and build the local SQLite copy.

- Read six source files: customers, instruments, orders, service history,
  troubleshooting articles, and evaluation cases. The CSVs remain the source
  of truth; this module never rewrites them.
- Check headers, row widths, unique IDs, supported order statuses, and links
  between customers, instruments, orders, service events, and evaluation
  logins. In particular, an order/event and its instrument must name the same
  customer, or later ownership checks could give a misleading answer.
- Only after all checks pass, replace the six source-backed SQLite tables and
  their indexes in one transaction. A failed load leaves the last good tables
  intact. Audit traces live in a separate table and survive source refreshes.
- Write a per-file JSON validation report on success or failure. Provide a
  read-only connection helper so identity, policy, MCP, and evaluation code
  cannot silently create a missing demo database.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Resolve paths from this module so the loader also works when run outside the
# project directory, such as from a notebook kernel or an IDE test runner.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "source"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "local" / "customer_assistant.sqlite3"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "local" / "validation_report.json"

# These contracts describe the six *input* files, not fields invented by the
# assistant. We permit reordered columns because CSV readers use names, but
# reject missing, duplicate, or extra names rather than silently dropping data.
HEADERS: dict[str, tuple[str, ...]] = {
    "customers.csv": (
        "customer_id", "customer_name", "region", "demo_login", "contact_email",
        "data_classification",
    ),
    "instruments.csv": (
        "instrument_id", "customer_id", "model", "serial_number", "installed_on",
        "warranty_expires_on", "instrument_status", "record_source",
    ),
    "orders.csv": (
        "order_id", "customer_id", "instrument_id", "created_on",
        "item_description", "demo_sku", "order_status", "shipped_on",
        "estimated_delivery", "tracking_reference", "record_source",
    ),
    "service_history.csv": (
        "service_event_id", "customer_id", "instrument_id", "event_date",
        "event_type", "reported_symptom", "technician_summary",
        "resolution_status", "access_tier", "record_source",
    ),
    "troubleshooting_articles.csv": (
        "article_id", "model_family", "symptom", "article_summary",
        "approved_assistant_response", "escalation_rule", "source_title",
        "source_locator", "source_url", "access_tier", "content_status",
    ),
    "evaluation_cases.csv": (
        "case_id", "demo_login", "question", "expected_skill",
        "expected_access", "expected_source_id", "expected_fact",
        "intentional_negative_case", "expected_gate_result",
        "expected_promotion", "candidate_override_text",
    ),
}
PRIMARY_KEYS = {
    "customers.csv": "customer_id",
    "instruments.csv": "instrument_id",
    "orders.csv": "order_id",
    "service_history.csv": "service_event_id",
    "troubleshooting_articles.csv": "article_id",
    "evaluation_cases.csv": "case_id",
}
# Order status is used in grounded customer answers and in the evaluation gate.
# Reject an unfamiliar value here so downstream code cannot give it an
# unsupported meaning; expanding the vocabulary needs an explicit review.
ORDER_STATUSES = frozenset({"processing", "in_transit", "delivered"})


@dataclass(frozen=True)
class CheckFailure:
    """A named failed check that callers can show without parsing exceptions."""

    check: str
    detail: str


@dataclass
class FileReport:
    """Row count and failures for one source file, including invalid files."""

    file: str
    rows: int = 0
    failed_checks: list[CheckFailure] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "failed" if self.failed_checks else "passed"

    def fail(self, check: str, detail: str) -> None:
        self.failed_checks.append(CheckFailure(check, detail))

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "rows": self.rows,
            "status": self.status,
            "failed_checks": [
                {"check": failure.check, "detail": failure.detail}
                for failure in self.failed_checks
            ],
        }


@dataclass
class ValidationReport:
    """The inspectable result of validation or a completed load."""

    files: list[FileReport]
    database: str | None = None

    @property
    def passed(self) -> bool:
        return all(file.status == "passed" for file in self.files)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": "Step 1",
            "summary": [
                "- Validate all six synthetic CSV files before loading.",
                "- Store an indexed local SQLite copy for later access checks.",
                "- Report row counts and failed checks for each file.",
            ],
            "status": "passed" if self.passed else "failed",
            "database": self.database,
            "files": [file.as_dict() for file in self.files],
        }


class ValidationError(ValueError):
    """The source CSVs failed validation; the SQLite copy was not changed."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("CSV validation failed; see the validation report")


def _read_and_validate(
    source_dir: Path,
) -> tuple[dict[str, list[dict[str, str]]], ValidationReport]:
    """Collect every file result before deciding whether a database is safe to load."""

    records: dict[str, list[dict[str, str]]] = {}
    # Store source line numbers separately from parsed records. A malformed
    # line still counts in the report but cannot become a row; later duplicate
    # ID errors should point to the original CSV line, not a shifted index.
    record_numbers: dict[str, list[int]] = {}
    file_reports = {name: FileReport(name) for name in HEADERS}
    report = ValidationReport(list(file_reports.values()))

    for name, expected in HEADERS.items():
        file_report = file_reports[name]
        rows: list[dict[str, str]] = []
        records[name] = rows
        record_numbers[name] = []
        try:
            # Spreadsheet exports may include a UTF-8 byte-order mark, and
            # quoted CSV cells may contain line breaks. These reader options
            # handle both without rewriting the user-supplied files.
            with (source_dir / name).open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, None)
                if header is None:
                    file_report.fail("headers", "file has no header row")
                    continue

                duplicate_headers = sorted({col for col in header if header.count(col) > 1})
                missing = sorted(set(expected) - set(header))
                unexpected = sorted(set(header) - set(expected))
                if duplicate_headers:
                    file_report.fail("headers", f"duplicate columns: {duplicate_headers}")
                if missing:
                    file_report.fail("headers", f"missing required columns: {missing}")
                if unexpected:
                    file_report.fail("headers", f"unexpected columns: {unexpected}")
                headers_valid = not (duplicate_headers or missing or unexpected)

                for row_number, values in enumerate(reader, start=2):
                    file_report.rows += 1
                    if len(values) != len(header):
                        file_report.fail(
                            "row_width",
                            f"row {row_number}: expected {len(header)} values, got {len(values)}",
                        )
                    elif headers_valid:
                        rows.append(dict(zip(header, values)))
                        record_numbers[name].append(row_number)
        except (OSError, UnicodeError, csv.Error) as exc:
            file_report.fail("file_read", f"{type(exc).__name__}: {exc}")

    for name, rows in records.items():
        file_report = file_reports[name]
        primary_key = PRIMARY_KEYS[name]
        seen: set[str] = set()
        # IDs are security-relevant join keys. Compare their exact CSV strings:
        # trimming whitespace or coercing numbers could silently relabel a
        # row and make it appear to belong to another customer.
        for row_number, row in zip(record_numbers[name], rows, strict=True):
            value = row[primary_key]
            if not value or value.isspace():
                file_report.fail("primary_id", f"row {row_number}: blank {primary_key}")
            elif value in seen:
                file_report.fail("primary_id", f"row {row_number}: duplicate {primary_key} {value!r}")
            seen.add(value)

    customers = {row["customer_id"] for row in records["customers.csv"]}
    logins: set[str] = set()
    for row in records["customers.csv"]:
        login = row["demo_login"]
        if not login or login.isspace():
            file_reports["customers.csv"].fail("demo_login", "blank demo_login")
        elif login in logins:
            file_reports["customers.csv"].fail("demo_login", f"duplicate demo_login {login!r}")
        logins.add(login)

    # Separate foreign keys show that the customer and instrument each exist;
    # they do not establish that the instrument belongs to that customer.
    # Require the order/service row and instrument to name the *same* owner,
    # otherwise later authorization queries could give a misleading result.
    instruments = {
        row["instrument_id"]: row["customer_id"]
        for row in records["instruments.csv"]
    }
    for row in records["instruments.csv"]:
        if row["customer_id"] not in customers:
            file_reports["instruments.csv"].fail(
                "customer_foreign_key",
                f"{row['instrument_id']}: unknown customer_id {row['customer_id']!r}",
            )

    for name in ("orders.csv", "service_history.csv"):
        file_report = file_reports[name]
        for row in records[name]:
            record_id = row[PRIMARY_KEYS[name]]
            customer_id = row["customer_id"]
            instrument_id = row["instrument_id"]
            if customer_id not in customers:
                file_report.fail(
                    "customer_foreign_key",
                    f"{record_id}: unknown customer_id {customer_id!r}",
                )
            if instrument_id not in instruments:
                file_report.fail(
                    "instrument_foreign_key",
                    f"{record_id}: unknown instrument_id {instrument_id!r}",
                )
            elif instruments[instrument_id] != customer_id:
                file_report.fail(
                    "instrument_customer_match",
                    f"{record_id}: {instrument_id} belongs to {instruments[instrument_id]!r}, "
                    f"not {customer_id!r}",
                )

    for row in records["orders.csv"]:
        if row["order_status"] not in ORDER_STATUSES:
            file_reports["orders.csv"].fail(
                "order_status",
                f"{row['order_id']}: unsupported status {row['order_status']!r}; "
                f"allowed: {', '.join(sorted(ORDER_STATUSES))}",
            )

    for row in records["evaluation_cases.csv"]:
        if row["demo_login"] not in logins:
            file_reports["evaluation_cases.csv"].fail(
                "demo_login_foreign_key",
                f"{row['case_id']}: unknown demo_login {row['demo_login']!r}",
            )

    # DEMO-ORD-1009 / DEMO-SVC-1009 deliberately conflict for the demo.
    # Live data products would reconcile timelines; PK/FKs cannot do that.
    return records, report


def validate_source(source_dir: Path = DEFAULT_SOURCE_DIR) -> ValidationReport:
    """Return a per-file report without writing a database."""

    _, report = _read_and_validate(Path(source_dir))
    return report


def connect_readonly(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the existing copy for reads; never create it as a side effect."""

    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Demo database not found: {path}")
    # SQLite normally creates a missing database on connect. Explicit read-only
    # URI mode prevents that confusing failure and keeps identity/policy reads
    # from accidentally modifying the demo copy.
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    # Preserve the exact validated CSV values by using TEXT even for dates and
    # tier labels. Numeric coercion could alter IDs or change a tier meaning.
    # Cross-table ownership is checked before loading because these simple
    # foreign keys alone do not enforce instrument/customer equality.
    foreign_keys = {
        "instruments.csv": [
            'FOREIGN KEY ("customer_id") REFERENCES "customers" ("customer_id")',
        ],
        "orders.csv": [
            'FOREIGN KEY ("customer_id") REFERENCES "customers" ("customer_id")',
            'FOREIGN KEY ("instrument_id") REFERENCES "instruments" ("instrument_id")',
        ],
        "service_history.csv": [
            'FOREIGN KEY ("customer_id") REFERENCES "customers" ("customer_id")',
            'FOREIGN KEY ("instrument_id") REFERENCES "instruments" ("instrument_id")',
        ],
        "evaluation_cases.csv": [
            'FOREIGN KEY ("demo_login") REFERENCES "customers" ("demo_login")',
        ],
    }
    for name, headers in HEADERS.items():
        table = name.removesuffix(".csv")
        columns = []
        for col in headers:
            definition = f'"{col}" TEXT'
            if col == PRIMARY_KEYS[name]:
                definition += " PRIMARY KEY NOT NULL"
            elif name == "customers.csv" and col == "demo_login":
                definition += " NOT NULL UNIQUE"
            elif col in ("customer_id", "instrument_id") and name != "customers.csv":
                definition += " NOT NULL"
            elif name == "evaluation_cases.csv" and col == "demo_login":
                definition += " NOT NULL"
            if name == "orders.csv" and col == "order_status":
                allowed = ", ".join(f"'{status}'" for status in sorted(ORDER_STATUSES))
                definition += f" NOT NULL CHECK (\"order_status\" IN ({allowed}))"
            columns.append(definition)
        columns.extend(foreign_keys.get(name, []))
        connection.execute(f'CREATE TABLE "{table}" ({", ".join(columns)})')

    # The Step 2 policy filters by both customer and instrument IDs *inside*
    # SQL. Index those keys so entitlement checks remain practical if more
    # synthetic records are added to the demo.
    for table, col in (
        ("instruments", "customer_id"),
        ("orders", "customer_id"),
        ("orders", "instrument_id"),
        ("service_history", "customer_id"),
        ("service_history", "instrument_id"),
    ):
        connection.execute(f'CREATE INDEX "idx_{table}_{col}" ON "{table}" ("{col}")')

    # A SQLite file cannot start with a human-readable Step header. This
    # metadata gives a reviewer the same provenance inside the binary file.
    connection.execute(
        "CREATE TABLE IF NOT EXISTS artifact_metadata "
        "(key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT OR REPLACE INTO artifact_metadata (key, value) VALUES (?, ?)",
        (
            ("step", "Step 1"),
            (
                "summary",
                "- Validated local SQLite copy of six synthetic CSV sources.\n"
                "- Customer and instrument indexes support entitlement reads.\n"
                "- Rebuild with python -m customer_assistant.database.",
            ),
        ),
    )


def load_database(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    db_path: Path = DEFAULT_DB_PATH,
) -> ValidationReport:
    """Validate all six files, then replace the SQLite demo copy in one transaction."""

    # Do every source and cross-file check before touching the target DB. That
    # makes a rejected CSV a reportable input error rather than a partial
    # replacement of an earlier known-good database.
    records, report = _read_and_validate(Path(source_dir))
    if not report.passed:
        raise ValidationError(report)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        # Treat dropping old tables, creating the schema, and loading all six
        # files as one refresh. A constraint or I/O error must roll it all back,
        # so readers do not see a mixture of old and new source records. The
        # separately managed audit_events table is not dropped here; its
        # retention policy belongs to audit.py and the Streamlit app.
        connection.execute("BEGIN IMMEDIATE")
        for name in reversed(HEADERS):
            connection.execute(f'DROP TABLE IF EXISTS "{name.removesuffix(".csv")}"')
        _create_schema(connection)
        for name, headers in HEADERS.items():
            table = name.removesuffix(".csv")
            columns = ", ".join(f'"{col}"' for col in headers)
            placeholders = ", ".join("?" for _ in headers)
            values = ([row[col] for col in headers] for row in records[name])
            connection.executemany(
                f'INSERT INTO "{table}" ({columns}) VALUES ({placeholders})',
                values,
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    report.database = str(db_path.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        report = (
            validate_source(args.source_dir)
            if args.validate_only
            else load_database(args.source_dir, args.db_path)
        )
    except ValidationError as exc:
        report = exc.report
    # Persist the same report format on success and on a rejected load. This
    # gives the next person a concrete failed check and source file to inspect.
    payload = json.dumps(report.as_dict(), indent=2)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
