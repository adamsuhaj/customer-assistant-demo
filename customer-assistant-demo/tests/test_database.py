"""Step 1 - Verify validation and loading of the six synthetic CSV files.

- Confirm load_database() creates the expected rows, foreign-key links, and indexes.
- Require eleven orders, six for Alice and five for Bob, so ID-free lists exercise more
  than one status and ownership relationship.
- Confirm a repeat load replaces rows instead of duplicating them.
- Corrupt only temporary CSV copies to check named validation errors and rollback.
- Compare source hashes so the loader cannot silently edit the supplied datasets.
"""

import csv
import hashlib
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from customer_assistant.database import (
    DEFAULT_SOURCE_DIR,
    HEADERS,
    ValidationError,
    load_database,
    validate_source,
)


def file_hash(path: Path) -> str:
    # Compare bytes, not parsed rows: a loader that rewrites quoting or line
    # endings would still be an unwanted change to the supplied datasets.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_orders(path: Path, change) -> None:
    """Change only a temporary order fixture to exercise validator failures."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    change(columns, rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class DatabaseLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets a disposable source copy so malformed fixtures cannot
        # touch the six project CSVs used by the live demo.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in HEADERS:
            shutil.copy2(DEFAULT_SOURCE_DIR / name, self.source / name)
        self.db_path = self.root / "local" / "demo.sqlite3"

    def test_loads_all_six_files_repeatably_without_changing_sources(self) -> None:
        # Counts make missing demo fixtures visible before they become a
        # misleading no-match answer in a later assistant step.
        before = {name: file_hash(self.source / name) for name in HEADERS}
        expected_counts = {
            "customers.csv": 2,
            "instruments.csv": 2,
            "orders.csv": 11,
            "service_history.csv": 3,
            "troubleshooting_articles.csv": 2,
            "evaluation_cases.csv": 6,
        }

        # A second load must replace rows rather than silently double them.
        for _ in range(2):
            report = load_database(self.source, self.db_path)
            self.assertTrue(report.passed)
            self.assertEqual(
                {file.file: file.rows for file in report.files}, expected_counts
            )
            self.assertTrue(all(not file.failed_checks for file in report.files))

        with closing(sqlite3.connect(self.db_path)) as connection:
            for filename, expected in expected_counts.items():
                table = filename.removesuffix(".csv")
                count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                self.assertEqual(count, expected)
            order = connection.execute(
                "SELECT order_id, order_status, typeof(order_id) FROM orders "
                "WHERE order_id = ?", ("DEMO-ORD-1007",)
            ).fetchone()
            self.assertEqual(order, ("DEMO-ORD-1007", "in_transit", "text"))
            self.assertEqual(
                connection.execute(
                    "SELECT customer_id, COUNT(*) FROM orders "
                    "GROUP BY customer_id ORDER BY customer_id"
                ).fetchall(),
                [("CUST-1001", 6), ("CUST-1002", 5)],
            )
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            indexes = {
                name
                for table in ("instruments", "orders", "service_history")
                for _, name, *_ in connection.execute(f'PRAGMA index_list("{table}")')
            }
            self.assertTrue({
                "idx_instruments_customer_id", "idx_orders_customer_id",
                "idx_orders_instrument_id", "idx_service_history_customer_id",
                "idx_service_history_instrument_id",
            }.issubset(indexes))
            metadata = dict(connection.execute(
                "SELECT key, value FROM artifact_metadata"
            ).fetchall())
            self.assertEqual(metadata["step"], "Step 1")
            self.assertIn("- Validated local SQLite copy", metadata["summary"])

        self.assertEqual(
            {name: file_hash(self.source / name) for name in HEADERS}, before
        )

    def test_broken_instrument_foreign_key_fails_without_replacing_database(self) -> None:
        # Seed a good copy first; the later invalid CSV must leave it intact.
        load_database(self.source, self.db_path)
        orders_path = self.source / "orders.csv"
        edit_orders(
            orders_path,
            lambda _columns, rows: rows[0].__setitem__(
                "instrument_id", "DEMO-INS-MISSING"
            ),
        )

        with self.assertRaises(ValidationError) as caught:
            load_database(self.source, self.db_path)
        report = caught.exception.report
        orders_report = next(file for file in report.files if file.file == "orders.csv")
        self.assertEqual(orders_report.rows, 11)
        self.assertIn(
            "instrument_foreign_key",
            {failure.check for failure in orders_report.failed_checks},
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 11)
            self.assertEqual(
                connection.execute(
                    "SELECT instrument_id FROM orders WHERE order_id = ?",
                    ("DEMO-ORD-1007",),
                ).fetchone()[0],
                "DEMO-INS-1001",
            )

    def test_rejects_missing_header_duplicate_id_bad_status_and_wrong_owner(self) -> None:
        # These cases represent the distinct validation gates requested by
        # the build guide; each starts from a clean order fixture.
        def missing_header(columns, rows):
            columns.remove("order_status")
            for row in rows:
                del row["order_status"]

        changes = (
            (missing_header, "headers"),
            (lambda _columns, rows: rows[1].__setitem__("order_id", rows[0]["order_id"]), "primary_id"),
            (lambda _columns, rows: rows[0].__setitem__("order_status", "shipped"), "order_status"),
            (lambda _columns, rows: rows[0].__setitem__("customer_id", "CUST-1002"), "instrument_customer_match"),
        )
        for change, expected_check in changes:
            with self.subTest(expected_check=expected_check):
                shutil.copy2(DEFAULT_SOURCE_DIR / "orders.csv", self.source / "orders.csv")
                edit_orders(self.source / "orders.csv", change)
                report = validate_source(self.source)
                self.assertFalse(report.passed)
                orders_report = next(file for file in report.files if file.file == "orders.csv")
                self.assertEqual(orders_report.rows, 11)
                self.assertIn(
                    expected_check,
                    {failure.check for failure in orders_report.failed_checks},
                )

    def test_report_keeps_original_row_number_after_a_malformed_row(self) -> None:
        orders_path = self.source / "orders.csv"
        edit_orders(
            orders_path,
            lambda _columns, rows: rows[1].__setitem__(
                "order_id", rows[0]["order_id"]
            ),
        )
        lines = orders_path.read_text(encoding="utf-8").splitlines(keepends=True)
        lines.insert(2, "too,few\n")
        orders_path.write_text("".join(lines), encoding="utf-8")

        report = validate_source(self.source)
        orders_report = next(file for file in report.files if file.file == "orders.csv")
        self.assertEqual(orders_report.rows, 12)
        self.assertTrue(any(
            failure.check == "row_width" and "row 3:" in failure.detail
            for failure in orders_report.failed_checks
        ))
        self.assertTrue(any(
            failure.check == "primary_id" and "row 4:" in failure.detail
            for failure in orders_report.failed_checks
        ))


if __name__ == "__main__":
    unittest.main()
