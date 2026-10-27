"""core.library.db.LibraryDB.tx: nested, only the outermost commits; an inner failure undoes its own writes; an error
after which SQLite has ended the transaction itself goes on as it was raised."""
import os
import tempfile
import unittest

from core.library.db import LibraryDB


class NestedTransactionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = LibraryDB(os.path.join(self._tmp.name, "library.db"))
        self.db.connect()
        self.addCleanup(self.db.close)
        self.db.conn.execute("CREATE TABLE t (v INTEGER)")
        self.db.conn.commit()

    def values(self):
        return [v for (v,) in self.db.conn.execute("SELECT v FROM t ORDER BY v")]

    def test_only_the_outermost_commits_and_an_inner_failure_is_its_own(self):
        with self.db.tx():
            self.db.conn.execute("INSERT INTO t VALUES (1)")
            with self.assertRaises(ValueError):
                with self.db.tx():
                    self.db.conn.execute("INSERT INTO t VALUES (2)")
                    raise ValueError("inner")
            with self.db.tx():
                self.db.conn.execute("INSERT INTO t VALUES (3)")
            self.assertTrue(self.db.conn.in_transaction)       # nothing committed yet
        self.assertEqual(self.values(), [1, 3])

    def test_the_original_error_when_sqlite_ended_the_transaction(self):
        with self.assertRaises(ValueError) as caught:
            with self.db.tx():
                self.db.conn.execute("INSERT INTO t VALUES (1)")
                with self.db.tx():
                    self.db.conn.execute("ROLLBACK")       # as SQLite does on a full disk or an I/O error
                    raise ValueError("the cause")
        self.assertEqual(str(caught.exception), "the cause")
        self.assertEqual(self.values(), [])
        self.assertFalse(self.db.conn.in_transaction)


if __name__ == "__main__":
    unittest.main()
