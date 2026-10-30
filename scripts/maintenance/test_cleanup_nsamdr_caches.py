"""Regression tests for safe NSAMDR cache-only cleanup."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("cleanup_nsamdr_caches.py")


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


class CleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        run("git", "init", "-q", cwd=self.root)
        (self.root / "tools/nsamdr/neural").mkdir(parents=True)
        (self.root / ".gitignore").write_text(
            "__pycache__/\n*.pyc\n.pytest_cache/\nartifacts/\nnode_modules/\n", encoding="utf-8"
        )
        self._write("tools/nsamdr/neural/core.py", "print('keep')\n")
        self._write("tools/nsamdr/neural/__pycache__/core.cpython.pyc", "CACHE")
        self._write("tools/nsamdr/neural/.pytest_cache/v/cache/nodeids", "CACHE")
        self._write("tools/nsamdr/neural/__pycache__/tracked.pyc", "TRACKED")
        self._write("tools/nsamdr/neural/__pycache__/keep.pt", "PRESERVE")
        self._write("tools/nsamdr/neural/node_modules/__pycache__/vendor.pyc", "PRESERVE")
        self._write("artifacts/nsamdr/__pycache__/checkpoint.pyc", "PRESERVE")
        self._write("other/__pycache__/other.pyc", "REPO")
        run("git", "add", ".gitignore", "tools/nsamdr/neural/core.py", cwd=self.root)
        run("git", "add", "-f", "tools/nsamdr/neural/__pycache__/tracked.pyc", cwd=self.root)

    def _write(self, name: str, value: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def command(self, *args: str) -> subprocess.CompletedProcess[str]:
        return run(sys.executable, str(SCRIPT), "--root", str(self.root), *args, cwd=self.root)

    def test_dry_run_is_non_destructive(self) -> None:
        p = self.command()
        self.assertIn("ignored cache files: 2", p.stdout)
        self.assertTrue((self.root / "tools/nsamdr/neural/__pycache__/core.cpython.pyc").exists())
        self.assertTrue((self.root / "tools/nsamdr/neural/.pytest_cache/v/cache/nodeids").exists())

    def test_apply_preserves_sources_tracked_files_dependencies_and_evidence(self) -> None:
        p = self.command("--apply")
        self.assertIn("Deleted 2 ignored cache files", p.stdout)
        for name in (
            "tools/nsamdr/neural/core.py",
            "tools/nsamdr/neural/__pycache__/tracked.pyc",
            "tools/nsamdr/neural/__pycache__/keep.pt",
            "tools/nsamdr/neural/node_modules/__pycache__/vendor.pyc",
            "artifacts/nsamdr/__pycache__/checkpoint.pyc",
            "other/__pycache__/other.pyc",
        ):
            self.assertTrue((self.root / name).exists(), name)
        self.assertIn("ignored cache files: 0", self.command().stdout)

    def test_repo_scope_only_expands_to_other_caches_not_artifacts(self) -> None:
        p = self.command("--scope", "repo", "--apply")
        self.assertIn("Deleted 3 ignored cache files", p.stdout)
        self.assertTrue((self.root / "artifacts/nsamdr/__pycache__/checkpoint.pyc").exists())

    def test_refuses_non_root(self) -> None:
        p = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root / "tools/nsamdr")],
            cwd=self.root, text=True, capture_output=True,
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("expected repository root", p.stderr)


if __name__ == "__main__":
    unittest.main()
