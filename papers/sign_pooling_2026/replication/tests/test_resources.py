"""Offline resource and publication contracts; no research solver required."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

REPLICATION = Path(__file__).resolve().parents[1]
PAPER = REPLICATION.parent


def load_paths():
    """Load the path module independently of the package's optional dependencies."""
    spec = importlib.util.spec_from_file_location("csda_paths", REPLICATION / "paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResourceTests(unittest.TestCase):
    """Check frozen inputs and make accidental manuscript overwrites fail early."""

    def test_frozen_input_hashes(self):
        """Input content survives path moves and ordinary Git newline conversion."""
        data = REPLICATION / "data"
        manifest = json.loads((data / "sha256.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 18)
        for name, expected in manifest.items():
            with self.subTest(name=name):
                content = (data / name).read_bytes()
                if name.endswith(".csv"):
                    content = content.replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected)

    def test_current_manuscript_dependencies_are_approved(self):
        """Every local figure/table/bibliography dependency is in the exact allowlist."""
        text = (PAPER / "paper/article.tex").read_text(encoding="utf-8")
        approved = set((PAPER / ".gitignore").read_text().splitlines())
        patterns = ((r"\\input\{([^}]+)\}", ".tex"),
                    (r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", ""),
                    (r"\\bibliography\{([^}]+)\}", ".bib"))
        for pattern, extension in patterns:
            for name in re.findall(pattern, text):
                relative = name if name.endswith(extension) else name + extension
                with self.subTest(name=relative):
                    self.assertTrue((PAPER / "paper" / relative).is_file())
                    self.assertIn("!/paper/" + relative, approved)

    def test_output_override_is_external(self):
        """A caller can select a fresh C-local run directory without creating it."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new-run"
            with patch.dict(os.environ, {"FACTORLASSO_PAPER_OUTPUT_DIR": str(target)}):
                module = load_paths()
                self.assertEqual(module.RESULTS_DIR, target / "results")
                self.assertEqual(module.REFERENCE_DIR, REPLICATION / "data/reference")
                self.assertFalse(target.exists())

    def test_output_cannot_overwrite_checkout_or_onedrive(self):
        """Reject both the manuscript directory and a OneDrive runtime override."""
        for directory in (PAPER / "paper", PAPER / "replication/data/reference",
                          Path(tempfile.gettempdir()) / "OneDrive" / "run"):
            with self.subTest(directory=directory):
                with patch.dict(os.environ, {"FACTORLASSO_PAPER_OUTPUT_DIR": str(directory)}):
                    with self.assertRaises(ValueError):
                        load_paths()

    def test_output_override_must_be_absolute(self):
        """Working-directory changes cannot redirect output back into the checkout."""
        with patch.dict(os.environ, {"FACTORLASSO_PAPER_OUTPUT_DIR": "results"}):
            with self.assertRaises(ValueError):
                load_paths()


if __name__ == "__main__":
    unittest.main()
