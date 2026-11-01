"""Regression checks for private publication and staged-policy enforcement."""

from pathlib import Path
import io
import tarfile
import tempfile
import unittest
import zipfile

from check_paper_policy import check_artifacts, check_repository, git


class PaperPolicyTests(unittest.TestCase):
    """Exercise actual temporary Git indexes, including force-added private files."""

    def setUp(self) -> None:
        """Create policy files and a local paper under the private defaults."""
        self.temp = tempfile.TemporaryDirectory(prefix="fl-paper-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")
        source = Path(__file__).resolve().parents[2]
        self.write(".gitignore", (source / ".gitignore").read_text(encoding="utf-8"))
        self.write("papers/AGENTS.md", "Paper contract\n")
        self.write("papers/README.md", "Private paper workspaces\n")
        self.write("papers/sign_pooling_2026/.gitignore", "!/paper/current.tex\n")
        self.write("papers/sign_pooling_2026/paper/current.tex", "Existing approved manuscript\n")
        self.write("papers/sign_pooling_2026/replication/reproduce.py", "print('example')\n")
        git(self.root, "add", ".")

    def write(self, name: str, text: str) -> None:
        """Write a fixture without depending on shell quoting or platform newlines."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_private_workspaces_are_not_added(self) -> None:
        """Ordinary staging includes only the policy and availability index."""
        names = git(self.root, "ls-files", "papers").decode().splitlines()
        self.assertEqual(names, ["papers/AGENTS.md", "papers/README.md"])
        self.assertEqual(check_repository(self.root), [])
        self.assertTrue((self.root / "papers/sign_pooling_2026/paper/current.tex").is_file())

    def test_force_added_private_material_fails(self) -> None:
        """Ignore rules do not protect an already staged or force-added file."""
        for section in ("paper", "replication", "private", "drafts", "agents",
                        "replication/data/local"):
            with self.subTest(section=section):
                name = f"papers/sign_pooling_2026/{section}/record.txt"
                self.write(name, "private\n")
                git(self.root, "add", "-f", name)
                self.assertTrue(any(name in error for error in check_repository(self.root)))
                git(self.root, "rm", "--cached", "-f", name)

    def test_nested_exception_cannot_publish_private_material(self) -> None:
        """Private section classification wins over deliberate ignore negations."""
        self.write("papers/sign_pooling_2026/private/.gitignore", "!record.txt\n")
        self.write("papers/sign_pooling_2026/private/record.txt", "private\n")
        git(self.root, "add", "-f", "papers/sign_pooling_2026/private")
        self.assertTrue(any("protected material" in error for error in check_repository(self.root)))

    def test_unapproved_pdf_fails(self) -> None:
        """Force-adding a manuscript PDF is not a publication decision."""
        name = "papers/sign_pooling_2026/paper/current.pdf"
        self.write(name, "pdf fixture\n")
        git(self.root, "add", "-f", name)
        self.assertTrue(any(name in error for error in check_repository(self.root)))

    def test_unstaged_exception_does_not_change_index_verdict(self) -> None:
        """The checker reads the staged .gitignore, not a more permissive worktree."""
        path = self.root / ".gitignore"
        correct = path.read_text()
        self.write(".gitignore", correct.replace("/papers/*\n", ""))
        git(self.root, "add", ".gitignore")
        self.assertTrue(check_repository(self.root))
        self.write(".gitignore", correct)
        self.assertTrue(check_repository(self.root))
        self.assertEqual(check_repository(self.root, worktree=True), [])
        git(self.root, "add", ".gitignore")
        self.assertEqual(check_repository(self.root), [])

    def test_missing_generic_protection_fails(self) -> None:
        """A policy regression is detected before a private file exists."""
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text().replace("/papers/*\n", ""))
        git(self.root, "add", ".gitignore")
        self.assertTrue(any("missing default" in error for error in check_repository(self.root)))

    def test_wildcard_publication_exception_fails(self) -> None:
        """A wildcard must not silently approve future manuscript versions."""
        self.write("papers/sign_pooling_2026/.gitignore", "!/paper/*.tex\n")
        git(self.root, "add", "-f", "papers/sign_pooling_2026/.gitignore")
        self.assertTrue(any("exact publication" in error for error in check_repository(self.root)))

    def test_open_directory_does_not_approve_all_figures(self) -> None:
        """Opening a parent directory cannot implicitly approve its contents."""
        self.write("papers/sign_pooling_2026/.gitignore", "!/paper/current.tex\n!/paper/figures/\n")
        self.write("papers/sign_pooling_2026/paper/figures/unreviewed.png", "image\n")
        git(self.root, "add", "-f", "papers/sign_pooling_2026")
        self.assertTrue(any("exact exception" in error for error in check_repository(self.root)))

    def test_other_paper_cannot_be_reopened(self) -> None:
        """No paper is approved even if root ignores are weakened."""
        name = "papers/jss_2026/replication/reproduce.py"
        self.write(name, "print('local only')\n")
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text() + "\n!/papers/jss_2026/\n")
        git(self.root, "add", ".gitignore", name)
        self.assertTrue(any("protected material" in e for e in check_repository(self.root)))

    def test_old_sign_pooling_exception_cannot_be_restored_silently(self) -> None:
        """An old exact allowlist cannot override the withdrawn publication approval."""
        path = self.root / ".gitignore"
        self.write(".gitignore", path.read_text() + "\n!/papers/sign_pooling_2026/\n")
        git(self.root, "add", ".gitignore", "papers/sign_pooling_2026")
        errors = check_repository(self.root)
        self.assertTrue(any("protected material is tracked: papers/sign_pooling_2026/paper/current.tex"
                            in error for error in errors))

    def test_unapproved_static_input_fails(self) -> None:
        """A new data input requires an exact approval as well as provenance."""
        name = "papers/sign_pooling_2026/replication/data/vendor.csv"
        self.write(name, "unreviewed\n")
        git(self.root, "add", "-f", name)
        self.assertTrue(any(name in e for e in check_repository(self.root)))

    def test_source_and_wheel_workspaces_fail(self) -> None:
        """Distribution checks inspect archive members, including source archives."""
        directory = self.root / "dist"
        directory.mkdir()
        with zipfile.ZipFile(directory / "example.whl", "w") as archive:
            archive.writestr("papers/sign_pooling_2026/private/report.txt", "private")
        with tarfile.open(directory / "example.tar.gz", "w:gz") as archive:
            entry = tarfile.TarInfo("example/agents/ROADMAP.md")
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
        self.assertEqual(len(check_artifacts(directory)), 2)


if __name__ == "__main__":
    unittest.main()
