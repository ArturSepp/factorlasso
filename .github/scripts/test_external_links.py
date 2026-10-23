"""Offline regressions for confirmed failures, remote outages, and builder crashes."""

import json
import contextlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import check_external_links as checker
import check_new_references as references


class Handler(BaseHTTPRequestHandler):
    """Serve real HTTP outcomes without contacting external websites."""

    def do_GET(self):
        code = {"/gone": 410, "/missing": 404, "/outage": 504, "/blocked": 403}.get(
            self.path, 200
        )
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b'<html><h1 id="present">Reference</h1></html>')

    def log_message(self, *args):
        pass


class CrossrefTests(unittest.TestCase):
    """A blocked DOI needs matching registry evidence before its warning is cleared."""

    doi = "10.1198/jasa.2009.tm08744"
    uri = "https://doi.org/" + doi

    def metadata(self, doi=None):
        """Return a realistic Crossref singleton with an independently registered landing URL."""
        doi = doi or self.doi
        return {"status": "ok", "message-type": "work", "message": {
            "DOI": doi, "title": ["The Dependent Wild Bootstrap"],
            "resource": {"primary": {"URL": "https://www.tandfonline.com/doi/full/" + doi}},
        }}

    @contextlib.contextmanager
    def responses(self, outcomes, code=403):
        """Block publisher requests and provide bounded, fully offline registry responses."""
        outcomes = iter(outcomes)

        def fetch(request, timeout):
            """Return the next registry response while retaining request evidence."""
            if request.full_url.startswith("https://api.crossref.org/works/"):
                outcome = next(outcomes)
                if isinstance(outcome, Exception):
                    raise outcome
                body = outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode()
                return io.BytesIO(body)
            raise HTTPError(request.full_url, code, "Publisher response", {}, None)

        with patch.object(references, "urlopen", side_effect=fetch) as mocked, patch.object(
            references.time, "sleep"
        ):
            yield mocked

    def test_all_seven_blocked_dois_are_verified(self):
        """Every audited DOI recovers from forbidden and rate-limited publisher responses."""
        dois = [
            "10.1080/01621459.1994.10476836", "10.1093/biomet/48.3-4.419",
            "10.1093/biomet/58.3.545", "10.1093/biomet/74.3.579",
            "10.1198/004017005000000517", self.doi, "10.1198/jasa.2011.tm10159",
        ]
        for doi in dois:
            for code in (403, 429):
                with self.subTest(doi=doi, code=code), self.responses(
                    [self.metadata(doi)], code=code
                ) as mocked:
                    uri = "https://doi.org/" + doi
                    result = references.inspect_url(uri)
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["verification"]["source"], "crossref")
                    self.assertEqual(result["verification"]["doi"], doi)
                    self.assertEqual(result["url"], uri)
                    self.assertEqual(mocked.call_count, 2)

    def test_encoded_legacy_doi_and_case_insensitive_identity(self):
        """Decode a legacy resolver URL while encoding the exact registry identifier safely."""
        uri = "https://dx.doi.org/10.1198%2FJASA.2009.TM08744"
        with self.responses([self.metadata()]) as mocked:
            result = references.inspect_url(uri)
        self.assertEqual(result["status"], "ok")
        request = mocked.call_args_list[1].args[0]
        self.assertEqual(request.full_url,
                         "https://api.crossref.org/works/10.1198%2FJASA.2009.TM08744")
        self.assertEqual(request.get_header("Accept"), "application/json")

    def test_non_doi_links_fragments_and_confirmed_failures_are_not_overridden(self):
        """Registry evidence cannot prove a page anchor, a query, or a missing publisher page."""
        cases = [
            ("https://example.com/" + self.doi, 403, "deferred"),
            ("https://doi.org.example.com/" + self.doi, 403, "deferred"),
            ("https://doi.org/not-a-doi", 403, "deferred"),
            (self.uri + "#missing", 403, "deferred"),
            (self.uri + "?download=true", 403, "deferred"),
            (self.uri, 404, "broken"), (self.uri, 410, "broken"),
            (self.uri, 503, "deferred"),
        ]
        for uri, code, expected in cases:
            with self.subTest(uri=uri, code=code), self.responses([], code=code) as mocked:
                self.assertEqual(references.inspect_url(uri)["status"], expected)
                self.assertTrue(all(not call.args[0].full_url.startswith(
                    "https://api.crossref.org/") for call in mocked.call_args_list))

    def test_registry_unavailability_retries_without_silently_passing(self):
        """An unavailable registry remains advisory; transient failures have bounded retries."""
        failures = [HTTPError(self.uri, code, "Registry error", {}, None)
                    for code in (403, 404, 410, 429, 500, 503)]
        failures += [URLError("offline"), TimeoutError("timeout")]
        for failure in failures:
            with self.subTest(failure=str(failure)), self.responses([failure] * 3) as mocked:
                result = references.inspect_url(self.uri)
                self.assertEqual(result["status"], "deferred")
                self.assertIn("Crossref", result["reason"])
                self.assertLessEqual(mocked.call_count, 4)
        # Crossref does not own every DOI: its 404 is not proof that a DOI is invalid.
        for failure in (failures[3], failures[5], failures[-1]):
            with self.subTest(recovery=str(failure)), self.responses(
                [failure, self.metadata()]
            ) as mocked:
                self.assertEqual(references.inspect_url(self.uri)["status"], "ok")
                self.assertEqual(mocked.call_count, 3)

    def test_malformed_or_unrelated_registry_records_do_not_verify_a_link(self):
        """HTTP success alone cannot hide a mismatched DOI or incomplete metadata."""
        mismatch = self.metadata("10.1234/different")
        missing_title = self.metadata()
        missing_title["message"]["title"] = []
        missing_landing = self.metadata()
        missing_landing["message"]["resource"] = {"primary": {"URL": "not-a-url"}}
        payloads = [b"not JSON", b"x" * (references.MAX_BODY + 1), [], {},
                    {"status": "ok", "message-type": "work", "message": None},
                    mismatch, missing_title, missing_landing]
        for payload in payloads:
            with self.subTest(payload=str(payload)[:80]), self.responses([payload]):
                result = references.inspect_url(self.uri)
                self.assertEqual(result["status"], "deferred")
                self.assertIn("Crossref", result["reason"])

    def test_verified_doi_clears_warning_and_retains_registry_evidence(self):
        """Exercise the final health report and annotations without real network requests."""
        record = {"uri": self.uri, "status": "broken", "info": "403 Forbidden"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def fake_run(command, check):
                """Write the same blocked-link evidence that Sphinx supplies."""
                (output / "output.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
                return type("Completed", (), {"returncode": 1})()

            captured = io.StringIO()
            with self.responses([self.metadata()]), patch.object(
                checker.subprocess, "run", fake_run
            ), contextlib.redirect_stdout(captured):
                self.assertEqual(checker.main(["--output-dir", str(output)]), 0)
            report = json.loads((output / "confirmed.json").read_text(encoding="utf-8"))
            self.assertEqual(report[0]["verification"]["source"], "crossref")
            self.assertEqual(report[0]["sphinx"], record)
            self.assertIn("Verified through Crossref:", captured.getvalue())
            self.assertIn("0 confirmed broken; 0 unavailable", captured.getvalue())
            self.assertNotIn("::warning::", captured.getvalue())


class ExternalLinkTests(unittest.TestCase):
    """A server outage is advisory; a confirmed 404, 410, or missing anchor is fatal."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.address = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_real_http_outcomes_and_anchor_confirmation(self):
        paths = ["/missing", "/gone", "/outage", "/blocked", "/ok#absent", "/ok#present"]
        records = [{"uri": self.address + path, "status": "broken"} for path in paths]
        results = checker.confirm_results(records)
        self.assertEqual(
            [r["status"] for r in results],
            ["broken", "broken", "deferred", "deferred", "broken", "ok"],
        )
        self.assertEqual(results[2]["sphinx"], records[2])

    def test_sphinx_service_unavailable_remains_visible(self):
        record = {"uri": self.address + "/outage", "status": "ignored",
                  "info": "service unavailable"}
        self.assertEqual(checker.confirm_results([record])[0]["status"], "deferred")

    def test_github_namespaced_anchors_keep_missing_sections_blocking(self):
        """Recognize GitHub content IDs without accepting absent or other-site anchors."""
        body = (b'<html><a id="user-content-persistence"></a>'
                b'<a id="user-content-fitted-state-and-input-alignment"></a></html>')
        cases = [
            ("github.com", "persistence", "ok"),
            ("github.com", "fitted-state-and-input-alignment", "ok"),
            ("github.com", "fitted%2Dstate%2Dand%2Dinput%2Dalignment", "ok"),
            ("github.com", "missing-section", "broken"),
            ("example.com", "persistence", "broken"),
            ("github.com.example.com", "persistence", "broken"),
        ]
        for host, fragment, expected in cases:
            address = f"https://{host}/ArturSepp/factorlasso/blob/main/COMPATIBILITY.md"
            record = {"uri": address + "#" + fragment, "status": "broken"}
            with self.subTest(host=host, fragment=fragment), patch.object(
                references, "urlopen"
            ) as fetch, patch.object(references.time, "sleep"):
                response = fetch.return_value.__enter__.return_value
                response.headers = {"Content-Type": "text/html"}
                response.geturl.return_value = address
                response.read.return_value = body
                self.assertEqual(checker.confirm_results([record])[0]["status"], expected)

    def run_builder(self, records, returncode=1):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def fake_run(command, check):
                (output / "output.json").write_text(
                    "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
                )
                return type("Completed", (), {"returncode": returncode})()

            with patch.object(checker.subprocess, "run", fake_run), contextlib.redirect_stdout(
                io.StringIO()
            ):
                return checker.main(["--output-dir", str(output)])

    def test_real_sphinx_report_keeps_broken_links_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "conf.py").write_text('project = "Link regression"\n', encoding="utf-8")
            (source / "index.rst").write_text(
                "References\n==========\n\n"
                f"`Missing <{self.address}/missing>`_\n\n"
                f"`Outage <{self.address}/outage>`_\n",
                encoding="utf-8",
            )
            output = Path(temporary) / "build"
            with contextlib.redirect_stdout(io.StringIO()):
                result = checker.main(["--source", str(source), "--output-dir", str(output)])
            self.assertEqual(result, 1)
            report = json.loads((output / "confirmed.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(r["status"] for r in report), ["broken", "deferred"])

    def test_unavailability_does_not_fail_run(self):
        self.assertEqual(self.run_builder([
            {"uri": self.address + "/outage", "status": "broken"}
        ]), 0)

    def test_builder_crash_after_partial_report_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "conf.py").write_text(
                'project = "Crash regression"\n'
                'def setup(app):\n'
                '    def crash(app, exception):\n'
                '        raise RuntimeError("intentional build-finished failure")\n'
                '    app.connect("build-finished", crash)\n', encoding="utf-8"
            )
            (source / "index.rst").write_text(
                "References\n==========\n\n"
                f"`Outage <{self.address}/outage>`_\n", encoding="utf-8"
            )
            output = Path(temporary) / "build"
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    checker.main(["--source", str(source), "--output-dir", str(output)])
            self.assertEqual(failure.exception.code, 2)
            self.assertTrue((output / "output.json").is_file())

    def test_confirmed_missing_link_fails_even_with_an_outage(self):
        self.assertEqual(self.run_builder([
            {"uri": self.address + "/outage", "status": "broken"},
            {"uri": self.address + "/missing", "status": "broken"},
        ]), 1)

    def test_builder_error_is_never_overridden(self):
        with self.assertRaises(SystemExit):
            self.run_builder([], returncode=2)

    def test_stale_report_cannot_hide_builder_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "output.json").write_text('{"status":"working"}\n', encoding="utf-8")
            completed = type("Completed", (), {"returncode": 2})()
            with patch.object(checker.subprocess, "run", return_value=completed):
                with self.assertRaises(SystemExit):
                    checker.main(["--output-dir", str(output)])


if __name__ == "__main__":
    unittest.main()
