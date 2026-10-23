"""Check newly added public references without gating on unrelated server outages."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urldefrag, urlsplit
from urllib.request import Request, urlopen

MAX_BODY = 4 * 1024 * 1024


class Anchors(HTMLParser):
    """Collect static anchors, using the same rendered IDs a link checker can see."""

    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"id", "name"} and value:
                self.names.add(value)


def added_urls(patch):
    """Extract external URLs only from added Markdown/RST lines."""
    urls = set()
    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for candidate in re.findall(r'https?://[^\s<>`"\x27]+', line[1:]):
            # Stop at the link's closing parenthesis, including when emphasis follows it.
            depth = 0
            for index, character in enumerate(candidate):
                if character == "(":
                    depth += 1
                elif character == ")":
                    if depth == 0:
                        candidate = candidate[:index]
                        break
                    depth -= 1
            candidate = candidate.rstrip(".,;]}*")
            if (
                "{" not in candidate
                and "PACKAGE" not in candidate
                and "REPOSITORY" not in candidate
            ):
                urls.add(candidate)
    return sorted(urls)


def verify_blocked_doi(url):
    """Verify a blocked bare DOI using matching Crossref metadata, retaining its evidence.

    Crossref cannot validate publisher anchors or query parameters. Its 404 may also
    mean another registry owns the DOI, so registry failures remain unavailable.
    """
    parsed = urlsplit(url)
    doi = unquote(parsed.path[1:])
    if (parsed.scheme not in {"http", "https"}
            or parsed.netloc.lower() not in {"doi.org", "dx.doi.org"}
            or parsed.query or parsed.fragment
            or not re.fullmatch(r"10\.\d{4,9}/[^\s]+", doi)):
        return None
    api = "https://api.crossref.org/works/" + quote(doi, safe="")
    request = Request(api, headers={
        "Accept": "application/json", "User-Agent": "OSS-documentation-check/1.0",
    })
    last = "unavailable"
    for attempt in range(3):
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                raise ValueError("metadata exceeds the response limit")
            payload = json.loads(body)
            if (not isinstance(payload, dict) or payload.get("status") != "ok"
                    or payload.get("message-type") != "work"):
                raise ValueError("expected a successful work record")
            metadata = payload["message"]
            registered = metadata["DOI"]
            titles = metadata["title"]
            landing = metadata["resource"]["primary"]["URL"]
            if (not isinstance(registered, str) or registered.casefold() != doi.casefold()
                    or not isinstance(titles, list) or not titles
                    or not all(isinstance(title, str) and title.strip() for title in titles)
                    or not isinstance(landing, str)
                    or urlsplit(landing).scheme not in {"http", "https"}
                    or not urlsplit(landing).hostname):
                raise ValueError("DOI identity, title, or landing URL did not verify")
            return {"url": url, "status": "ok", "reason": "verified DOI through Crossref",
                    "verification": {"source": "crossref", "url": api, "doi": registered,
                                     "title": titles, "landing_url": landing}}
        except HTTPError as error:
            last = f"HTTP {error.code}"
            if error.code not in {429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, OSError) as error:
            last = str(error)
        except (ValueError, KeyError, TypeError) as error:
            last = f"invalid metadata ({error})"
            break
        if attempt < 2:
            time.sleep(2 ** attempt)
    return {"url": url, "status": "deferred",
            "reason": f"Crossref verification unavailable: {last}"}


def inspect_url(url):
    """Classify a confirmed broken reference separately from temporary unavailability."""
    address, fragment = urldefrag(url)
    last = "unavailable"
    for attempt in range(2):
        try:
            request = Request(address, headers={"User-Agent": "OSS-documentation-check/1.0"})
            with urlopen(request, timeout=15) as response:
                kind = response.headers.get("Content-Type", "")
                if not fragment or "html" not in kind:
                    return {"url": url, "status": "ok"}
                resolved_address = response.geturl()
                body = response.read(MAX_BODY + 1)
            if len(body) > MAX_BODY:
                return {"url": url, "status": "deferred", "reason": "large HTML requires review"}
            parser = Anchors()
            parser.feed(body.decode("utf-8", "replace"))
            anchor = unquote(fragment)
            # GitHub prefixes content IDs while its public permalink fragments stay bare.
            if anchor in parser.names or (
                urlsplit(resolved_address).hostname == "github.com"
                and f"user-content-{anchor}" in parser.names
            ):
                return {"url": url, "status": "ok"}
            last = f"static anchor #{fragment} not found"
        except HTTPError as error:
            if error.code in {403, 429}:
                verified = verify_blocked_doi(url)
                if verified is not None:
                    return verified
            if error.code not in {404, 410}:
                return {"url": url, "status": "deferred", "reason": f"HTTP {error.code}"}
            last = f"HTTP {error.code}"
        except (URLError, TimeoutError, OSError) as error:
            return {"url": url, "status": "deferred", "reason": str(error)}
        if attempt == 0:
            time.sleep(0.5)
    return {"url": url, "status": "broken", "reason": last}


def main():
    """Check introduced references; emit a machine-readable maintenance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patch = subprocess.check_output(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--unified=0",
            args.base,
            args.head,
            "--",
            "*.md",
            "*.rst",
        ],
        text=True,
        encoding="utf-8",
    )
    results = [inspect_url(url) for url in added_urls(patch)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['status']}: {result['url']} {result.get('reason', '')}")
    deferred = sum(item["status"] == "deferred" for item in results)
    print(
        f"Checked {len(results)} introduced references; {deferred} need external-health follow-up."
    )
    raise SystemExit(1 if any(item["status"] == "broken" for item in results) else 0)


if __name__ == "__main__":
    main()
