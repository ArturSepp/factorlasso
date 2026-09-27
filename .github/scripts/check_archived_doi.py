"""Check the Zenodo DOI through DataCite when Zenodo blocks automated link checks."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DOI = "10.5281/zenodo.21000294"
API = f"https://api.datacite.org/dois/{DOI}"
EXPECTED_TITLE = "Gated Cluster-Pooled Sign Constraints for Multi-Output Sparse Regression"


def _record() -> dict:
    """Fetch DOI metadata, retrying only temporary registry or network failures."""
    request = Request(API, headers={"Accept": "application/vnd.api+json",
                                    "User-Agent": "factorlasso-docs-link-health"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("DOI retry loop ended without a response")


def main() -> None:
    """Require a findable DOI that resolves to the cited Zenodo research bundle."""
    data = _record()["data"]
    metadata = data["attributes"]
    title = " ".join(item["title"] for item in metadata["titles"])
    landing = urlsplit(metadata["url"])
    if (data["id"].casefold() != DOI.casefold()
            or metadata["state"] != "findable"
            or landing.hostname != "zenodo.org"
            or EXPECTED_TITLE.casefold() not in title.casefold()):
        raise SystemExit(f"Unexpected DOI metadata for {DOI}: {metadata['url']}, {title}")
    print(f"PASS: {DOI} is findable and points to the cited Zenodo bundle")


if __name__ == "__main__":
    main()
