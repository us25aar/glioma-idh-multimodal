"""Authenticate public TCIA downloads and retry transient HTTP failures."""
import base64
import html
import http.cookiejar
import json
import re
import time
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import build_opener, HTTPCookieProcessor, Request
from urllib.error import HTTPError, URLError

host = "https://faspex.cancerimagingarchive.net"
opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))


def request(url, data=None, headers=None):
    headers = {"User-Agent": "Glioma-MSc-research", **(headers or {})}
    if data is not None:
        data = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    for attempt in range(4):
        try:
            return opener.open(Request(url, data=data, headers=headers), timeout=45)
        except HTTPError as error:
            if error.code not in {429, 502, 503, 504} or attempt == 3:
                raise
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def authenticate():
    page = request("https://www.cancerimagingarchive.net/collection/utsw-glioma/").read().decode()
    link = html.unescape(re.search(r'https://faspex[^"\s]+', page).group())
    context = json.loads(base64.b64decode(parse_qs(urlparse(link).query)["context"][0]))
    config = request(host + "/aspera/faspex/config.js").read().decode()
    value = lambda key: re.search(rf"{key}:\s*'([^']+)'", config).group(1)
    api = host + value("api_url")
    client_id, redirect = value("client_id"), host + value("redirect_uri")
    state = base64.b64encode(json.dumps(context).encode()).decode()
    auth = api.replace("/api/v5", "/auth")
    result = request(auth + "/authorize_public_link?" + urlencode({"response_type": "code", "client_id": client_id, "redirect_uri": redirect, "state": state}))
    query = parse_qs(urlparse(result.url).query)
    if "code" not in query:
        raise RuntimeError("Public-link flow did not return an authorization code.")
    token = json.loads(request(auth + "/token", {"code": query["code"][0], "state": query.get("state", [state])[0],
                                                  "grant_type": "authorization_code", "client_id": client_id, "redirect_uri": redirect}).read())
    headers = {"Authorization": "Bearer " + token["access_token"]}
    package_id = context["id"]
    return api, headers, package_id
