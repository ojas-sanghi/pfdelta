#!/usr/bin/env python3
"""Deploy a generated interactive export to Vercel via the REST API."""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

API_BASE_URL = "https://api.vercel.com"
POLL_INTERVAL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 300.0


def main() -> int:
    args = parse_args()
    site_dir = Path(args.site_dir).resolve()
    if not site_dir.is_dir():
        raise SystemExit(f"Site directory not found: {site_dir}")

    token = os.environ.get("VERCEL_TOKEN")
    if not token:
        raise SystemExit(
            "VERCEL_TOKEN is required for REST API deploys. "
            "Create a Vercel token and export it before running this script."
        )

    link_metadata = load_link_metadata(site_dir)
    query = build_owner_query(link_metadata)
    project_hint = resolve_project_hint(link_metadata)
    if not project_hint:
        raise SystemExit(
            "Could not determine the Vercel project to deploy to. "
            "Set VERCEL_PROJECT_NAME or VERCEL_PROJECT_ID, or provide a linked "
            f"project file at {site_dir / '.vercel' / 'project.json'}."
        )

    project_info = api_json(
        method="GET",
        path=f"/v9/projects/{urllib.parse.quote(str(project_hint), safe='')}",
        token=token,
        query=query,
    )
    project_name = project_info.get("name")
    project_id = project_info.get("id")
    if not project_name or not project_id:
        raise SystemExit(
            f"Could not resolve a valid project from hint {project_hint!r}. Response: {project_info}"
        )

    files = collect_files(site_dir)
    if not files:
        raise SystemExit(f"No deployable files were found under {site_dir}")

    target = args.target
    request_body = {
        "name": project_name,
        "project": project_name,
        "target": target,
        "files": files,
        "meta": {
            "deployedBy": "deploy_interactive_exports_api.py",
            "siteDir": str(site_dir),
        },
    }

    deployment = api_json(
        method="POST",
        path="/v13/deployments",
        token=token,
        query=query,
        body=request_body,
    )
    deployment_id = deployment.get("id")
    deployment_url = deployment.get("url")
    if not deployment_id:
        raise SystemExit(f"Vercel deployment response did not include an id: {deployment}")

    print(f"Created deployment {deployment_id} for project {project_name} ({target}).")
    if deployment_url:
        print(f"Deployment URL: https://{deployment_url}")

    final_deployment = wait_for_ready_state(
        deployment_id=deployment_id,
        token=token,
        query=query,
        timeout_seconds=args.timeout,
    )

    ready_state = final_deployment.get("readyState") or final_deployment.get("status")
    final_url = final_deployment.get("url") or deployment_url
    aliases = final_deployment.get("alias") or []

    print(f"Final state: {ready_state}")
    if final_url:
        print(f"Deployment URL: https://{final_url}")
    if aliases:
        print("Aliases:")
        for alias in aliases:
            print(f" - https://{alias}")

    if ready_state != "READY":
        error_message = (
            final_deployment.get("errorMessage")
            or final_deployment.get("readyStateReason")
            or final_deployment.get("errorCode")
            or "Deployment did not reach READY state."
        )
        raise SystemExit(error_message)

    if target == "production":
        print(
            "Production deployment completed. If the project has production domains attached, "
            "this deployment now backs the stable production URL."
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", help="Directory containing the generated static export")
    parser.add_argument(
        "--target",
        choices=("production", "preview"),
        default="production",
        help="Vercel deployment target",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Seconds to wait for the deployment to reach a terminal state",
    )
    return parser.parse_args()


def load_link_metadata(site_dir: Path) -> Dict[str, Any]:
    candidates = [
        site_dir / ".vercel" / "project.json",
        Path.cwd() / ".vercel" / "project.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse Vercel project metadata at {candidate}: {exc}")
    return {}


def build_owner_query(link_metadata: Dict[str, Any]) -> Dict[str, str]:
    query = {}  # type: Dict[str, str]
    team_id = os.environ.get("VERCEL_TEAM_ID")
    team_slug = os.environ.get("VERCEL_TEAM_SLUG")
    if team_id:
        query["teamId"] = team_id
        return query
    if team_slug:
        query["slug"] = team_slug
        return query

    org_id = str(link_metadata.get("orgId", "")).strip()
    if org_id.startswith("team_"):
        query["teamId"] = org_id
    return query


def resolve_project_hint(link_metadata: Dict[str, Any]) -> str:
    for env_name in ("VERCEL_PROJECT_ID", "VERCEL_PROJECT_NAME", "VERCEL_PROJECT_ID_OR_NAME"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    project_id = str(link_metadata.get("projectId", "")).strip()
    return project_id


def collect_files(site_dir: Path) -> List[Dict[str, str]]:
    files = []  # type: List[Dict[str, str]]
    for path in sorted(site_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(site_dir).as_posix()
        if relative_path.startswith(".vercel/"):
            continue
        file_bytes = path.read_bytes()
        files.append(
            {
                "file": relative_path,
                "data": base64.b64encode(file_bytes).decode("ascii"),
                "encoding": "base64",
            }
        )
    return files


def wait_for_ready_state(
    deployment_id: str,
    token: str,
    query: Dict[str, str],
    timeout_seconds: float,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while True:
        deployment = api_json(
            method="GET",
            path=f"/v13/deployments/{deployment_id}",
            token=token,
            query=query,
        )
        ready_state = deployment.get("readyState") or deployment.get("status")
        if ready_state in {"READY", "ERROR", "CANCELED"}:
            return deployment
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out waiting for deployment {deployment_id} to finish. Last state: {ready_state}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def api_json(
    method: str,
    path: str,
    token: str,
    query: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    data = None  # type: Optional[bytes]
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = decode_json_or_text(raw)
        raise SystemExit(f"Vercel API request failed ({exc.code}) for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to reach Vercel API at {url}: {exc}") from exc

    if not raw:
        return {}
    parsed = decode_json_or_text(raw)
    if isinstance(parsed, dict):
        return parsed
    raise SystemExit(f"Expected JSON response from {url}, received: {parsed}")


def decode_json_or_text(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


if __name__ == "__main__":
    sys.exit(main())
