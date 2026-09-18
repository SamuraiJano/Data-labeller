#!/usr/bin/env python3
"""Bootstrap and launch Data Labeller from a single command.

Python and frontend dependencies are installed only when their manifests change.
The React application is rebuilt only when one of its inputs changes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
FRONTEND_DIR = ROOT / "frontend"
PYTHON_STAMP = VENV_DIR / ".data-labeller-dependencies"
NODE_STAMP = FRONTEND_DIR / "node_modules" / ".data-labeller-dependencies"
BUILD_STAMP = FRONTEND_DIR / "dist" / ".data-labeller-build"
STAMP_VERSION = "1"
MINIMUM_PYTHON = (3, 12)
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu130"


def _load_local_env(path: Path = ROOT / ".env") -> None:
    """Load simple KEY=VALUE workstation settings without overriding the shell."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"> {_command_text(command)}", flush=True)
    environment = os.environ.copy()
    if os.name == "nt" and "GIT_CONFIG_COUNT" not in environment:
        # Git-for-Windows can otherwise miss corporate/local CA roots while pip
        # resolves pinned Git dependencies. Schannel keeps verification enabled
        # and uses the same Windows trust store as the rest of the launcher.
        environment.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.sslBackend",
                "GIT_CONFIG_VALUE_0": "schannel",
            }
        )
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"launcher-stamp-v{STAMP_VERSION}\0".encode())
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _stamp_matches(path: Path, expected: str) -> bool:
    try:
        return path.read_text(encoding="utf-8").strip() == expected
    except OSError:
        return False


def _write_stamp(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _torch_can_use_cuda(python: Path) -> bool:
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import torch; "
                "raise SystemExit(not (torch.version.cuda is not None and torch.cuda.is_available()))"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
    )
    return probe.returncode == 0


def _nvidia_gpu_available() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False
    try:
        probe = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0 and bool(probe.stdout.strip())


def _ensure_cuda_torch(python: Path) -> None:
    requested_device = os.getenv("GROUNDING_DINO_DEVICE", "auto").strip().lower()
    if requested_device == "cpu" or _torch_can_use_cuda(python) or not _nvidia_gpu_available():
        return

    index_url = os.getenv("PYTORCH_CUDA_INDEX_URL", PYTORCH_CUDA_INDEX_URL).strip()
    print("NVIDIA GPU detected; installing the CUDA-enabled PyTorch build...", flush=True)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "torch",
            "torchvision",
            "--index-url",
            index_url,
        ]
    )
    if not _torch_can_use_cuda(python):
        raise RuntimeError(
            "CUDA PyTorch was installed, but PyTorch still cannot use the NVIDIA GPU. "
            "Update the NVIDIA display driver and restart the launcher."
        )


def _base_python() -> list[str]:
    if sys.version_info >= MINIMUM_PYTHON:
        return [sys.executable]
    if os.name == "nt" and shutil.which("py"):
        probe = subprocess.run(
            ["py", "-3.12", "-c", "import sys; raise SystemExit(sys.version_info < (3, 12))"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return ["py", "-3.12"]
    raise RuntimeError("Data Labeller requires Python 3.12 or newer.")


def _ensure_python_environment() -> Path:
    python = _venv_python()
    if not python.exists():
        print("Creating the local Python environment...", flush=True)
        _run([*_base_python(), "-m", "venv", str(VENV_DIR)])

    version = subprocess.run(
        [str(python), "-c", "import sys; raise SystemExit(sys.version_info < (3, 12))"],
        cwd=ROOT,
    )
    if version.returncode:
        raise RuntimeError(
            f"{VENV_DIR} uses an unsupported Python version. Remove that environment and run this command again."
        )

    dependency_hash = _fingerprint([ROOT / "pyproject.toml"])
    dependency_probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.util; "
                "names=('fastapi','uvicorn','PIL','yaml','numpy','cv2','torch','transformers','sam2'); "
                "raise SystemExit(any(importlib.util.find_spec(name) is None for name in names))"
            ),
        ],
        cwd=ROOT,
    )
    if not _stamp_matches(PYTHON_STAMP, dependency_hash) or dependency_probe.returncode:
        print("Installing Python dependencies...", flush=True)
        _run([str(python), "-m", "pip", "install", "-e", f"{ROOT}[detection,sam2]"])
        _write_stamp(PYTHON_STAMP, dependency_hash)
    _ensure_cuda_torch(python)
    return python


def _npm_command() -> str:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("Node.js and npm are required to build the Data Labeller frontend.")
    return npm


def _frontend_inputs() -> list[Path]:
    inputs = [
        FRONTEND_DIR / "index.html",
        FRONTEND_DIR / "package.json",
        FRONTEND_DIR / "vite.config.js",
    ]
    lockfile = FRONTEND_DIR / "package-lock.json"
    if lockfile.exists():
        inputs.append(lockfile)
    inputs.extend(path for path in (FRONTEND_DIR / "src").rglob("*") if path.is_file())
    return inputs


def _ensure_frontend() -> None:
    npm = _npm_command()
    package_inputs = [FRONTEND_DIR / "package.json"]
    lockfile = FRONTEND_DIR / "package-lock.json"
    if lockfile.exists():
        package_inputs.append(lockfile)
    dependency_hash = _fingerprint(package_inputs)

    vite_executable = FRONTEND_DIR / "node_modules" / ".bin" / ("vite.cmd" if os.name == "nt" else "vite")
    if not _stamp_matches(NODE_STAMP, dependency_hash) or not vite_executable.exists():
        print("Installing frontend dependencies...", flush=True)
        install_command = "ci" if lockfile.exists() else "install"
        _run([npm, install_command], cwd=FRONTEND_DIR)
        _write_stamp(NODE_STAMP, dependency_hash)

    build_hash = _fingerprint(_frontend_inputs())
    build_exists = (FRONTEND_DIR / "dist" / "index.html").exists() and (
        FRONTEND_DIR / "dist" / "assets"
    ).is_dir()
    if not _stamp_matches(BUILD_STAMP, build_hash) or not build_exists:
        print("Building the frontend...", flush=True)
        _run([npm, "run", "build"], cwd=FRONTEND_DIR)
        _write_stamp(BUILD_STAMP, build_hash)


def _browser_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _open_browser_when_ready(process: subprocess.Popen[bytes], url: str) -> None:
    health_url = f"{url}/api/health"
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, URLError):
            time.sleep(0.2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install anything that changed, build the frontend, and start Data Labeller."
    )
    parser.add_argument("--project-dir", default=str(Path("dataset") / "annotations"))
    parser.add_argument("--source-dir")
    parser.add_argument(
        "--provider",
        choices=("mock", "sam2", "sam3"),
        default=os.getenv("JANO_SEGMENTATION_PROVIDER", "sam2").strip().lower(),
        help="Initial-mask provider (default: SAM 2.1; mock is only for tests/UI development).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the app in the default browser.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Install dependencies and build the frontend without starting the server.",
    )
    return parser.parse_args()


def main() -> int:
    _load_local_env()
    args = _parse_args()
    try:
        python = _ensure_python_environment()
        _ensure_frontend()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1

    if args.prepare_only:
        print("Data Labeller is ready to start.")
        return 0

    command = [
        str(python),
        "-m",
        "tools.dataset_labeler",
        "start",
        "--project-dir",
        args.project_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.source_dir:
        command.extend(["--source-dir", args.source_dir])
    if args.provider:
        command.extend(["--provider", args.provider])

    url = f"http://{_browser_host(args.host)}:{args.port}"
    print(f"Starting Data Labeller at {url}", flush=True)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, cwd=ROOT)
        if not args.no_browser:
            threading.Thread(target=_open_browser_when_ready, args=(process, url), daemon=True).start()
        return process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
        return 130
    except OSError as exc:
        print(f"Could not start Data Labeller: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
