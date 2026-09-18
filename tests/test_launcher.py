from __future__ import annotations

from pathlib import Path

import run as launcher


def test_cuda_torch_install_is_skipped_when_cpu_is_requested(monkeypatch) -> None:
    monkeypatch.setenv("GROUNDING_DINO_DEVICE", "cpu")
    monkeypatch.setattr(
        launcher,
        "_torch_can_use_cuda",
        lambda _python: (_ for _ in ()).throw(AssertionError("CUDA probe should not run")),
    )

    launcher._ensure_cuda_torch(Path("python"))


def test_cuda_torch_is_installed_for_an_nvidia_gpu(monkeypatch) -> None:
    probes = iter((False, True))
    commands: list[list[str]] = []
    monkeypatch.delenv("GROUNDING_DINO_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_INDEX_URL", raising=False)
    monkeypatch.setattr(launcher, "_torch_can_use_cuda", lambda _python: next(probes))
    monkeypatch.setattr(launcher, "_nvidia_gpu_available", lambda: True)
    monkeypatch.setattr(launcher, "_run", lambda command: commands.append(command))

    launcher._ensure_cuda_torch(Path("python"))

    assert commands == [
        [
            "python",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "torch",
            "torchvision",
            "--index-url",
            launcher.PYTORCH_CUDA_INDEX_URL,
        ]
    ]


def test_cuda_torch_install_is_skipped_without_an_nvidia_gpu(monkeypatch) -> None:
    monkeypatch.delenv("GROUNDING_DINO_DEVICE", raising=False)
    monkeypatch.setattr(launcher, "_torch_can_use_cuda", lambda _python: False)
    monkeypatch.setattr(launcher, "_nvidia_gpu_available", lambda: False)
    monkeypatch.setattr(
        launcher,
        "_run",
        lambda _command: (_ for _ in ()).throw(AssertionError("pip should not run")),
    )

    launcher._ensure_cuda_torch(Path("python"))
