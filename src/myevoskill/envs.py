"""Environment hashing and cache management."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from .models import EnvCacheRecord


@dataclass(frozen=True)
class EnvSpec:
    """Stable environment specification used for cache reuse."""

    python_version: str
    requirements: List[str]
    system_packages: List[str] = field(default_factory=list)
    task_family: str = ""
    compute_profile: str = "mixed"
    cuda: str = ""
    container_image: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


class EnvManager:
    """Manage reusable task environments and per-run workspace resets."""

    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.base_image_cache_root = self.cache_root / "base_images"
        self.task_env_cache_root = self.cache_root / "task_envs"
        self.dataset_cache_root = self.cache_root / "datasets"
        self.artifact_cache_root = self.cache_root / "artifacts"
        self.checkpoint_cache_root = self.cache_root / "checkpoints"
        for path in (
            self.base_image_cache_root,
            self.task_env_cache_root,
            self.dataset_cache_root,
            self.artifact_cache_root,
            self.checkpoint_cache_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def compute_env_hash(self, spec: EnvSpec) -> str:
        payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def ensure_env(self, spec: EnvSpec) -> EnvCacheRecord:
        env_hash = self.compute_env_hash(spec)
        base_dir = self.base_image_cache_root / (
            spec.container_image.replace("/", "_").replace(":", "_") or "default"
        )
        env_dir = self.task_env_cache_root / env_hash
        dataset_dir = self.dataset_cache_root / env_hash
        artifact_dir = self.artifact_cache_root / env_hash
        checkpoint_dir = self.checkpoint_cache_root / env_hash
        for path in (base_dir, env_dir, dataset_dir, artifact_dir, checkpoint_dir):
            path.mkdir(parents=True, exist_ok=True)
        env_file = env_dir / "env_spec.json"
        if not env_file.exists():
            env_file.write_text(
                json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8"
            )
        return EnvCacheRecord(
            env_hash=env_hash,
            env_dir=env_dir,
            base_image_cache_dir=base_dir,
            task_env_cache_dir=env_dir,
            dataset_cache_dir=dataset_dir,
            artifact_cache_dir=artifact_dir,
            checkpoint_cache_dir=checkpoint_dir,
        )

    def reset_run_workspace(self, run_root: Path) -> None:
        run_root = Path(run_root)
        for name in ("work", "output"):
            target = run_root / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    def stage_checkpoint(
        self, cache_record: EnvCacheRecord, checkpoint_name: str, content: str
    ) -> Path:
        path = cache_record.checkpoint_cache_dir / checkpoint_name
        path.write_text(content, encoding="utf-8")
        return path

    def restore_checkpoint(
        self, cache_record: EnvCacheRecord, run_root: Path, checkpoint_name: str
    ) -> Path:
        source = cache_record.checkpoint_cache_dir / checkpoint_name
        if not source.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_name}")
        target_dir = Path(run_root) / "checkpoints"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / checkpoint_name
        shutil.copy2(source, target)
        return target
