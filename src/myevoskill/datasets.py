"""Dataset split manifests and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class FamilySplit:
    """Fixed split manifest for a task family."""

    family: str
    distill_train: List[str]
    transfer_val: List[str]
    final_test: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        overlap = (
            set(self.distill_train) & set(self.transfer_val)
            | set(self.distill_train) & set(self.final_test)
            | set(self.transfer_val) & set(self.final_test)
        )
        if overlap:
            raise ValueError(
                f"family '{self.family}' has overlapping split entries: {sorted(overlap)}"
            )


@dataclass
class DatasetManifest:
    """Collection of family split manifests with leakage-safe accessors."""

    families: Dict[str, FamilySplit]

    @classmethod
    def from_splits(cls, splits: Iterable[FamilySplit]) -> "DatasetManifest":
        mapping: Dict[str, FamilySplit] = {}
        for split in splits:
            if split.family in mapping:
                raise ValueError(f"duplicate family split: {split.family}")
            mapping[split.family] = split
        return cls(mapping)

    def get_family(self, family: str) -> FamilySplit:
        try:
            return self.families[family]
        except KeyError as exc:
            raise KeyError(f"unknown family split: {family}") from exc

    def tasks_for_split(self, family: str, split_name: str) -> List[str]:
        split = self.get_family(family)
        if split_name == "distill_train":
            return list(split.distill_train)
        if split_name == "transfer_val":
            return list(split.transfer_val)
        if split_name == "final_test":
            return list(split.final_test)
        raise ValueError(f"unknown split name: {split_name}")

    def assert_no_validation_leakage(
        self, family: str, proposed_task_ids: Iterable[str]
    ) -> None:
        split = self.get_family(family)
        leaked = set(proposed_task_ids) & set(split.transfer_val)
        if leaked:
            raise ValueError(
                f"validation leakage detected for family '{family}': {sorted(leaked)}"
            )

