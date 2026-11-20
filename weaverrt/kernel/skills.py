"""Skill Registry & Progressive Disclosure Engine.

Models reusable Agent Skills (versioned instruction bundles + trigger keywords + scoped tool allowlists),
so the runtime dynamically injects ONLY the relevant Skill instructions and minimum required tools per task.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class SkillSpec:
    name: str
    version: str
    description: str
    triggers: tuple[str, ...]
    instructions: str
    allowed_tools: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ("read",)

    @property
    def skill_id(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["skill_id"] = self.skill_id
        return d

    @classmethod
    def from_markdown(cls, md_text: str) -> "SkillSpec":
        """Parse standard SKILL.md format with YAML frontmatter + markdown instruction body."""
        text = md_text.strip()
        if not text.startswith("---"):
            raise ValueError("SKILL.md must begin with YAML frontmatter '---'")
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Malformed SKILL.md frontmatter")
        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        return cls(
            name=str(meta["name"]),
            version=str(meta.get("version", "1.0.0")),
            description=str(meta.get("description", "")),
            triggers=tuple(str(t).lower() for t in (meta.get("triggers") or [])),
            instructions=body,
            allowed_tools=tuple(str(t) for t in (meta.get("allowed_tools") or [])),
            required_scopes=tuple(str(s) for s in (meta.get("required_scopes") or ["read"])),
        )


class SkillRegistry:
    """Versioned Skill catalog supporting dynamic matching and progressive disclosure."""

    def __init__(self) -> None:
        self._skills: dict[str, dict[str, SkillSpec]] = {}
        self._latest: dict[str, str] = {}

    def register(self, spec: SkillSpec) -> SkillSpec:
        self._skills.setdefault(spec.name, {})[spec.version] = spec
        self._latest[spec.name] = spec.version
        return spec

    def load_file(self, path: str | Path) -> SkillSpec:
        return self.register(SkillSpec.from_markdown(Path(path).read_text(encoding="utf-8")))

    def get(self, name: str, version: str | None = None) -> SkillSpec | None:
        vs = self._skills.get(name)
        if not vs:
            return None
        v = version if (version and version in vs) else self._latest[name]
        return vs[v]

    def list_skills(self) -> list[SkillSpec]:
        return [self._skills[n][v] for n, v in sorted(self._latest.items())]

    def match_for_query(self, query: str, max_skills: int = 2) -> list[SkillSpec]:
        q = query.lower()
        scored: list[tuple[int, SkillSpec]] = []
        for spec in self.list_skills():
            hits = sum(1 for kw in spec.triggers if kw and kw in q)
            if hits > 0:
                scored.append((hits, spec))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:max_skills]]

    def render_prompt_block(self, matched: list[SkillSpec]) -> str:
        if not matched:
            return ""
        blocks = [f"#### Skill: {s.skill_id} ({s.description})\n{s.instructions}" for s in matched]
        return "### Activated Skills (Progressive Disclosure)\n" + "\n\n".join(blocks)
