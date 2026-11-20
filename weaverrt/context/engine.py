"""Budgeted Context Engineering Engine:
  1. Large Tool Output Externalization -> Workspace Artifacts (on-demand line-range readback)
  2. Layered Memory & Grounded RAG injection with explicit [KB:doc#chunk] citations
  3. Tool-call-pair-safe Sliding Window + Deterministic Compaction under strict Token Budget
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from weaverrt.context.memory import LayeredMemoryStore
from weaverrt.domain import Message, RiskLevel
from weaverrt.kernel.context import ContextBuildResult, ContextBuilder
from weaverrt.kernel.model import estimate_messages_tokens, estimate_tokens
from weaverrt.kernel.skills import SkillRegistry
from weaverrt.kernel.tools import ToolContext, ToolSpec


CITATION_RE = re.compile(r"\[KB:[^\]\s]+\]")


def verify_grounding(output_text: str, retrieved_citations: list[str]) -> tuple[bool, list[str]]:
    """Deterministic Grounding Verifier: flags any fabricated [KB:...] citation not present in retrieved chunks."""
    cited = CITATION_RE.findall(output_text or "")
    allowed = set(retrieved_citations)
    hallucinated = [c for c in cited if c not in allowed]
    return len(hallucinated) == 0, hallucinated


def make_read_artifact_tool() -> ToolSpec:
    """Allows the agent to selectively read line ranges from externalized Workspace Artifacts."""

    def read_artifact(args: dict[str, Any], ctx: ToolContext) -> str:
        rel = str(args["artifact_path"]).lstrip("/\\")
        ws_root = Path(ctx.workspace_path).resolve()
        target = (ws_root / rel).resolve()
        if not str(target).startswith(str(ws_root)):
            raise PermissionError("Path traversal blocked outside workspace")
        if not target.is_file():
            raise FileNotFoundError(f"Artifact not found: {rel}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(args.get("start_line", 1)))
        end = min(len(lines), int(args.get("end_line", start + 49)))
        snippet = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"[Artifact {rel} lines {start}-{end}/{len(lines)}]\n{snippet}"

    return ToolSpec(
        name="read_artifact",
        version="1.0.0",
        description="Read a specific line range from an externalized workspace artifact.",
        parameters={
            "type": "object",
            "properties": {
                "artifact_path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["artifact_path"],
        },
        handler=read_artifact,
        side_effect=False,
        risk_level=RiskLevel.LOW,
        required_scopes=("read",),
    )


class BudgetedContextBuilder(ContextBuilder):
    """Production Context Engineering Engine enforcing hard token budgets."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 2000,
        max_tool_output_chars: int = 600,
        keep_recent_messages: int = 6,
        memory_store: LayeredMemoryStore | None = None,
        skill_registry: SkillRegistry | None = None,
        rag_top_k: int = 3,
        rag_token_budget: int = 400,
        user_id: str = "default",
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_tool_output_chars = max_tool_output_chars
        self.keep_recent_messages = keep_recent_messages
        self.memory_store = memory_store
        self.skill_registry = skill_registry
        self.rag_top_k = rag_top_k
        self.rag_token_budget = rag_token_budget
        self.user_id = user_id

    def _externalize_large_tool_outputs(
        self, history: list[Message], workspace_path: str
    ) -> tuple[list[Message], list[str]]:
        out: list[Message] = []
        artifacts: list[str] = []
        art_dir = Path(workspace_path) / ".artifacts"
        for idx, m in enumerate(history):
            if m.role == "tool" and len(m.content) > self.max_tool_output_chars:
                art_dir.mkdir(parents=True, exist_ok=True)
                safe_id = (m.tool_call_id or f"idx_{idx}").replace("/", "_").replace("\\", "_")
                rel_path = f".artifacts/tool_{safe_id}.txt"
                full_path = Path(workspace_path) / rel_path
                if not full_path.exists():
                    full_path.write_text(m.content, encoding="utf-8")
                lines = m.content.splitlines()
                head = "\n".join(lines[:3])[:220]
                tail = "\n".join(lines[-2:])[-160:] if len(lines) > 3 else ""
                preview = (
                    f"[EXTERNALIZED ARTIFACT: {rel_path} | {len(m.content)} chars | {len(lines)} lines]\n"
                    f"Head: {head}\n"
                    + (f"Tail: {tail}\n" if tail else "")
                    + f"Use read_artifact(artifact_path='{rel_path}', start_line=..., end_line=...) to inspect details."
                )
                out.append(
                    Message(
                        role="tool",
                        content=preview,
                        tool_call_id=m.tool_call_id,
                        tool_name=m.tool_name,
                        metadata=dict(m.metadata, externalized=rel_path),
                    )
                )
                artifacts.append(rel_path)
            else:
                out.append(m)
        return out, artifacts

    def _build_memory_and_rag_block(self, session_id: str, query: str) -> tuple[str, list[str], list[str]]:
        sections: list[str] = []
        citations: list[str] = []
        activated_skills: list[str] = []
        used_tokens = 0

        if self.skill_registry is not None and query:
            matched = self.skill_registry.match_for_query(query, max_skills=2)
            if matched:
                sblock = self.skill_registry.render_prompt_block(matched)
                sections.append(sblock)
                activated_skills = [s.skill_id for s in matched]
                used_tokens += estimate_tokens(sblock)

        if self.memory_store is None:
            return "\n\n".join(sections), citations, activated_skills

        profile = self.memory_store.get_user_profile(self.user_id)
        if profile:
            lines = [f"- {k}: {v}" for k, v in profile.items()]
            block = "### User Personalization Profile\n" + "\n".join(lines)
            sections.append(block)
            used_tokens += estimate_tokens(block)

        working = self.memory_store.get_working(session_id)
        if working:
            lines = [f"- {k}: {v}" for k, v in working.items()]
            block = "### Working Memory (Pinned Session Facts)\n" + "\n".join(lines)
            sections.append(block)
            used_tokens += estimate_tokens(block)

        episodes = self.memory_store.recent_episodes(self.user_id, limit=3)
        if episodes:
            lines = [f"- ({ep['episode_id']}) {ep['summary']}" for ep in episodes]
            block = "### Episodic Memory (Cross-Session Recall)\n" + "\n".join(lines)
            if used_tokens + estimate_tokens(block) <= self.rag_token_budget + 300:
                sections.append(block)
                used_tokens += estimate_tokens(block)

        if query:
            chunks = self.memory_store.search_kb(query, top_k=self.rag_top_k)
            kb_lines: list[str] = []
            for ch in chunks:
                entry = f"- {ch.citation} (source={ch.source}, score={ch.score}): {ch.content}"
                t = estimate_tokens(entry)
                if used_tokens + t > self.rag_token_budget + 400 and kb_lines:
                    break
                kb_lines.append(entry)
                citations.append(ch.citation)
                used_tokens += t
            if kb_lines:
                sections.append("### Grounded Knowledge Retrieval (Cite IDs when referencing)\n" + "\n".join(kb_lines))

        return "\n\n".join(sections), citations, activated_skills

    @staticmethod
    def _summarize_middle(middle: list[Message], max_chars: int = 800) -> str:
        """Deterministic, zero-LLM structured compaction preserving facts, tool outcomes & artifact handles."""
        bullets: list[str] = []
        for m in middle:
            if m.role == "user":
                bullets.append(f"User: {m.content[:120]}")
            elif m.role == "assistant":
                if m.tool_calls:
                    calls = ", ".join(f"{tc.tool_name}({list(tc.arguments.keys())})" for tc in m.tool_calls)
                    bullets.append(f"Assistant called [{calls}]")
                elif m.content:
                    bullets.append(f"Assistant: {m.content[:120]}")
            elif m.role == "tool":
                first_line = (m.content.splitlines() or [""])[0][:140]
                bullets.append(f"Tool[{m.tool_name}]: {first_line}")
        body = "\n".join(f"- {b}" for b in bullets)
        if len(body) > max_chars:
            # Keep earliest and latest bullets so initial constraints & latest progress survive
            half = max_chars // 2 - 20
            body = body[:half] + "\n... [middle turns compacted] ...\n" + body[-half:]
        return "[Compacted Earlier Turns]\n" + body

    def build(
        self,
        system_prompt: str,
        history: list[Message],
        *,
        session_id: str,
        workspace_path: str,
        query: str = "",
    ) -> ContextBuildResult:
        raw_msgs = [Message(role="system", content=system_prompt)] + list(history)
        raw_tokens = estimate_messages_tokens(raw_msgs)

        # 1. Externalize huge tool outputs into workspace artifacts
        processed, ext_artifacts = self._externalize_large_tool_outputs(list(history), workspace_path)

        # 2. Grounded Memory + Personalization + Progressive Skill Disclosure + RAG injection
        mem_block, citations, activated_skills = self._build_memory_and_rag_block(session_id, query)
        full_system = system_prompt + (f"\n\n{mem_block}" if mem_block else "")
        sys_msg = Message(role="system", content=full_system)

        candidate = [sys_msg] + processed
        compacted = False
        compacted_count = 0

        # 3. Sliding-window compaction when exceeding budget
        if estimate_messages_tokens(candidate) > self.max_context_tokens and len(processed) > 2:
            compacted = True
            first_user = processed[0]
            rest = processed[1:]
            # Keep trailing messages without splitting an assistant(tool_calls) -> tool pair
            keep_n = min(self.keep_recent_messages, len(rest))
            while keep_n > 1:
                cut = len(rest) - keep_n
                # Avoid starting recent slice with orphan tool message
                while cut > 0 and rest[cut].role == "tool":
                    cut -= 1
                middle = rest[:cut]
                tail = rest[cut:]
                if not middle:
                    break
                summary_msg = Message(
                    role="system",
                    content=self._summarize_middle(middle, max_chars=max(240, self.max_context_tokens)),
                )
                trial = [sys_msg, first_user, summary_msg] + tail
                if estimate_messages_tokens(trial) <= self.max_context_tokens:
                    candidate = trial
                    compacted_count = len(middle)
                    break
                keep_n -= 2
            else:
                # Tight budget fallback: keep first user + compact summary + last 1-2 messages
                cut = max(0, len(rest) - 2)
                while cut > 0 and rest[cut].role == "tool":
                    cut -= 1
                middle = rest[:cut]
                tail = rest[cut:]
                summary_budget_chars = max(180, (self.max_context_tokens - estimate_messages_tokens([sys_msg, first_user] + tail)) * 2)
                summary_msg = Message(role="system", content=self._summarize_middle(middle, max_chars=summary_budget_chars))
                candidate = [sys_msg, first_user, summary_msg] + tail
                compacted_count = len(middle)

        final_tokens = estimate_messages_tokens(candidate)
        saved_pct = round(max(0.0, (raw_tokens - final_tokens) / max(1, raw_tokens) * 100.0), 1)
        return ContextBuildResult(
            messages=candidate,
            estimated_tokens=final_tokens,
            compacted=compacted or bool(ext_artifacts),
            externalized_artifacts=ext_artifacts,
            retrieved_citations=citations,
            stats={
                "strategy": "budgeted_context_engine",
                "raw_tokens": raw_tokens,
                "final_tokens": final_tokens,
                "token_saved_pct": saved_pct,
                "compacted_messages": compacted_count,
                "externalized_count": len(ext_artifacts),
                "citation_count": len(citations),
                "activated_skills": activated_skills,
            },
        )
