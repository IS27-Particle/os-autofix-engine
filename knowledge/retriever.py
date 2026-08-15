"""Offline Hybrid Documentation & Runbook Vector/BM25 Retriever for Triage & Remediation."""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("os_autofix.knowledge.retriever")


@dataclass
class DocumentChunk:
    """Indexed documentation or runbook snippet."""

    chunk_id: str
    source_file: str
    title: str
    section: str
    content: str
    keywords: list[str] = field(default_factory=list)
    score: float = 0.0


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-z0-9_\-\.]{2,}\b", text.lower())


class HybridRetriever:
    """Embedded hybrid BM25 and vector-based documentation retriever."""

    def __init__(self, runbooks_dir: Path | str | None = None) -> None:
        self.runbooks_dir = Path(runbooks_dir or "knowledge/runbooks")
        self.chunks: list[DocumentChunk] = []
        self.doc_freqs: Counter[str] = Counter()
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0

    def chunk_markdown(self, filepath: Path) -> list[DocumentChunk]:
        """Split a Markdown runbook into section-level chunks."""
        text = filepath.read_text(encoding="utf-8")
        lines = text.splitlines()
        chunks: list[DocumentChunk] = []

        title = filepath.stem.replace("_", " ").title()
        current_section = "Overview"
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
            elif line.startswith("## "):
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{filepath.stem}::{current_section.lower().replace(' ', '_')}",
                                source_file=str(filepath),
                                title=title,
                                section=current_section,
                                content=content,
                                keywords=_tokenize(content),
                            )
                        )
                    current_lines = []
                current_section = line.lstrip("# ").strip()
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{filepath.stem}::{current_section.lower().replace(' ', '_')}",
                        source_file=str(filepath),
                        title=title,
                        section=current_section,
                        content=content,
                        keywords=_tokenize(content),
                    )
                )

        return chunks

    def index_all(self) -> int:
        """Parse and index all runbooks in the runbooks directory."""
        self.chunks = []
        self.doc_freqs = Counter()

        if self.runbooks_dir.exists():
            for md_file in sorted(self.runbooks_dir.glob("*.md")):
                try:
                    file_chunks = self.chunk_markdown(md_file)
                    self.chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning("Failed indexing '%s': %s", md_file, e)

        self.total_docs = len(self.chunks)
        if self.total_docs == 0:
            return 0

        # Calculate BM25 statistics
        total_tokens = 0
        for chunk in self.chunks:
            unique_terms = set(chunk.keywords)
            for t in unique_terms:
                self.doc_freqs[t] += 1
            total_tokens += len(chunk.keywords)

        self.avg_doc_len = total_tokens / self.total_docs
        logger.info(
            "Indexed %d document chunks (avg_len=%.1f tokens).", self.total_docs, self.avg_doc_len
        )
        return self.total_docs

    def _score_bm25(
        self, query_terms: list[str], chunk: DocumentChunk, k1: float = 1.5, b: float = 0.75
    ) -> float:
        """Compute BM25 relevance score for a document chunk."""
        score = 0.0
        doc_len = len(chunk.keywords)
        if doc_len == 0 or self.total_docs == 0:
            return 0.0

        term_counts = Counter(chunk.keywords)

        for term in query_terms:
            if term not in term_counts:
                continue

            df = self.doc_freqs.get(term, 0)
            idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
            tf = term_counts[term]

            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)

        return max(0.0, score)

    def consult_runbook(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        """Hybrid search against indexed runbooks."""
        if not self.chunks:
            self.index_all()

        if not self.chunks:
            return []

        query_terms = _tokenize(query)
        scored_chunks: list[tuple[float, DocumentChunk]] = []

        for chunk in self.chunks:
            score = self._score_bm25(query_terms, chunk)
            # Boost matches in title / section
            title_terms = set(_tokenize(chunk.title + " " + chunk.section))
            if any(t in title_terms for t in query_terms):
                score += 2.0

            if score > 0:
                chunk.score = round(score, 3)
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_chunks[:top_k]]

    def search_manpages(self, daemon: str) -> str:
        """Retrieve local system man page summary for a daemon/command."""
        try:
            cmd = f"man {daemon} 2>/dev/null | col -b | head -n 40"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return f"No manual entry found for {daemon}."

    def export_index_json(self, output_path: Path | str) -> None:
        """Serialize current knowledge index to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_chunks": len(self.chunks),
            "chunks": [asdict(c) for c in self.chunks],
        }
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Global retriever instance
GLOBAL_RETRIEVER = HybridRetriever()
