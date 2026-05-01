"""Token-level ablation analysis for LLM prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..methods.llm.prompts import parse_weights_from_response
from ..methods.base import project_to_simplex


@dataclass
class TokenAnalysis:
    """Results of analyzing token-level importance for a prompt."""

    tokens: List[str]
    importance: np.ndarray  # (T,) importance scores
    original_weights: np.ndarray  # (N,) original output
    response: Optional[str] = None  # Raw LLM response if available


_TOKEN_RE = re.compile(r"\s+|[^\s]+")


def tokenize_prompt(prompt: str) -> List[str]:
    """Tokenize a prompt into whitespace and non-whitespace chunks."""
    return _TOKEN_RE.findall(prompt)


def _token_spans(tokens: List[str]) -> List[tuple[int, int]]:
    """Compute (start, end) spans for tokens based on concatenation."""
    spans = []
    pos = 0
    for tok in tokens:
        start = pos
        pos += len(tok)
        spans.append((start, pos))
    return spans


def _generate_weights(client, prompt: str, n_assets: int, temperature: float) -> np.ndarray:
    """Generate weights from a prompt using the LLM client."""
    response = client.generate(prompt=prompt, temperature=temperature, n_assets=n_assets)
    weights, _ = parse_weights_from_response(response, n_assets)
    return project_to_simplex(np.array(weights))


def analyze_prompt_tokens(
    client,
    prompt: str,
    n_assets: int,
    temperature: float = 0.0,
    ablation_token: str = "",
    progress_callback=None,
) -> TokenAnalysis:
    """Compute token-level importance scores via ablation.

    For each non-whitespace token in the prompt:
    1. Replace the token with ablation_token
    2. Generate new weights
    3. Compute L1 distance from original weights

    Args:
        client: LLM client with .generate(...) method.
        prompt: Full prompt text.
        n_assets: Number of assets (for schema enforcement).
        temperature: LLM temperature.
        ablation_token: Replacement string for ablated tokens.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        TokenAnalysis with tokens, importance scores, and original weights.
    """
    tokens = tokenize_prompt(prompt)
    original_weights = _generate_weights(client, prompt, n_assets, temperature)
    importance = np.zeros(len(tokens), dtype=float)

    # Only ablate non-whitespace tokens
    ablatable_indices = [i for i, tok in enumerate(tokens) if not tok.isspace()]
    total = len(ablatable_indices)

    for idx, token_idx in enumerate(ablatable_indices, start=1):
        if progress_callback:
            progress_callback(idx, total)
        modified_tokens = tokens.copy()
        modified_tokens[token_idx] = ablation_token
        modified_prompt = "".join(modified_tokens)
        new_weights = _generate_weights(client, modified_prompt, n_assets, temperature)
        importance[token_idx] = np.abs(new_weights - original_weights).sum()

    return TokenAnalysis(
        tokens=tokens,
        importance=importance,
        original_weights=original_weights,
    )


def _latex_escape(text: str) -> str:
    """Escape text for LaTeX (plain text only)."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _color_for_norm(norm: float) -> str:
    """Map normalized importance [0,1] to xcolor gradient."""
    if norm <= 0.5:
        p = int(round(norm * 200))
        return f"blue!{p}!yellow"
    p = int(round((norm - 0.5) * 200))
    return f"yellow!{p}!red"


def _render_tokens(
    tokens: List[str],
    norm_importance: np.ndarray,
    start_idx: int,
    end_idx: int,
    compact: bool = True,
) -> str:
    """Render a token slice with background highlights."""
    parts: List[str] = []
    last_space = False
    for i in range(start_idx, end_idx):
        tok = tokens[i]
        if tok.isspace():
            if "\n" in tok:
                if not last_space:
                    parts.append("\n")
                last_space = True
            else:
                if not last_space:
                    parts.append(" ")
                last_space = True
            continue
        color = _color_for_norm(float(norm_importance[i]))
        escaped = _latex_escape(tok)
        parts.append(rf"\colorbox{{{color}}}{{\strut {escaped}}}")
        last_space = False
    return "".join(parts)


def _extract_number_importances(
    tokens: List[str],
    norm_importance: np.ndarray,
    spans: List[tuple[int, int]],
    data_span: tuple[int, int],
) -> List[float]:
    """Extract importance scores for numeric tokens inside data span."""
    number_imps: List[float] = []
    number_re = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    for i, (start, end) in enumerate(spans):
        if end <= data_span[0] or start >= data_span[1]:
            continue
        tok = tokens[i]
        if tok.isspace():
            continue
        if number_re.search(tok):
            number_imps.append(float(norm_importance[i]))
    return number_imps


def render_prompt_latex(
    prompt: str,
    tokens: List[str],
    importance: np.ndarray,
    data_str: str | None = None,
    matrix: list | None = None,
    compact: bool = True,
    dense_data: bool = True,
    arraystretch: float = 1.2,
    tabcolsep_pt: int = 2,
) -> str:
    """Render a prompt as LaTeX with token-level background highlights.

    Args:
        tokens: Token list from tokenize_prompt.
        importance: Importance scores aligned to tokens.
        compact: If True, collapse whitespace/newlines into single spaces.
    """
    max_imp = float(np.max(importance)) if len(importance) else 0.0
    norm = importance / max_imp if max_imp > 0 else importance

    data_span = None
    if data_str:
        start = prompt.find(data_str)
        if start != -1:
            data_span = (start, start + len(data_str))

    spans = _token_spans(tokens)

    if data_span is None or matrix is None:
        # Fallback: render all tokens inline
        content = _render_tokens(tokens, norm, 0, len(tokens), compact=compact)
        return (
            "\\begin{flushleft}\n"
            "{\\ttfamily\\small\\obeyspaces\\obeylines\n"
            + content
            + "\n}\n\\end{flushleft}"
        )

    # Render prefix and suffix tokens around data block
    prefix_end = 0
    while prefix_end < len(spans) and spans[prefix_end][1] <= data_span[0]:
        prefix_end += 1
    suffix_start = prefix_end
    while suffix_start < len(spans) and spans[suffix_start][0] < data_span[1]:
        suffix_start += 1

    prefix = _render_tokens(tokens, norm, 0, prefix_end, compact=compact)
    suffix = _render_tokens(tokens, norm, suffix_start, len(tokens), compact=compact)

    # Build matrix table from extracted numeric token importances
    number_imps = _extract_number_importances(tokens, norm, spans, data_span)
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    total_needed = rows * cols
    if len(number_imps) < total_needed:
        number_imps = number_imps + [0.0] * (total_needed - len(number_imps))
    if len(number_imps) > total_needed:
        number_imps = number_imps[:total_needed]

    # Build tabular with colored cells
    lines: List[str] = []
    lines.append("\\par\\smallskip")
    lines.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep_pt}pt}}")
    lines.append("\\begin{tabular}{" + "r" * cols + "}")
    idx = 0
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            val = matrix[r][c]
            imp = number_imps[idx]
            idx += 1
            color = _color_for_norm(imp)
            row_cells.append(rf"\\colorbox{{{color}}}{{\\strut {val:.4f}}}")
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\end{tabular}")
    lines.append("\\par\\smallskip")
    table_block = "\n".join(lines)

    content = prefix + table_block + suffix

    return (
        "\\begin{flushleft}\n"
        "{\\ttfamily\\small\\obeyspaces\\obeylines\n"
        + content
        + "\n}\n\\end{flushleft}"
    )
