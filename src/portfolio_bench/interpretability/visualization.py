"""Visualization utilities for LLM interpretability analysis."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analyzer import WindowAnalysis


def importance_color(normalized_value: float) -> str:
    """Map normalized importance [0, 1] to a color style.

    Color gradient: blue -> yellow -> red

    Args:
        normalized_value: Importance value normalized to [0, 1].

    Returns:
        Rich style string for the color.
    """
    if normalized_value < 0.33:
        return "blue"
    elif normalized_value < 0.66:
        return "yellow"
    else:
        return "red"


def render_colored_prompt(
    analysis: WindowAnalysis,
    policy_name: str,
    precision: int = 4,
) -> Panel:
    """Render state matrix with per-cell colors based on importance.

    Args:
        analysis: WindowAnalysis containing state and importance matrices.
        policy_name: Name of the policy for the title.
        precision: Decimal precision for formatting values.

    Returns:
        Rich Panel with colored matrix visualization.
    """
    state = analysis.state
    importance = analysis.importance
    weights = analysis.original_weights
    L, N = state.shape

    # Normalize importance for coloring
    max_imp = importance.max()
    norm = importance / max_imp if max_imp > 0 else importance

    text = Text()

    # Header with asset indices
    text.append("       ", style="dim")  # Padding for row labels
    for j in range(N):
        text.append(f"Asset {j:>2}  ", style="bold cyan")
    text.append("\n")

    # Matrix rows with time labels
    for i in range(L):
        # Row label (t-L, t-L+1, ..., t-1)
        t_label = f"t-{L-i:>2}"
        text.append(f"{t_label} [", style="dim")

        for j in range(N):
            val = state[i, j]
            color = importance_color(norm[i, j])
            text.append(f"{val:>8.{precision}f}", style=color)
            if j < N - 1:
                text.append(", ")
        text.append("]\n")

    # Add weights output
    text.append("\n")
    text.append("Weights: ", style="bold")
    weights_str = json.dumps({"weights": [round(w, 4) for w in weights.tolist()]})
    text.append(weights_str, style="green")

    # Create legend
    legend = Text()
    legend.append("Importance: ", style="bold")
    legend.append("blue", style="blue")
    legend.append(" (low) -> ")
    legend.append("yellow", style="yellow")
    legend.append(" -> ")
    legend.append("red", style="red")
    legend.append(" (high)")

    return Panel(
        text,
        title=f"[bold]{policy_name} - Window {analysis.window_idx}[/bold]",
        subtitle=legend,
        border_style="blue",
    )


def render_summary_panel(stats: dict, policy_name: str) -> Panel:
    """Render aggregate statistics as a summary panel.

    Args:
        stats: Dictionary from compute_aggregate_stats().
        policy_name: Name of the policy.

    Returns:
        Rich Panel with summary statistics.
    """
    text = Text()

    text.append("Recency Analysis\n", style="bold underline")
    text.append(f"  Correlation (row index vs importance): ")
    corr = stats.get("recency_correlation", 0)
    corr_style = "green" if corr > 0.5 else "yellow" if corr > 0 else "red"
    text.append(f"{corr:.3f}\n", style=corr_style)

    text.append(f"  Recency ratio (newest/oldest): ")
    ratio = stats.get("recency_ratio", 1)
    ratio_style = "green" if ratio > 1.5 else "yellow" if ratio > 1 else "dim"
    if ratio == float("inf"):
        text.append("inf\n", style=ratio_style)
    else:
        text.append(f"{ratio:.2f}x\n", style=ratio_style)

    text.append(f"  Mean importance: {stats.get('total_mean_importance', 0):.4f}\n")
    text.append(f"  Windows analyzed: {stats.get('n_windows', 0)}\n")

    # Importance by row (time step)
    mean_by_row = stats.get("mean_importance_by_row", [])
    if mean_by_row:
        text.append("\nImportance by Time Step\n", style="bold underline")
        L = len(mean_by_row)
        max_row_imp = max(mean_by_row) if mean_by_row else 1
        for i, imp in enumerate(mean_by_row):
            t_label = f"t-{L-i:>2}"
            bar_len = int(20 * imp / max_row_imp) if max_row_imp > 0 else 0
            bar = "█" * bar_len
            color = importance_color(imp / max_row_imp if max_row_imp > 0 else 0)
            text.append(f"  {t_label}: ")
            text.append(f"{bar:<20}", style=color)
            text.append(f" {imp:.4f}\n")

    # Importance by asset
    mean_by_col = stats.get("mean_importance_by_col", [])
    if mean_by_col:
        text.append("\nImportance by Asset\n", style="bold underline")
        max_col_imp = max(mean_by_col) if mean_by_col else 1
        for j, imp in enumerate(mean_by_col):
            bar_len = int(20 * imp / max_col_imp) if max_col_imp > 0 else 0
            bar = "█" * bar_len
            color = importance_color(imp / max_col_imp if max_col_imp > 0 else 0)
            text.append(f"  Asset {j}: ")
            text.append(f"{bar:<20}", style=color)
            text.append(f" {imp:.4f}\n")

    return Panel(
        text,
        title=f"[bold]{policy_name} - Summary Statistics[/bold]",
        border_style="green",
    )


def export_html(
    analyses: List[WindowAnalysis],
    stats: dict,
    policy_name: str,
    output_path: Path,
) -> None:
    """Export visualizations as an HTML file.

    Args:
        analyses: List of WindowAnalysis results.
        stats: Aggregate statistics dictionary.
        policy_name: Name of the policy.
        output_path: Path to write HTML file.
    """
    console = Console(record=True, force_terminal=True, width=120)

    # Render summary
    console.print(render_summary_panel(stats, policy_name))
    console.print()

    # Render individual windows
    for analysis in analyses:
        console.print(render_colored_prompt(analysis, policy_name))
        console.print()

    # Save HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.save_html(str(output_path), theme=None)
