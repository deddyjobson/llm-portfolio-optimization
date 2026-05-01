"""CLI interface for portfolio benchmark."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .backtest import Backtester, RollingEvaluator, bootstrap_metrics
from .config import load_config
from .data import download_olps_data, load_dataset, make_baby_dataset
from .data.load import split_dataset
from .methods.llm import OllamaClient
from .methods.llm.ollama_client import FakeOllamaClient
from .methods.llm.policies import (
    ChainOfThoughtPolicy,
    DirectPolicy,
    FewShotPolicy,
    create_llm_policies,
)
from .methods.operations_research import (
    CVaRPolicy,
    HRPPolicy,
    MeanVariancePolicy,
    WassersteinDROPolicy,
)

app = typer.Typer(
    name="portfolio-bench",
    help="FAIR benchmark comparing OR and LLM methods for portfolio allocation",
)
console = Console()


@app.command()
def download_data(
    dataset: str = typer.Option("djia", help="Dataset name (djia, msci, sp500, tse, nyse, nyse-n, nyse-o)"),
    output_dir: str = typer.Option("data", help="Output directory"),
):
    """Download and process OLPS portfolio dataset."""
    console.print(f"[bold]Downloading {dataset} dataset...[/bold]")
    try:
        output_path = download_olps_data(dataset, output_dir)
        console.print(f"[green]Dataset saved to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def make_baby(
    input_path: str = typer.Option(
        "data/processed/djia_full.npz", help="Input dataset path"
    ),
    output_path: str = typer.Option(
        "data/processed/djia_baby.npz", help="Output dataset path"
    ),
    periods: int = typer.Option(220, help="Number of time periods"),
    assets: int = typer.Option(6, help="Number of assets"),
):
    """Create baby dataset for quick testing."""
    console.print(f"[bold]Creating baby dataset from {input_path}...[/bold]")
    try:
        out = make_baby_dataset(input_path, output_path, periods, assets)
        console.print(f"[green]Baby dataset saved to: {out}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def fetch_references(
    output_dir: str = typer.Option("references", help="Output directory for references"),
):
    """Download reference papers and repositories."""
    from .utils.fetch_references import fetch_all_references

    console.print("[bold]Fetching reference materials...[/bold]")
    try:
        fetch_all_references(output_dir)
        console.print(f"[green]References saved to: {output_dir}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    config: Annotated[str, typer.Option(help="Path to config file")] = "configs/baby.toml",
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    skip_llm: Annotated[bool, typer.Option(help="Skip LLM methods")] = False,
    skip_dro: Annotated[bool, typer.Option(help="Skip Wasserstein DRO")] = False,
    use_fake_llm: Annotated[bool, typer.Option(help="Use fake LLM client")] = False,
    bootstrap: Annotated[bool, typer.Option(help="Compute bootstrap CIs")] = False,
):
    """Run benchmark on all methods and generate results."""
    console.print(f"[bold]Loading config from {config}...[/bold]")
    cfg = load_config(config)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/runs/{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[bold]Loading dataset from {cfg.dataset.path}...[/bold]")
    try:
        price_relatives, log_relatives = load_dataset(cfg.dataset.path)
    except FileNotFoundError:
        console.print(f"[red]Dataset not found at {cfg.dataset.path}[/red]")
        console.print("Run 'portfolio-bench download-data' first")
        raise typer.Exit(1)

    T, N = price_relatives.shape
    console.print(f"Dataset shape: T={T}, N={N}")

    # Split data
    train_pr, val_pr, test_pr = split_dataset(
        price_relatives,
        cfg.backtest.train_ratio,
        cfg.backtest.val_ratio,
        cfg.backtest.test_ratio,
    )
    train_lr, val_lr, test_lr = split_dataset(
        log_relatives,
        cfg.backtest.train_ratio,
        cfg.backtest.val_ratio,
        cfg.backtest.test_ratio,
    )

    console.print(f"Train: {len(train_pr)}, Val: {len(val_pr)}, Test: {len(test_pr)}")

    # Initialize policies
    policies = []

    # OR methods
    console.print("[bold]Initializing OR methods...[/bold]")
    policies.extend([
        MeanVariancePolicy(),
        CVaRPolicy(),
        HRPPolicy(),
    ])

    # Wasserstein DRO
    if not skip_dro and cfg.dro.enabled:
        console.print("[bold]Initializing Wasserstein DRO...[/bold]")
        dro_policy = WassersteinDROPolicy(
            eta=cfg.dro.eta,
            epsilon=cfg.dro.epsilon,
            rho=cfg.dro.rho,
            support_radius=cfg.dro.support_radius,
            solver_method=cfg.dro.solver_method,
            solver=cfg.dro.solver,
        )
        policies.append(dro_policy)
    elif skip_dro:
        console.print("[yellow]Skipping Wasserstein DRO[/yellow]")

    # LLM methods
    if not skip_llm:
        console.print("[bold]Initializing LLM methods...[/bold]")
        # Check if Ollama is available
        if use_fake_llm:
            console.print("[yellow]Using fake LLM client[/yellow]")
            llm_policies = create_llm_policies(
                base_url=cfg.llm.base_url,
                model=cfg.llm.model,
                temperature=cfg.llm.temperature,
                use_fake=True,
                multiline=cfg.llm.multiline,
            )
        else:
            client = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
            if client.is_available():
                llm_policies = create_llm_policies(
                    base_url=cfg.llm.base_url,
                    model=cfg.llm.model,
                    temperature=cfg.llm.temperature,
                    use_fake=False,
                    multiline=cfg.llm.multiline,
                )
            else:
                console.print("[yellow]Ollama not available, using fake client[/yellow]")
                llm_policies = create_llm_policies(
                    base_url=cfg.llm.base_url,
                    model=cfg.llm.model,
                    temperature=cfg.llm.temperature,
                    use_fake=True,
                    multiline=cfg.llm.multiline,
                )
        policies.extend(llm_policies)
    else:
        console.print("[yellow]Skipping LLM methods[/yellow]")

    # Run backtest on test set
    console.print("[bold]Running backtest on test set...[/bold]")
    backtester = Backtester(
        price_relatives=test_pr,
        log_relatives=test_lr,
        lookback=cfg.dataset.lookback,
        transaction_cost=cfg.backtest.transaction_cost,
    )

    results = backtester.run_all(policies)

    # Compute bootstrap CIs if enabled
    bootstrap_enabled = bootstrap or cfg.bootstrap.enabled
    metrics_with_ci = {}
    if bootstrap_enabled:
        console.print("[bold]Computing bootstrap confidence intervals...[/bold]")
        for result in results:
            metrics_ci = bootstrap_metrics(
                result.equity_curve,
                result.weights_history,
                n_bootstrap=cfg.bootstrap.n_bootstrap,
                block_size=cfg.bootstrap.block_size,
                confidence_level=cfg.bootstrap.confidence_level,
                seed=cfg.bootstrap.seed,
            )
            metrics_with_ci[result.policy_name] = metrics_ci

    # Display results table
    table = Table(title="Portfolio Benchmark Results")
    table.add_column("Method", style="cyan")
    table.add_column("Total Return (%)", justify="right")
    table.add_column("Ann. Return (%)", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Max DD (%)", justify="right")
    table.add_column("Sortino", justify="right")
    table.add_column("Calmar", justify="right")
    table.add_column("Avg Turnover (%)", justify="right")

    def format_with_ci(value: float, ci: tuple, fmt: str = ".2f") -> str:
        """Format a value with its confidence interval."""
        return f"{value:{fmt}} [{ci[0]:{fmt}}, {ci[1]:{fmt}}]"

    metrics_list = []
    for result in results:
        m = result.metrics
        if bootstrap_enabled:
            ci = metrics_with_ci[result.policy_name]
            table.add_row(
                result.policy_name,
                format_with_ci(m.total_return, ci.total_return_ci),
                format_with_ci(m.annualized_return, ci.annualized_return_ci),
                format_with_ci(m.sharpe_ratio, ci.sharpe_ratio_ci, ".3f"),
                format_with_ci(m.max_drawdown, ci.max_drawdown_ci),
                format_with_ci(m.sortino_ratio, ci.sortino_ratio_ci, ".3f"),
                format_with_ci(m.calmar_ratio, ci.calmar_ratio_ci, ".3f"),
                format_with_ci(m.avg_turnover, ci.avg_turnover_ci),
            )
            entry = {
                "method": result.policy_name,
                **ci.to_dict(),
            }
        else:
            table.add_row(
                result.policy_name,
                f"{m.total_return:.2f}",
                f"{m.annualized_return:.2f}",
                f"{m.sharpe_ratio:.3f}",
                f"{m.max_drawdown:.2f}",
                f"{m.sortino_ratio:.3f}",
                f"{m.calmar_ratio:.3f}",
                f"{m.avg_turnover:.2f}",
            )
            entry = {
                "method": result.policy_name,
                **m.to_dict(),
            }
        if result.parse_error_rate is not None:
            entry["parse_error_rate"] = result.parse_error_rate
        metrics_list.append(entry)

    console.print(table)

    # Save results
    # Metrics JSON
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)

    # Summary CSV
    summary_df = pd.DataFrame(metrics_list)
    summary_df.to_csv(output_path / "summary_table.csv", index=False)

    # Equity curves
    equity_dir = output_path / "equity_curves"
    equity_dir.mkdir(exist_ok=True)
    for result in results:
        equity_df = pd.DataFrame({
            "step": range(len(result.equity_curve)),
            "equity": result.equity_curve,
        })
        equity_df.to_csv(equity_dir / f"{result.policy_name}.csv", index=False)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")


@app.command()
def rolling_eval(
    config: Annotated[str, typer.Option(help="Path to config file")] = "configs/baby.toml",
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    train_size: Annotated[int, typer.Option(help="Training window size")] = 100,
    val_size: Annotated[int, typer.Option(help="Validation window size")] = 30,
    test_size: Annotated[int, typer.Option(help="Test window size")] = 50,
    skip_llm: Annotated[bool, typer.Option(help="Skip LLM methods")] = False,
    skip_or: Annotated[bool, typer.Option(help="Skip OR methods")] = False,
    skip_dro: Annotated[bool, typer.Option(help="Skip Wasserstein DRO")] = False,
    use_fake_llm: Annotated[bool, typer.Option(help="Use fake LLM client")] = False,
    llm_model: Annotated[Optional[str], typer.Option(help="Override LLM model (e.g., qwen2.5:7b)")] = None,
):
    """Run rolling window evaluation across multiple test periods."""
    console.print(f"[bold]Loading config from {config}...[/bold]")
    cfg = load_config(config)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/rolling/{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[bold]Loading dataset from {cfg.dataset.path}...[/bold]")
    try:
        price_relatives, log_relatives = load_dataset(cfg.dataset.path)
    except FileNotFoundError:
        console.print(f"[red]Dataset not found at {cfg.dataset.path}[/red]")
        console.print("Run 'portfolio-bench download-data' first")
        raise typer.Exit(1)

    T, N = price_relatives.shape
    console.print(f"Dataset shape: T={T}, N={N}")

    # Initialize rolling evaluator
    evaluator = RollingEvaluator(
        price_relatives=price_relatives,
        log_relatives=log_relatives,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        lookback=cfg.dataset.lookback,
        transaction_cost=cfg.backtest.transaction_cost,
    )

    # Generate windows and report
    windows = evaluator.generate_windows()
    n_windows = len(windows)
    console.print(f"[bold]Generated {n_windows} evaluation windows[/bold]")
    for w in windows:
        console.print(
            f"  Window {w.window_idx}: "
            f"Train [{w.train_start}:{w.train_end}], "
            f"Val [{w.val_start}:{w.val_end}], "
            f"Test [{w.test_start}:{w.test_end}]"
        )

    # Initialize policies
    policies = []

    # OR methods
    if not skip_or:
        console.print("[bold]Initializing OR methods...[/bold]")
        policies.extend([
            MeanVariancePolicy(),
            CVaRPolicy(),
            HRPPolicy(),
        ])
    else:
        console.print("[yellow]Skipping OR methods[/yellow]")

    # Wasserstein DRO
    if not skip_dro and cfg.dro.enabled:
        console.print("[bold]Initializing Wasserstein DRO...[/bold]")
        dro_policy = WassersteinDROPolicy(
            eta=cfg.dro.eta,
            epsilon=cfg.dro.epsilon,
            rho=cfg.dro.rho,
            support_radius=cfg.dro.support_radius,
            solver_method=cfg.dro.solver_method,
            solver=cfg.dro.solver,
        )
        policies.append(dro_policy)
    elif skip_dro:
        console.print("[yellow]Skipping Wasserstein DRO[/yellow]")

    # LLM methods
    if not skip_llm:
        # Use command-line override or config model
        model_to_use = llm_model if llm_model else cfg.llm.model
        console.print(f"[bold]Initializing LLM methods with {model_to_use}...[/bold]")
        if use_fake_llm:
            console.print("[yellow]Using fake LLM client[/yellow]")
            llm_policies = create_llm_policies(
                base_url=cfg.llm.base_url,
                model=model_to_use,
                temperature=cfg.llm.temperature,
                use_fake=True,
                multiline=cfg.llm.multiline,
            )
        else:
            client = OllamaClient(base_url=cfg.llm.base_url, model=model_to_use)
            if client.is_available():
                llm_policies = create_llm_policies(
                    base_url=cfg.llm.base_url,
                    model=model_to_use,
                    temperature=cfg.llm.temperature,
                    use_fake=False,
                    multiline=cfg.llm.multiline,
                )
            else:
                console.print("[yellow]Ollama not available, using fake client[/yellow]")
                llm_policies = create_llm_policies(
                    base_url=cfg.llm.base_url,
                    model=model_to_use,
                    temperature=cfg.llm.temperature,
                    use_fake=True,
                    multiline=cfg.llm.multiline,
                )
        policies.extend(llm_policies)
    else:
        console.print("[yellow]Skipping LLM methods[/yellow]")

    # Run rolling evaluation
    console.print("[bold]Running rolling window evaluation...[/bold]")

    def progress_callback(current, total):
        console.print(f"  Processing window {current}/{total}")

    results = evaluator.run_evaluation(
        policies=policies,
        confidence_level=cfg.rolling.confidence_level if cfg.rolling.enabled else 0.95,
        progress_callback=progress_callback,
    )

    # Display results table
    table = Table(title=f"Rolling Window Evaluation ({n_windows} windows)")
    table.add_column("Method", style="cyan")
    table.add_column("Mean Return (%)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Mean Sharpe", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Mean Max DD (%)", justify="right")
    table.add_column("95% CI", justify="right")

    def format_ci(ci_low: float, ci_high: float, fmt: str = ".2f") -> str:
        return f"[{ci_low:{fmt}}, {ci_high:{fmt}}]"

    metrics_list = []
    for policy_name, result in results.items():
        ms = result.metrics_summary
        table.add_row(
            policy_name,
            f"{ms['total_return'].mean:.2f}",
            format_ci(ms['total_return'].ci_low, ms['total_return'].ci_high),
            f"{ms['sharpe_ratio'].mean:.3f}",
            format_ci(ms['sharpe_ratio'].ci_low, ms['sharpe_ratio'].ci_high, ".3f"),
            f"{ms['max_drawdown'].mean:.2f}",
            format_ci(ms['max_drawdown'].ci_low, ms['max_drawdown'].ci_high),
        )

        entry = {
            "method": policy_name,
            "n_windows": result.n_windows,
        }
        for metric_name, summary in ms.items():
            entry[f"{metric_name}_mean"] = summary.mean
            entry[f"{metric_name}_std"] = summary.std
            entry[f"{metric_name}_ci_low"] = summary.ci_low
            entry[f"{metric_name}_ci_high"] = summary.ci_high
            entry[f"{metric_name}_values"] = summary.values

        if result.mean_parse_error_rate is not None:
            entry["mean_parse_error_rate"] = result.mean_parse_error_rate
            entry["parse_error_rates"] = result.parse_error_rates

        metrics_list.append(entry)

    console.print(table)

    # Show parse error rates if any LLM methods
    llm_results = {k: v for k, v in results.items() if v.mean_parse_error_rate is not None}
    if llm_results:
        console.print("\n[bold]LLM Parse Error Rates:[/bold]")
        for name, result in llm_results.items():
            console.print(f"  {name}: {result.mean_parse_error_rate:.2%} (mean across windows)")

    # Save results
    metrics_path = output_path / "rolling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)

    # Summary CSV
    summary_data = []
    for entry in metrics_list:
        row = {
            "method": entry["method"],
            "n_windows": entry["n_windows"],
        }
        for metric in ["total_return", "sharpe_ratio", "max_drawdown", "sortino_ratio", "calmar_ratio", "avg_turnover"]:
            row[f"{metric}_mean"] = entry.get(f"{metric}_mean")
            row[f"{metric}_ci_low"] = entry.get(f"{metric}_ci_low")
            row[f"{metric}_ci_high"] = entry.get(f"{metric}_ci_high")
        if "mean_parse_error_rate" in entry:
            row["mean_parse_error_rate"] = entry["mean_parse_error_rate"]
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "rolling_summary.csv", index=False)

    # Save per-window results
    windows_dir = output_path / "windows"
    windows_dir.mkdir(exist_ok=True)
    for policy_name, result in results.items():
        policy_dir = windows_dir / policy_name.replace("/", "_")
        policy_dir.mkdir(exist_ok=True)
        for i, wr in enumerate(result.window_results):
            equity_df = pd.DataFrame({
                "step": range(len(wr.equity_curve)),
                "equity": wr.equity_curve,
            })
            equity_df.to_csv(policy_dir / f"window_{i}_equity.csv", index=False)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")


@app.command()
def interpret(
    config: Annotated[str, typer.Option(help="Path to config file")] = "configs/baby.toml",
    num_examples: Annotated[int, typer.Option("-n", help="Number of windows to analyze")] = 5,
    policy: Annotated[Optional[str], typer.Option(help="Policy to analyze: LLM-Direct, LLM-FewShot, or LLM-CoT")] = None,
    html: Annotated[bool, typer.Option(help="Export HTML visualizations")] = False,
    output_dir: Annotated[Optional[str], typer.Option(help="Output directory")] = None,
    use_fake_llm: Annotated[bool, typer.Option(help="Use fake LLM client")] = False,
):
    """Visualize which input values influence LLM portfolio decisions."""
    from .interpretability import (
        analyze_windows,
        render_colored_prompt,
        render_summary_panel,
    )
    from .interpretability.analyzer import compute_aggregate_stats
    from .interpretability.visualization import export_html

    console.print(f"[bold]Loading config from {config}...[/bold]")
    cfg = load_config(config)

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/interpret/{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    console.print(f"[bold]Loading dataset from {cfg.dataset.path}...[/bold]")
    try:
        price_relatives, log_relatives = load_dataset(cfg.dataset.path)
    except FileNotFoundError:
        console.print(f"[red]Dataset not found at {cfg.dataset.path}[/red]")
        console.print("Run 'portfolio-bench download-data' first")
        raise typer.Exit(1)

    T, N = price_relatives.shape
    lookback = cfg.dataset.lookback
    console.print(f"Dataset shape: T={T}, N={N}, lookback={lookback}")

    # Generate states (windows) from test portion
    train_ratio = cfg.backtest.train_ratio
    val_ratio = cfg.backtest.val_ratio
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # Use test data for analysis
    test_log_relatives = log_relatives[val_end:]
    n_available = len(test_log_relatives) - lookback
    n_windows = min(num_examples, n_available)

    if n_windows <= 0:
        console.print("[red]Not enough data for the specified lookback[/red]")
        raise typer.Exit(1)

    # Sample windows evenly from test set
    indices = np.linspace(0, n_available - 1, n_windows, dtype=int)
    states = [test_log_relatives[i : i + lookback] for i in indices]

    console.print(f"[bold]Analyzing {n_windows} windows...[/bold]")

    # Initialize LLM policy
    if use_fake_llm:
        console.print("[yellow]Using fake LLM client[/yellow]")
        client = FakeOllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
    else:
        client = OllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model)
        if not client.is_available():
            console.print("[yellow]Ollama not available, using fake client[/yellow]")
            client = FakeOllamaClient(base_url=cfg.llm.base_url, model=cfg.llm.model)

    # Determine which policies to analyze
    policy_map = {
        "LLM-Direct": lambda: DirectPolicy(client, cfg.llm.temperature, cfg.llm.multiline),
        "LLM-FewShot": lambda: FewShotPolicy(client, cfg.llm.temperature, cfg.llm.multiline),
        "LLM-CoT": lambda: ChainOfThoughtPolicy(client, cfg.llm.temperature, cfg.llm.multiline),
    }

    if policy:
        if policy not in policy_map:
            console.print(f"[red]Unknown policy: {policy}[/red]")
            console.print(f"Valid options: {', '.join(policy_map.keys())}")
            raise typer.Exit(1)
        policies_to_analyze = [(policy, policy_map[policy]())]
    else:
        # Default to LLM-Direct
        policies_to_analyze = [("LLM-Direct", policy_map["LLM-Direct"]())]

    # Analyze each policy
    all_results = {}
    for policy_name, llm_policy in policies_to_analyze:
        console.print(f"\n[bold]Analyzing {policy_name}...[/bold]")

        def progress_callback(current, total):
            console.print(f"  Window {current}/{total}")

        analyses = analyze_windows(
            policy=llm_policy,
            states=states,
            progress_callback=progress_callback,
        )

        # Compute aggregate stats
        stats = compute_aggregate_stats(analyses)
        all_results[policy_name] = {"analyses": analyses, "stats": stats}

        # Display summary
        console.print()
        console.print(render_summary_panel(stats, policy_name))

        # Display individual windows
        for analysis in analyses:
            console.print()
            console.print(render_colored_prompt(analysis, policy_name))

        # Export HTML if requested
        if html:
            html_path = output_path / "visualizations" / f"{policy_name.replace('/', '_')}.html"
            export_html(analyses, stats, policy_name, html_path)
            console.print(f"\n[green]HTML exported to: {html_path}[/green]")

    # Save summary JSON
    summary_data = {}
    for policy_name, data in all_results.items():
        summary_data[policy_name] = data["stats"]

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    app()
