# Portfolio Benchmark (Replication Bundle)

A FAIR experimental benchmark comparing Operations Research (OR) and Large Language Model (LLM) methods for portfolio allocation.

**Disclaimer:** This software is for research and educational purposes only. It is not financial advice. Do not use this for actual trading decisions.

**Note:** This is a minimal replication bundle. It includes preprocessed datasets and the core code needed to reproduce the paper's experiments.

## Overview

### What is "SOTA-ish"?

The methods implemented here are "SOTA-ish" in the sense that they represent well-established, widely-cited approaches within their respective paradigms (OR, LLM). They are not necessarily the absolute state-of-the-art for each category, but rather canonical implementations that serve as strong baselines for comparison. The goal is fair comparison across paradigms, not achieving peak performance in any single paradigm.

This benchmark provides a unified framework for evaluating portfolio allocation methods across 2 paradigms:

### Operations Research Methods
- **Mean-Variance (Markowitz)**: Classic quadratic optimization maximizing risk-adjusted returns
- **CVaR (Conditional Value-at-Risk)**: Linear programming minimizing tail risk
- **HRP (Hierarchical Risk Parity)**: Hierarchical clustering-based allocation

### LLM Methods
- **Direct Prompting**: Simple instruction-based allocation
- **Few-Shot Prompting**: In-context learning with examples
- **Chain-of-Thought**: Step-by-step reasoning for allocation

## FAIR Design Principles

All methods are evaluated under identical conditions:
- **Same Input**: Historical log-returns matrix (L x N)
- **Same Output**: Portfolio weights summing to 1
- **Same Evaluation**: Unified backtester with transaction costs
- **Same Metrics**: Total return, Sharpe ratio, max drawdown, etc.

## Installation

```bash
# Install from this bundle
pip install -e .
```

## Quick Start

```bash
# Run a small benchmark (NYSE 5-asset subset)
python -m portfolio_bench.cli run --config configs/nyse_scale_5.toml

# If Ollama is not available, use the fake LLM client
python -m portfolio_bench.cli run --config configs/nyse_scale_5.toml --use-fake-llm
```

## Usage

### CLI Commands

```bash
# Run NYSE full dataset
python -m portfolio_bench.cli run --config configs/nyse.toml

# Run DJIA full dataset (via full.toml)
python -m portfolio_bench.cli run --config configs/full.toml

# Run SP500 / MSCI full datasets
python -m portfolio_bench.cli run --config configs/sp500.toml
python -m portfolio_bench.cli run --config configs/msci.toml

# Visualize LLM input importance (interpretability)
python -m portfolio_bench.cli interpret --config configs/nyse_scale_5.toml
```

### Configuration

Configuration files use TOML format:

```toml
[dataset]
path = "data/processed/nyse_scale_5.npz"
lookback = 10

[backtest]
transaction_cost = 0.001
train_ratio = 0.0
val_ratio = 0.0
test_ratio = 1.0

[llm]
model = "qwen3:1.7b"
base_url = "http://localhost:11434"
temperature = 0
```

### LLM Setup

For LLM methods, you need a running Ollama server:

```bash
# Install Ollama (macOS)
brew install ollama

# Start server
ollama serve

# Pull model
ollama pull qwen3:1.7b
```

If Ollama is not available, use `--use-fake-llm` for testing.

## Project Structure

```
llm-portfolio-optimization/
├── configs/
│   ├── nyse_scale_5.toml
│   ├── nyse.toml
│   ├── sp500.toml
│   ├── msci.toml
│   └── full.toml
├── data/
│   └── processed/         # Preprocessed datasets used in the paper
├── src/portfolio_bench/
│   ├── backtest/
│   ├── data/
│   ├── interpretability/
│   ├── methods/
│   ├── utils/
│   ├── cli.py
│   └── config.py
├── pyproject.toml
└── REPLICATION.md
```

## Metrics

The benchmark computes the following metrics:

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative return over test period |
| Annualized Return | Annualized geometric return |
| Sharpe Ratio | Risk-adjusted return (assumes 0 risk-free rate) |
| Max Drawdown | Maximum peak-to-trough decline |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | Return relative to max drawdown |
| Avg Turnover | Average portfolio turnover per period |

## Output

Results are saved to `outputs/runs/<timestamp>/`:
- `metrics.json`: All metrics in JSON format
- `summary_table.csv`: Summary table
- `equity_curves/`: Per-method equity curves

## Reproducibility

This benchmark is designed with reproducibility in mind:

- **Deterministic Evaluation**: OR methods are deterministic given the same input.
- **LLM Temperature**: LLM methods use temperature=0 by default for (relatively) deterministic outputs.
- **Data Pipeline**: Dataset processing is deterministic given the same source .mat file. SHA-256 hash is recorded in metadata.json.

To reproduce results:
1. Use the same config file
2. Ensure the same random seed
3. Use the same dataset (verified by hash in metadata.json)

Note: LLM outputs may vary slightly across Ollama versions or model updates even with temperature=0.

## References

- [OLPS Toolbox](https://github.com/OLPS/OLPS)
- [Universal Portfolios](https://github.com/Marigold/universal-portfolios)

## License

MIT License
