# Replication Guide

Exact steps to reproduce the benchmark experiments in the paper.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) for LLM inference (optional if using `--use-fake-llm`)

## Setup

```bash
# Install from this bundle
pip install -e .

# Install and start Ollama (optional, for LLM runs)
brew install ollama      # macOS
ollama serve             # Start server (keep running in background)

# Pull the LLM model
ollama pull qwen3:1.7b
```

## Data Preparation

Preprocessed datasets are already included under `data/processed/`.
If you want to re-download from source:

```bash
python -m portfolio_bench.cli download-data --dataset djia
python -m portfolio_bench.cli download-data --dataset nyse
python -m portfolio_bench.cli download-data --dataset sp500
python -m portfolio_bench.cli download-data --dataset msci
```

## Running Experiments

### Small benchmark (NYSE 5-asset subset)

```bash
python -m portfolio_bench.cli run --config configs/nyse_scale_5.toml
```

### Full dataset benchmarks

```bash
# NYSE full
python -m portfolio_bench.cli run --config configs/nyse.toml

# DJIA full (via full.toml)
python -m portfolio_bench.cli run --config configs/full.toml

# SP500 / MSCI full
python -m portfolio_bench.cli run --config configs/sp500.toml
python -m portfolio_bench.cli run --config configs/msci.toml
```

### Testing without Ollama

Use fake LLM (returns uniform weights for testing):

```bash
python -m portfolio_bench.cli run --config configs/nyse_scale_5.toml --use-fake-llm
```

## Output

Results are saved to `outputs/runs/<timestamp>/`:
- `metrics.json` - All metrics in JSON format
- `summary_table.csv` - Summary table

## Troubleshooting

### Ollama connection error

Ensure Ollama server is running:
```bash
ollama serve
```

### Model not found

Pull the model:
```bash
ollama pull qwen3:1.7b
```
