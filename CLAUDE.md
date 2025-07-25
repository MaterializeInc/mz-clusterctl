# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `mz-clusterctl`, an external cluster controller for Materialize. It's a Python CLI tool that manages Materialize cluster replicas based on configurable strategies like auto-scaling and idle shutdown.

## Development Commands

This project uses `uv` as the Python package manager. Common commands:

```bash
# Install dependencies and set up development environment
uv sync

# Run the CLI tool
uv run mz-clusterctl

# Install in development mode
uv pip install -e .

# Run with Python directly
uv run python -m mz_clusterctl

# Formatting and linting with ruff
uv run ruff format         # Format code
uv run ruff check          # Lint code
uv run ruff check --fix    # Fix auto-fixable lint issues
```

## Architecture

The project follows a stateless CLI execution model with the following key components:

- **Database Integration**: Uses PostgreSQL connection to Materialize via `DATABASE_URL` environment variable
- **Strategy System**: Pluggable strategy classes for different scaling behaviors (burst, idle_suspend, etc.)
- **State Management**: Persistent state stored in Materialize tables (`mz_cluster_strategies`, `mz_cluster_strategy_state`, `mz_cluster_strategy_actions`)
- **CLI Interface**: Three main commands - `dry-run`, `apply`, and `wipe-state`

### Core Architecture Components

```
mz_clusterctl/
├─ __main__.py          # CLI argument parsing and mode dispatch
├─ db.py                # PostgreSQL connection pool and database helpers
├─ models.py            # Data classes for StrategyState, ReplicaSpec, etc.
├─ signals.py           # Queries for activity and hydration status
├─ environment.py       # Environment configuration and detection
├─ constants.py         # Application constants and defaults
├─ strategies/
│   ├─ base.py          # Strategy interface: decide_desired_state() -> DesiredState
│   ├─ target_size.py   # Target size strategy implementation
│   ├─ burst.py         # Auto-scaling strategy implementation
│   ├─ shrink_to_fit.py # Shrink to fit strategy implementation
│   └─ idle_suspend.py  # Idle suspend strategy implementation
├─ engine.py            # Orchestration: load config → run strategies → merge → render SQL
├─ executor.py          # SQL execution for apply mode
└─ log.py               # Structured logging to stdout and audit tables
```

### Decision Cycle

Each invocation follows this pattern:
1. **Bootstrap**: Load cluster strategies and metadata
2. **State Hydration**: Restore previous state from database
3. **Run Strategies**: Execute strategy logic to generate actions
4. **Dry-Run/Apply**: Either display actions (dry-run) or execute them (apply)
5. **Persist State**: Save updated state back to database

## Configuration

- Environment variables: `DATABASE_URL` for Materialize connection
- Strategy configuration stored as JSON in `mz_cluster_strategies` table
- No external config files - all configuration is database-driven

## Key Design Decisions

- **Stateless**: Each invocation is independent, designed for cron/scheduled execution
- **Single Instance**: No leader election or coordination between instances
- **Best Effort**: Failures abort current run but don't affect future runs
- **Audit Trail**: All actions logged to `mz_cluster_strategy_actions` table
- **Strategy Registry**: Strategies registered in `strategies/__init__.py`, no dynamic plugins
