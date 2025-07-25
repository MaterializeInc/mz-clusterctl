# mz-clusterctl

mz-clusterctl is an external cluster controller for Materialize that
automatically manages cluster replicas based on configurable scaling
strategies. It monitors cluster activity, replica hydration status, and
resource utilization to make intelligent decisions about scaling up, scaling
down, or suspending replicas to optimize performance and cost.

The tool operates as a stateless CLI that can be run periodically (e.g. via
Github Actions or cron) to continually manage Materialize clusters according to
your scaling requirements.

> [!NOTE]
> This is an experimental tool and might change in the future. We are actively
> working on it and features might migrate into Materialize proper eventually.

## Setup

The suggested way of running this tool is to use a GitHub Actions workflow for
running it periodically against you Materialize environment. See
[README-github-actions.md](README-github-actions.md) to learn how to set that
up.

For a local development workflow, you can follow these instructions:

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure database connection:**
   Create a `.env` file with your Materialize connection string:
   ```bash
   # For local Materialize instance
   DATABASE_URL=postgresql://materialize@localhost:6875/materialize
   
   # For Materialize Cloud instance
   DATABASE_URL=postgresql:<your-materialize-connection-string>
   ```

3. **Initialize the controller:**
   ```bash
   # This creates the necessary tables
   uv run mz-clusterctl dry-run
   ```

## User Interface

### Configuring Strategies

Strategies are configured by inserting records into the `mz_cluster_strategies`
table in your Materialize database. Each strategy is defined by:

- **cluster_id**: The ID of the cluster to manage
- **strategy_type**: The type of strategy (e.g., "target_size", "burst", "idle_suspend")
- **config**: A JSON object containing strategy-specific parameters

**Basic example:**
```sql
INSERT INTO mz_cluster_strategies (cluster_id, strategy_type, config) VALUES
  ('u1', 'target_size', '{"target_size": "200cc"}');
```

**Advanced multi-strategy example:**
```sql
INSERT INTO mz_cluster_strategies (cluster_id, strategy_type, config) VALUES
  ('u1', 'target_size', '{"target_size": "50cc"}'),
  ('u1', 'burst', '{"burst_replica_size": "800cc", "cooldown_s": 60}'),
  ('u1', 'idle_suspend', '{"idle_after_s": 1800}');
```

### Strategy Execution Order

When multiple strategies are configured for the same cluster, they execute in
priority order:

1. `target_size` (priority 0) - lowest priority
2. `shrink_to_fit` (priority 1)
3. `burst` (priority 2)
4. `idle_suspend` (priority 3)

Higher priority strategies can modify or override decisions from lower priority strategies.

### Managing Strategies

**View current strategies:**
```sql
SELECT * FROM mz_cluster_strategies;
```

**Update a strategy:**
```sql
UPDATE mz_cluster_strategies
  SET config = '{"target_size": "large"}'
  WHERE cluster_id = 'u1' AND strategy_type = 'target_size';
```

**Remove a strategy:**
```sql
DELETE FROM mz_cluster_strategies 
WHERE cluster_id = 'u1' AND strategy_type = 'idle_suspend';
```

## Reconfiguration Downtime (or lack thereof)

The scheduling strategies will not retire hydrated replicas unless there is at
least one other replica that is hydrated and can serve queries. For example,
you can use the `target_size` strategy (see below) to do a zero-downtime change
in cluster sizing: when you update the `target_size` strategy for a cluster (or
set one in the first place), a replica of the target size will be spun up and
only once it is hydrated will other replicas be retired.

## Managed and Unmanaged Clusters

Materialize has the concept of _managed_ and _unmanaged_ clusters. This tool
requires clusters to be configured as unmanaged. You can either do that by
creating an unmanaged cluster in the first place or re-configuring a managed
one:

```sql
-- create unmanaged cluster
CREATE CLUSTER c REPLICAS ();

-- re-configure an existing cluster
ALTER CLUSTER c SET (MANAGED = false);
```

## Strategies

### target_size

Ensures a cluster has exactly one replica of a specified size. This can be used
to do a zero-downtime resizing of a cluster: you set the new target size, and
the existing replica(s) will only be retired once the new-size replica is
hydrated.

**Configuration:**
- `target_size` (required): The desired replica size (e.g., "25cc", "100cc", "200cc", "800cc")
- `replica_name` (optional): Custom name for the replica (default: "r_{target_size}")

**Example:**
```json
{
  "target_size": "200cc",
  "replica_name": "main_replica"
}
```

**Use case:** Maintain a consistent baseline replica size.

### burst

Adds large "burst" replicas when existing replicas are not hydrated (not ready
to serve queries), and removes them when other replicas become hydrated.

**Configuration:**
- `burst_replica_size` (required): Size of the burst replica (e.g., "800cc", "1600cc")
- `cooldown_s` (optional): Cooldown period in seconds between decisions (default: 0)

**Example:**
```json
{
  "burst_replica_size": "800cc",
  "cooldown_s": 60
}
```

**Use case:** Temporarily scale up during high-demand periods or when primary
replicas are starting up.

### idle_suspend

Suspends all cluster replicas after a configured period of inactivity.

**Configuration:**
- `idle_after_s` (required): Number of seconds of inactivity before suspending replicas
- `cooldown_s` (optional): Cooldown period in seconds between decisions (default: 0)

**Example:**
```json
{
  "idle_after_s": 1800,
  "cooldown_s": 300
}
```

**Use case:** Reduce costs by automatically suspending unused clusters.

### shrink_to_fit

Creates replicas of multiple sizes, then removes larger ones when smaller ones
can handle the workload.

> [!NOTE]
> This is an experimental strategy that is disabled by default. Enable using
> `--enable-experimental-strategies`. For this strategy the semantics are less
> obvious that as for the others, and it's easy to create a lot of expensive
> replicas. So use with caution.

**Configuration:**
- `max_replica_size` (required): Maximum replica size to create (for example, "800cc" or "1600cc")
- `cooldown_s` (optional): Cooldown period in seconds between decisions (default: 0)
- `min_oom_count` (optional): Minimum OOM count to consider a replica crash-looping (default: 1)
- `min_crash_count` (optional): Minimum crash count to consider a replica crash-looping (default: 1)

**Example:**
```json
{
  "max_replica_size": "1600cc",
  "cooldown_s": 120
}
```

**Use case:** Optimize replica sizes based on actual resource requirements.

## Operations

### Environment Configuration

The tool requires a PostgreSQL connection to your Materialize instance:

```bash
# Set environment variable
export DATABASE_URL=postgresql://materialize@localhost:6875/materialize

# Or use a .env file
echo "DATABASE_URL=postgresql://materialize@localhost:6875/materialize" > .env
```

Here and below, `materialize` is a placeholder for both a database and a
username which you want to use for running the tool. They should both be
included in the connection string.


**Important:** The database user/connection string must have access to the
following builtin collections and permissions:

**System catalog tables (read access required):**
- `mz_catalog.mz_clusters` - to enumerate clusters and their metadata
- `mz_catalog.mz_cluster_replicas` - to get replica information (name, size,
  etc.)
- `mz_catalog.mz_indexes` - to determine which objects need to be hydrated

**System internal tables (read access required):**
- `mz_internal.mz_statement_execution_history_redacted` - to track cluster
  activity for idle detection
- `mz_internal.mz_hydration_statuses` - to monitor replica hydration status for
  scaling decisions
- `mz_internal.mz_cluster_replica_status_history` - to track replica crashes
  and status changes
- `mz_internal.mz_active_peeks` - to get active query count for workload
  assessment

**DDL permissions required:**
- `CREATE CLUSTER REPLICA` - to create new replicas based on scaling strategies
- `DROP CLUSTER REPLICA` - to remove replicas when scaling down or suspending
- `CREATE TABLE` - to create management tables (`mz_cluster_strategies`,
  `mz_cluster_strategy_state`, `mz_cluster_strategy_actions`)
- `INSERT`, `SELECT`, `DELETE` - on the management tables for configuration and
  state persistence

**RBAC Setup:** If using Materialize Cloud or a system with RBAC enabled,
ensure the user specified in your connection string has the appropriate
permissions.

### Manual Table Creation

If you prefer not to grant `CREATE TABLE` privileges to the user, you can
manually create the required tables using the following SQL statements:

```sql
CREATE TABLE IF NOT EXISTS mz_cluster_strategies (
    cluster_id    TEXT            NOT NULL,
    strategy_type TEXT            NOT NULL,
    config        JSONB           NOT NULL,
    updated_at    TIMESTAMPTZ     DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mz_cluster_strategy_state (
    cluster_id    TEXT            NOT NULL,
    state_version INT             NOT NULL,
    payload       JSONB           NOT NULL,
    updated_at    TIMESTAMPTZ     DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mz_cluster_strategy_actions (
    action_id     TEXT            NOT NULL,
    cluster_id    TEXT,
    action_sql    TEXT,
    decision_ctx  JSONB,
    executed      BOOL,
    error_message TEXT,
    created_at    TIMESTAMPTZ     DEFAULT now()
);
```

After creating these tables, you'll also need to grant the necessary permissions:

```sql
-- Replace 'your_username' with your actual database username
GRANT SELECT, INSERT, DELETE ON mz_cluster_strategy_state TO your_username;
GRANT SELECT, INSERT, DELETE ON mz_cluster_strategy_actions TO your_username;
GRANT SELECT, INSERT, DELETE ON mz_cluster_strategies TO your_username;
```

### Running the Controller

**Dry-run mode** (view planned actions without executing):
```bash
uv run mz-clusterctl dry-run
```

**Apply mode** (execute actions):
```bash
uv run mz-clusterctl apply
```

**Verbose output** (for debugging):
```bash
uv run mz-clusterctl dry-run --verbose
```

**Target specific clusters**:
```bash
uv run mz-clusterctl apply --cluster "production-.*"
```

### Periodic Operation

The tool is designed to run periodically. A typical setup might run it every 1-5 minutes:

**Using GitHub Actions:**
See [README-github-actions.md](README-github-actions.md) for setup
instructions.

**Using cron:**
```bash
# Run every 2 minutes
*/2 * * * * cd /path/to/mz-clusterctl && uv run mz-clusterctl apply
```

**Using a simple loop:**
This can be useful for iterating on a local testing setup.

```bash
while true; do
    uv run mz-clusterctl apply
    sleep 60
done
```

### Monitoring

**View audit trail:**
```sql
SELECT * FROM mz_cluster_strategy_actions ORDER BY created_at DESC LIMIT 10;
```

**View current state:**
```sql
SELECT * FROM mz_cluster_strategy_state;
```

**View failed actions:**
```sql
SELECT * FROM mz_cluster_strategy_actions WHERE executed = false;
```

### State Management

**Clear state for debugging:**
```bash
uv run mz-clusterctl wipe-state
```

**Clear state for specific cluster:**
```bash
uv run mz-clusterctl wipe-state --cluster "problematic-cluster"
```

## Quirks

### Reaction Time Limitations

The controller can only react to changes at the interval it is run. This means:

- **Delayed Response**: If you run the controller every 5 minutes, it may take
  up to 5 minutes to react to changes in cluster activity or replica status.
- **Idle Suspend Timing**: The `idle_suspend` strategy cannot immediately spin
  up replicas when activity resumes. It will only detect activity and create
  replicas on the next execution cycle.
- **Burst Scaling**: The `burst` strategy may not immediately respond to
  replica hydration changes, potentially keeping burst replicas active longer
  than necessary.

### Non-Atomic Operations

Changes to controller state and cluster replicas are not atomic. This differs
from what might be expected if the scheduling functionality were integrated
directly into Materialize:

- **State Persistence**: Controller state is persisted to the database
  separately from cluster replica changes.
- **Failure Recovery**: If the controller fails after updating state but before
  creating/destroying replicas, the next execution may result in duplicate
  actions or inconsistent state.
- **Concurrent Execution**: Running multiple instances of the controller
  simultaneously is not supported and may lead to conflicting actions.

### Resource Monitoring Limitations

The controller relies on Materialize's system tables for monitoring:

- **Activity Detection**: Based on `mz_statement_execution_history_redacted`,
  which may not capture all forms of cluster activity.
- **Hydration Status**: Based on `mz_hydration_statuses`, which reflects the
  current state but may not predict future readiness.
- **Crash Detection**: Based on `mz_cluster_replica_status_history`, which may
  have delays in reporting status changes.

### Configuration Persistence

Strategy configurations are stored in the database and persist across
controller restarts. However:

- **No Validation**: Invalid configurations are not validated until execution
  time.
- **JSON Format**: Configuration parameters must be valid JSON and match
  expected schema.
- **Manual Management**: There is no built-in UI for managing strategies; all
  configuration is done via SQL.
