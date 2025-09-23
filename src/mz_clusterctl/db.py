"""
Database connection and helper functions for mz-clusterctl

Provides PostgreSQL connection pool and database schema management.
"""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .log import get_logger
from .models import ClusterInfo, ReplicaInfo, StrategyConfig, StrategyState

logger = get_logger(__name__)


def _sanitize_database_url(url: str) -> str:
    """Sanitize database URL to hide credentials"""
    # Pattern to match database URLs and hide credentials
    # postgres://user:password@host:port/db -> postgres://***:***@host:port/db
    # Also handle edge cases like URLs without passwords or with special characters
    patterns = [
        # Standard postgres://user:password@host pattern
        (r"(postgres(?:ql)?://)[^:@/]+:[^@/]+(@[^/]+/?.*)$", r"\1***:***\2"),
        # postgres://user@host pattern (no password)
        (r"(postgres(?:ql)?://)[^:@/]+(@[^/]+/?.*)$", r"\1***\2"),
        # Handle URLs with encoded characters
        (r"(postgres(?:ql)?://)[^@/]+(@[^/]+/?.*)$", r"\1***\2"),
    ]

    sanitized = url
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
        if sanitized != url:
            break

    return sanitized


def _sanitize_error_message(message: str, database_url: str) -> str:
    """Remove database URL from error message"""
    if database_url in message:
        sanitized_url = _sanitize_database_url(database_url)
        return message.replace(database_url, sanitized_url)
    return message


class Database:
    """Database connection manager and helper methods"""

    def __init__(self, database_url: str, cluster: str | None = None):
        self._database_url = database_url
        self._cluster = cluster
        try:
            self.pool = ConnectionPool(
                conninfo=database_url, min_size=1, max_size=10, open=True
            )
        except Exception as e:
            sanitized_error = _sanitize_error_message(str(e), database_url)
            logger.error(
                "Failed to create database connection pool",
                extra={"error": sanitized_error},
            )
            raise RuntimeError(f"Database connection failed: {sanitized_error}") from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the connection pool"""
        try:
            self.pool.close()
        except Exception as e:
            sanitized_error = _sanitize_error_message(str(e), self._database_url)
            logger.error(
                "Error closing database connection pool",
                extra={"error": sanitized_error},
            )
            # Don't re-raise as we're likely in cleanup

    @contextmanager
    def get_connection(self) -> Iterator[psycopg.Connection]:
        """Get a connection from the pool"""
        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            # Set cluster if specified
            if self._cluster:
                with conn.cursor() as cur:
                    set_cluster_sql = f"SET cluster = {self._cluster}"
                    logger.debug(
                        "Setting cluster for connection",
                        extra={"cluster": self._cluster, "sql": set_cluster_sql},
                    )
                    try:
                        cur.execute(set_cluster_sql)
                    except Exception as e:
                        sanitized_error = _sanitize_error_message(
                            str(e), self._database_url
                        )
                        logger.error(
                            "Error setting cluster",
                            extra={"cluster": self._cluster, "error": sanitized_error},
                            exc_info=True,
                        )
                        raise
            yield conn

    def ensure_tables(self):
        """Create required tables if they don't exist"""
        # Execute each CREATE TABLE in its own transaction
        tables = [
            (
                "mz_cluster_strategies",
                """
                CREATE TABLE IF NOT EXISTS mz_cluster_strategies (
                    cluster_id    TEXT            NOT NULL,
                    strategy_type TEXT            NOT NULL,
                    config        JSONB           NOT NULL,
                    updated_at    TIMESTAMPTZ     DEFAULT now()
                )
            """,
            ),
            (
                "mz_cluster_strategy_state",
                """
                CREATE TABLE IF NOT EXISTS mz_cluster_strategy_state (
                    cluster_id    TEXT            NOT NULL,
                    state_version INT             NOT NULL,
                    payload       JSONB           NOT NULL,
                    updated_at    TIMESTAMPTZ     DEFAULT now()
                )
            """,
            ),
            (
                "mz_cluster_strategy_actions",
                """
                CREATE TABLE IF NOT EXISTS mz_cluster_strategy_actions (
                    action_id     TEXT            NOT NULL,
                    cluster_id    TEXT,
                    action_sql    TEXT,
                    decision_ctx  JSONB,
                    executed      BOOL,
                    error_message TEXT,
                    created_at    TIMESTAMPTZ     DEFAULT now()
                )
            """,
            ),
        ]

        for _table_name, sql in tables:
            with self.get_connection() as conn, conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()

        logger.debug("Database tables ensured")

    def get_clusters(self, name_filter: str | None = None) -> list[ClusterInfo]:
        """Get cluster information from mz_catalog.mz_clusters"""
        with self.get_connection() as conn, conn.cursor() as cur:
            sql = "SELECT id, name FROM mz_catalog.mz_clusters"
            logger.debug("Executing SQL", extra={"sql": sql, "params": None})
            try:
                cur.execute(sql)
            except Exception as e:
                sanitized_error = _sanitize_error_message(str(e), self._database_url)
                logger.error(
                    "Error executing SQL",
                    extra={"sql": sql, "params": None, "error": sanitized_error},
                    exc_info=True,
                )
                raise
            clusters = []

            for row in cur.fetchall():
                cluster = ClusterInfo.from_db_row(row)

                # Only process user clusters (IDs starting with 'u')
                if not cluster.id.startswith("u"):
                    logger.debug(
                        "Skipping non-user cluster",
                        extra={
                            "cluster_id": cluster.id,
                            "cluster_name": cluster.name,
                        },
                    )
                    continue

                # Apply name filter if provided
                if name_filter and not self._matches_filter(cluster.name, name_filter):
                    continue

                # Get replicas for this cluster using catalog query instead of SHOW
                sql = (
                    "SELECT name, size FROM mz_catalog.mz_cluster_replicas "
                    "WHERE cluster_id = %s"
                )
                params = (cluster.id,)
                logger.debug(
                    "Executing SQL",
                    extra={
                        "sql": sql,
                        "params": params,
                    },
                )
                try:
                    cur.execute(sql, params)
                    replica_infos = [
                        ReplicaInfo(name=replica_row["name"], size=replica_row["size"])
                        for replica_row in cur.fetchall()
                    ]
                    # ClusterInfo is frozen, so we need to create a new instance
                    cluster = ClusterInfo(
                        id=cluster.id,
                        name=cluster.name,
                        replicas=tuple(replica_infos),
                        managed=cluster.managed,
                    )
                except Exception as e:
                    sanitized_error = _sanitize_error_message(
                        str(e), self._database_url
                    )
                    logger.error(
                        "Error executing SQL",
                        extra={"sql": sql, "params": params, "error": sanitized_error},
                        exc_info=True,
                    )
                    raise

                logger.debug(
                    "Created ClusterInfo",
                    extra={
                        "cluster_id": cluster.id,
                        "cluster_name": cluster.name,
                        "replicas": cluster.replicas,
                    },
                )

                clusters.append(cluster)

            return clusters

    def get_strategy_configs(self) -> list[StrategyConfig]:
        """Get strategy configurations from mz_cluster_strategies"""
        with self.get_connection() as conn, conn.cursor() as cur:
            sql = """
             SELECT DISTINCT ON (cluster_id, strategy_type)
                    cluster_id, strategy_type, config, updated_at
             FROM mz_cluster_strategies
             ORDER BY cluster_id, strategy_type, updated_at DESC
             """
            logger.debug("Executing SQL", extra={"sql": sql, "params": None})
            try:
                cur.execute(sql)
            except Exception as e:
                sanitized_error = _sanitize_error_message(str(e), self._database_url)
                logger.error(
                    "Error executing SQL",
                    extra={"sql": sql, "params": None, "error": sanitized_error},
                    exc_info=True,
                )
                raise

            return [StrategyConfig.from_db_row(row) for row in cur.fetchall()]

    def get_strategy_state(self, cluster_id: str) -> StrategyState | None:
        """Get strategy state for a cluster"""
        with self.get_connection() as conn, conn.cursor() as cur:
            sql = (
                "SELECT * FROM mz_cluster_strategy_state WHERE cluster_id = %s "
                "ORDER BY updated_at DESC LIMIT 1"
            )
            params = (cluster_id,)
            logger.debug(
                "Executing SQL",
                extra={
                    "sql": sql,
                    "params": params,
                },
            )
            try:
                cur.execute(sql, params)
                row = cur.fetchone()
            except Exception as e:
                sanitized_error = _sanitize_error_message(str(e), self._database_url)
                logger.error(
                    "Error executing SQL",
                    extra={"sql": sql, "params": params, "error": sanitized_error},
                    exc_info=True,
                )
                raise

            if not row:
                return None

            return StrategyState(
                cluster_id=row["cluster_id"],
                strategy_type="",  # Not stored in state table
                state_version=row["state_version"],
                payload=row["payload"],
                updated_at=row["updated_at"],
            )

    # For now, we're not upserting but only appending. Helps with debugging
    # state transitions.
    def upsert_strategy_state(self, state: StrategyState):
        """Insert or update strategy state"""
        logger.debug(
            "Starting upsert_strategy_state",
            extra={
                "cluster_id": state.cluster_id,
                "state_version": state.state_version,
            },
        )
        with self.get_connection() as conn, conn.cursor() as cur:
            sql = """
                INSERT INTO mz_cluster_strategy_state 
                    (cluster_id, state_version, payload, updated_at)
                    VALUES (%s, %s, %s, now())
                """
            try:
                params = (
                    state.cluster_id,
                    state.state_version,
                    json.dumps(state.payload),
                )
            except (TypeError, ValueError) as e:
                logger.error(
                    "Error serializing strategy state payload",
                    extra={"cluster_id": state.cluster_id, "error": str(e)},
                    exc_info=True,
                )
                raise ValueError(f"Invalid strategy state payload: {e}") from e

            logger.debug(
                "Executing SQL",
                extra={
                    "sql": sql,
                    "params": params,
                },
            )
            try:
                cur.execute(sql, params)
                conn.commit()
            except Exception as e:
                sanitized_error = _sanitize_error_message(str(e), self._database_url)
                logger.error(
                    "Error executing SQL",
                    extra={"sql": sql, "params": params, "error": sanitized_error},
                    exc_info=True,
                )
                raise

    def log_action(
        self,
        cluster_id: str,
        action_sql: str,
        decision_ctx: dict[str, Any],
        executed: bool,
        error_message: str | None = None,
    ) -> str:
        """Log an action to the audit table"""
        action_id = str(uuid4())
        logger.debug(
            "Starting log_action",
            extra={
                "action_id": action_id,
                "cluster_id": cluster_id,
                "executed": executed,
            },
        )
        with self.get_connection() as conn, conn.cursor() as cur:
            sql = """
                INSERT INTO mz_cluster_strategy_actions
                    (action_id, cluster_id, action_sql, decision_ctx, executed, 
                     error_message)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
            try:
                params = (
                    action_id,
                    cluster_id,
                    action_sql,
                    json.dumps(decision_ctx),
                    executed,
                    error_message,
                )
            except (TypeError, ValueError) as e:
                logger.error(
                    "Error serializing decision context",
                    extra={"cluster_id": cluster_id, "error": str(e)},
                    exc_info=True,
                )
                raise ValueError(f"Invalid decision context: {e}") from e

            logger.debug(
                "Executing SQL",
                extra={
                    "sql": sql,
                    "params": params,
                },
            )
            try:
                cur.execute(sql, params)
                conn.commit()
                return action_id
            except Exception as e:
                sanitized_error = _sanitize_error_message(str(e), self._database_url)
                logger.error(
                    "Error executing SQL",
                    extra={"sql": sql, "params": params, "error": sanitized_error},
                    exc_info=True,
                )
                raise

    def execute_sql(self, sql: str) -> dict[str, Any]:
        """Execute arbitrary SQL and return result info"""
        logger.debug("Starting execute_sql", extra={"sql": sql})
        with self.get_connection() as conn, conn.cursor() as cur:
            logger.debug("Executing SQL", extra={"sql": sql, "params": None})
            try:
                cur.execute(sql)
                conn.commit()
                return {
                    "rowcount": cur.rowcount,
                    "statusmessage": cur.statusmessage,
                }
            except Exception as e:
                sanitized_error = _sanitize_error_message(str(e), self._database_url)
                logger.error(
                    "Error executing SQL",
                    extra={"sql": sql, "params": None, "error": sanitized_error},
                    exc_info=True,
                )
                raise

    def wipe_strategy_state(self, cluster_id: str | None = None):
        """Clear strategy state table"""
        logger.debug("Starting wipe_strategy_state", extra={"cluster_id": cluster_id})
        with self.get_connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                if cluster_id:
                    sql = "DELETE FROM mz_cluster_strategy_state WHERE cluster_id = %s"
                    params = (cluster_id,)
                    logger.debug(
                        "Executing SQL",
                        extra={
                            "sql": sql,
                            "params": params,
                        },
                    )
                    try:
                        cur.execute(sql, params)
                    except Exception as e:
                        sanitized_error = _sanitize_error_message(
                            str(e), self._database_url
                        )
                        logger.error(
                            "Error executing SQL",
                            extra={
                                "sql": sql,
                                "params": params,
                                "error": sanitized_error,
                            },
                            exc_info=True,
                        )
                        raise
                else:
                    sql = "DELETE FROM mz_cluster_strategy_state"
                    logger.debug("Executing SQL", extra={"sql": sql, "params": None})
                    try:
                        cur.execute(sql)
                    except Exception as e:
                        sanitized_error = _sanitize_error_message(
                            str(e), self._database_url
                        )
                        logger.error(
                            "Error executing SQL",
                            extra={
                                "sql": sql,
                                "params": None,
                                "error": sanitized_error,
                            },
                            exc_info=True,
                        )
                        raise

                logger.info(f"Wiped strategy state for {cur.rowcount} clusters")

    def _matches_filter(self, name: str, pattern: str) -> bool:
        """Check if name matches filter pattern (simple regex)"""
        try:
            return bool(re.search(pattern, name))
        except re.error:
            # If regex is invalid, fall back to substring match
            logger.warning(
                "Invalid regex pattern, falling back to substring match",
                extra={"pattern": pattern},
            )
            return pattern.lower() in name.lower()
