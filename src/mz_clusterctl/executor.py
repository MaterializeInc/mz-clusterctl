"""
SQL execution engine for mz-clusterctl

Handles execution of actions in apply mode with proper error handling and audit logging.
"""

from typing import Any

from .db import Database
from .log import get_logger
from .models import Action

logger = get_logger(__name__)


class Executor:
    """
    Executes SQL actions and maintains audit trail

    Responsible for:
    1. Executing SQL commands in sequence
    2. Logging all actions to audit table
    3. Handling errors gracefully
    4. Providing execution feedback
    """

    def __init__(self, db: Database):
        self.db = db

    def execute_actions(self, cluster_id: str, actions: list[Action]) -> dict[str, Any]:
        """
        Execute a list of actions for a cluster

        Args:
            cluster_id: ID of the cluster
            actions: List of actions to execute

        Returns:
            Dictionary with execution summary
        """
        if not actions:
            return {"total": 0, "executed": 0, "failed": 0, "errors": []}

        summary = {"total": len(actions), "executed": 0, "failed": 0, "errors": []}

        logger.debug(
            "Starting action execution",
            extra={"cluster_id": cluster_id, "total_actions": len(actions)},
        )

        for i, action in enumerate(actions, 1):
            action_id = None
            executed = False
            error_message = None

            try:
                logger.debug(
                    f"Executing action {i}/{len(actions)}",
                    extra={
                        "cluster_id": cluster_id,
                        "action_sql": action.sql,
                    },
                )

                # Create decision context for audit log
                decision_ctx = {
                    "action_index": i,
                    "total_actions": len(actions),
                    "reasons": action.reasons,
                }

                # Execute the SQL
                logger.debug(
                    "About to execute action SQL",
                    extra={"cluster_id": cluster_id, "action_sql": action.sql},
                )
                result = self.db.execute_sql(action.sql)
                executed = True
                summary["executed"] += 1

                # Update decision context with execution results
                decision_ctx["execution_result"] = result

                print(f"✓ {action.sql}")
                if result.get("rowcount", 0) > 0:
                    print(f"  Affected rows: {result['rowcount']}")

                logger.info(
                    "Action executed successfully",
                    extra={
                        "cluster_id": cluster_id,
                        "action_sql": action.sql,
                        "rowcount": result.get("rowcount", 0),
                    },
                )

            except Exception as e:
                executed = False
                error_message = str(e)
                summary["failed"] += 1
                summary["errors"].append(
                    {"action_index": i, "sql": action.sql, "error": error_message}
                )

                decision_ctx = {
                    "action_index": i,
                    "total_actions": len(actions),
                    "error": error_message,
                    "reasons": action.reasons,
                }

                print(f"✗ {action.sql}")
                print(f"  Error: {error_message}")
                print()

                logger.error(
                    "Action execution failed",
                    extra={
                        "cluster_id": cluster_id,
                        "action_sql": action.sql,
                        "error": error_message,
                    },
                    exc_info=True,
                )

            finally:
                # Always log to audit table
                try:
                    logger.debug(
                        "Logging action to audit table",
                        extra={"cluster_id": cluster_id, "executed": executed},
                    )
                    action_id = self.db.log_action(
                        cluster_id=cluster_id,
                        action_sql=action.sql,
                        decision_ctx=decision_ctx,
                        executed=executed,
                        error_message=error_message,
                    )
                    logger.debug(
                        "Action logged to audit table successfully",
                        extra={"cluster_id": cluster_id, "action_id": action_id},
                    )

                    logger.debug(
                        "Action logged to audit table",
                        extra={
                            "cluster_id": cluster_id,
                            "action_id": str(action_id),
                            "executed": executed,
                        },
                    )

                except Exception as audit_error:
                    # Don't fail the entire execution if audit logging fails
                    logger.error(
                        "Failed to log action to audit table",
                        extra={
                            "cluster_id": cluster_id,
                            "action_sql": action.sql,
                            "audit_error": str(audit_error),
                        },
                        exc_info=True,
                    )

            # Stop execution on first error (fail-fast approach)
            if not executed:
                logger.warning(
                    "Stopping execution due to error",
                    extra={
                        "cluster_id": cluster_id,
                        "failed_action_index": i,
                        "remaining_actions": len(actions) - i,
                    },
                )
                break

        logger.info(
            "Action execution completed",
            extra={"cluster_id": cluster_id, "summary": summary},
        )

        # Print execution summary
        if summary["failed"] > 0:
            print(
                f"Execution completed with errors: "
                f"{summary['executed']}/{summary['total']} actions succeeded"
            )
            print("Errors:")
            for error in summary["errors"]:
                print(f"  Action {error['action_index']}: {error['error']}")
        else:
            print(f"All {summary['executed']} actions executed successfully")

        return summary
