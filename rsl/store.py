import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA_VERSION = 4

_V2_COLUMNS = (
    ("rsl_decisions", "proposed_model", "TEXT"),
    ("rsl_decisions", "rollout_bucket", "INTEGER"),
    ("rsl_outcomes", "fallback_reason", "TEXT"),
)
_V3_COLUMNS = (
    ("rsl_rollout_state", "revision", "INTEGER NOT NULL DEFAULT 0"),
)
_V4_COLUMNS = (
    ("rsl_decisions", "action_distribution_json", "TEXT"),
)
_MIGRATIONS = (
    (2, _V2_COLUMNS),
    (3, _V3_COLUMNS),
    (4, _V4_COLUMNS),
)


class RolloutRevisionConflict(RuntimeError):
    pass


class RSLStore:
    """Persistence for RSL data, isolated to rsl_* tables in the router database."""

    def __init__(self, path: str):
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def migrate(self) -> None:
        conn = self._require_connection()
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rsl_schema_version (
                version INTEGER PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS rsl_decisions (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                api_key_id INTEGER NOT NULL DEFAULT 0,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                user_id TEXT,
                routing_profile_id TEXT,
                requested_model TEXT NOT NULL,
                baseline_model TEXT NOT NULL,
                selected_model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                features_json TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'baseline',
                behavior_policy TEXT NOT NULL DEFAULT 'baseline_only',
                policy_version TEXT,
                propensity REAL NOT NULL DEFAULT 1.0,
                applied INTEGER NOT NULL DEFAULT 0,
                eligible INTEGER NOT NULL DEFAULT 1,
                skip_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rsl_decisions_created
                ON rsl_decisions(created_at);
            CREATE INDEX IF NOT EXISTS idx_rsl_decisions_task_model
                ON rsl_decisions(task_type, baseline_model, created_at);
            CREATE TABLE IF NOT EXISTS rsl_outcomes (
                request_id TEXT PRIMARY KEY
                    REFERENCES rsl_decisions(request_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                executed_model TEXT,
                status TEXT NOT NULL,
                error_type TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                actual_cost REAL,
                latency_ms INTEGER,
                finish_reason TEXT,
                empty_response INTEGER,
                judge_score REAL,
                quality_score REAL,
                quality_source TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rsl_outcomes_created
                ON rsl_outcomes(created_at);
            CREATE TABLE IF NOT EXISTS rsl_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL
                    REFERENCES rsl_decisions(request_id) ON DELETE CASCADE,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                rating REAL,
                label TEXT,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(request_id, subject_type, subject_id)
            );
            CREATE TABLE IF NOT EXISTS rsl_datasets (
                dataset_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                filters_json TEXT NOT NULL,
                samples_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rsl_policies (
                policy_version TEXT PRIMARY KEY,
                parent_version TEXT,
                dataset_id TEXT NOT NULL
                    REFERENCES rsl_datasets(dataset_id),
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                policy_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                evaluation_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rsl_policies_created
                ON rsl_policies(created_at);
            CREATE TABLE IF NOT EXISTS rsl_rollout_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                stage TEXT NOT NULL DEFAULT 'disabled',
                policy_version TEXT,
                canary_percent REAL NOT NULL DEFAULT 0.0,
                previous_policy_version TEXT,
                updated_at TEXT NOT NULL,
                note TEXT
            );
            CREATE TABLE IF NOT EXISTS rsl_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                action TEXT NOT NULL,
                policy_version TEXT,
                from_stage TEXT,
                to_stage TEXT,
                actor TEXT,
                details_json TEXT NOT NULL
            );
            INSERT OR IGNORE INTO rsl_schema_version(version) VALUES (1);
            """
        )
        await conn.commit()
        cursor = await conn.execute(
            "SELECT MAX(version) AS version FROM rsl_schema_version"
        )
        row = await cursor.fetchone()
        database_version = int(row["version"]) if row and row["version"] is not None else 0
        if database_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"RSL database schema version {database_version} is newer than "
                f"supported version {SCHEMA_VERSION}"
            )

        for version, columns in _MIGRATIONS:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                for table, column, column_type in columns:
                    cursor = await conn.execute(f"PRAGMA table_info({table})")
                    existing = {item["name"] for item in await cursor.fetchall()}
                    if column not in existing:
                        await conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                        )
                await conn.execute(
                    "INSERT OR IGNORE INTO rsl_schema_version(version) VALUES (?)",
                    (version,),
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def record_decision(self, decision: dict[str, Any]) -> bool:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            INSERT INTO rsl_decisions (
                request_id, created_at, api_key_id, subject_type, subject_id,
                user_id, routing_profile_id, requested_model, baseline_model,
                selected_model, task_type, features_json, mode, behavior_policy,
                policy_version, propensity, applied, eligible, skip_reason,
                proposed_model, rollout_bucket, action_distribution_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO NOTHING
            """,
            (
                decision["request_id"],
                decision.get("created_at", _utc_now()),
                decision.get("api_key_id", 0),
                decision["subject_type"],
                decision["subject_id"],
                decision.get("user_id"),
                decision.get("routing_profile_id"),
                decision["requested_model"],
                decision["baseline_model"],
                decision["selected_model"],
                decision["task_type"],
                json.dumps(decision.get("features", {}), ensure_ascii=False, sort_keys=True),
                decision.get("mode", "baseline"),
                decision.get("behavior_policy", "baseline_only"),
                decision.get("policy_version"),
                float(decision.get("propensity", 1.0)),
                int(bool(decision.get("applied", False))),
                int(bool(decision.get("eligible", True))),
                decision.get("skip_reason"),
                decision.get("proposed_model"),
                decision.get("rollout_bucket"),
                json.dumps(
                    decision.get("action_distribution", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def record_outcome(self, outcome: dict[str, Any]) -> bool:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            INSERT INTO rsl_outcomes (
                request_id, created_at, executed_model, status, error_type,
                prompt_tokens, completion_tokens, total_tokens, actual_cost,
                latency_ms, finish_reason, empty_response, judge_score,
                quality_score, quality_source, fallback_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO NOTHING
            """,
            (
                outcome["request_id"],
                outcome.get("created_at", _utc_now()),
                outcome.get("executed_model"),
                outcome["status"],
                outcome.get("error_type"),
                outcome.get("prompt_tokens"),
                outcome.get("completion_tokens"),
                outcome.get("total_tokens"),
                outcome.get("actual_cost"),
                outcome.get("latency_ms"),
                outcome.get("finish_reason"),
                None
                if outcome.get("empty_response") is None
                else int(bool(outcome["empty_response"])),
                outcome.get("judge_score"),
                outcome.get("quality_score"),
                outcome.get("quality_source"),
                outcome.get("fallback_reason"),
            ),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def get_decision(self, request_id: str) -> dict[str, Any] | None:
        conn = self._require_connection()
        cursor = await conn.execute(
            "SELECT * FROM rsl_decisions WHERE request_id=?",
            (request_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def save_feedback(
        self,
        *,
        request_id: str,
        subject_type: str,
        subject_id: str,
        rating: float | None,
        label: str | None,
        source: str,
    ) -> None:
        conn = self._require_connection()
        await conn.execute(
            """
            INSERT INTO rsl_feedback (
                request_id, subject_type, subject_id, rating, label, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id, subject_type, subject_id) DO UPDATE SET
                rating=excluded.rating,
                label=excluded.label,
                source=excluded.source,
                created_at=excluded.created_at
            """,
            (
                request_id,
                subject_type,
                subject_id,
                rating,
                label,
                source,
                _utc_now(),
            ),
        )
        await conn.commit()

    async def get_training_rows(
        self,
        *,
        window_start: str,
        window_end: str,
        outcomes_before: str,
    ) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT
                d.request_id, d.created_at, d.subject_type, d.subject_id,
                d.baseline_model, d.selected_model,
                d.proposed_model, d.mode, d.applied, d.policy_version,
                d.task_type, d.features_json, d.behavior_policy, d.propensity,
                d.action_distribution_json,
                o.executed_model, o.status, o.error_type, o.prompt_tokens,
                o.completion_tokens, o.total_tokens, o.actual_cost, o.latency_ms,
                o.finish_reason, o.empty_response, o.judge_score, o.fallback_reason,
                (
                    SELECT f.rating FROM rsl_feedback AS f
                    WHERE f.request_id=d.request_id
                    ORDER BY f.created_at DESC, f.id DESC LIMIT 1
                ) AS feedback_rating,
                (
                    SELECT f.label FROM rsl_feedback AS f
                    WHERE f.request_id=d.request_id
                    ORDER BY f.created_at DESC, f.id DESC LIMIT 1
                ) AS feedback_label,
                (
                    SELECT f.source FROM rsl_feedback AS f
                    WHERE f.request_id=d.request_id
                    ORDER BY f.created_at DESC, f.id DESC LIMIT 1
                ) AS feedback_source
            FROM rsl_decisions AS d
            JOIN rsl_outcomes AS o ON o.request_id=d.request_id
            WHERE d.eligible=1
              AND d.created_at >= ?
              AND d.created_at <= ?
              AND o.created_at <= ?
            ORDER BY d.created_at, d.request_id
            """,
            (window_start, window_end, outcomes_before),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def create_dataset(
        self,
        *,
        dataset_id: str,
        window_start: str,
        window_end: str,
        filters: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> None:
        conn = self._require_connection()
        await conn.execute(
            """
            INSERT INTO rsl_datasets (
                dataset_id, created_at, window_start, window_end,
                sample_count, filters_json, samples_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                _utc_now(),
                window_start,
                window_end,
                len(samples),
                json.dumps(filters, ensure_ascii=False, sort_keys=True),
                json.dumps(samples, ensure_ascii=False, sort_keys=True),
            ),
        )
        await conn.commit()

    async def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        conn = self._require_connection()
        cursor = await conn.execute(
            "SELECT * FROM rsl_datasets WHERE dataset_id=?",
            (dataset_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        result["filters"] = json.loads(result.pop("filters_json"))
        result["samples"] = json.loads(result.pop("samples_json"))
        return result

    async def save_policy(
        self,
        *,
        policy_version: str,
        parent_version: str | None,
        dataset_id: str,
        policy: dict[str, Any],
        metrics: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> None:
        conn = self._require_connection()
        await conn.execute(
            """
            INSERT INTO rsl_policies (
                policy_version, parent_version, dataset_id, created_at,
                status, policy_json, metrics_json, evaluation_json
            ) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?)
            """,
            (
                policy_version,
                parent_version,
                dataset_id,
                _utc_now(),
                json.dumps(policy, ensure_ascii=False, sort_keys=True),
                json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                json.dumps(evaluation, ensure_ascii=False, sort_keys=True),
            ),
        )
        await conn.commit()

    async def get_policy(self, policy_version: str) -> dict[str, Any] | None:
        conn = self._require_connection()
        cursor = await conn.execute(
            "SELECT * FROM rsl_policies WHERE policy_version=?",
            (policy_version,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        for source, target in (
            ("policy_json", "policy"),
            ("metrics_json", "metrics"),
            ("evaluation_json", "evaluation"),
        ):
            raw = result.pop(source)
            result[target] = json.loads(raw) if raw is not None else None
        return result

    async def latest_policy_version(self) -> str | None:
        conn = self._require_connection()
        cursor = await conn.execute(
            "SELECT policy_version FROM rsl_policies ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return str(row["policy_version"]) if row else None

    async def list_policies(self, *, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            """
            SELECT policy_version, parent_version, dataset_id, created_at, status,
                   metrics_json, evaluation_json
            FROM rsl_policies
            ORDER BY created_at DESC, policy_version DESC
            LIMIT ?
            """,
            (max(1, min(200, int(limit))),),
        )
        rows = []
        for row in await cursor.fetchall():
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            raw_eval = item.pop("evaluation_json")
            item["evaluation"] = json.loads(raw_eval) if raw_eval is not None else None
            rows.append(item)
        return rows

    async def get_rollout_state(self) -> dict[str, Any] | None:
        conn = self._require_connection()
        cursor = await conn.execute("SELECT * FROM rsl_rollout_state WHERE id=1")
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def set_rollout_state(
        self,
        *,
        stage: str,
        policy_version: str | None,
        canary_percent: float,
        previous_policy_version: str | None,
        note: str | None,
    ) -> dict[str, Any]:
        conn = self._require_connection()
        updated_at = _utc_now()
        await conn.execute(
            """
            INSERT INTO rsl_rollout_state (
                id, stage, policy_version, canary_percent,
                previous_policy_version, updated_at, note, revision
            ) VALUES (1, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(id) DO UPDATE SET
                stage=excluded.stage,
                policy_version=excluded.policy_version,
                canary_percent=excluded.canary_percent,
                previous_policy_version=excluded.previous_policy_version,
                updated_at=excluded.updated_at,
                note=excluded.note,
                revision=rsl_rollout_state.revision + 1
            """,
            (stage, policy_version, float(canary_percent), previous_policy_version, updated_at, note),
        )
        await conn.commit()
        cursor = await conn.execute("SELECT revision FROM rsl_rollout_state WHERE id=1")
        row = await cursor.fetchone()
        return {
            "id": 1,
            "stage": stage,
            "policy_version": policy_version,
            "canary_percent": float(canary_percent),
            "previous_policy_version": previous_policy_version,
            "updated_at": updated_at,
            "note": note,
            "revision": int(row["revision"]),
        }

    async def commit_rollout_transition(
        self,
        *,
        expected_revision: int,
        stage: str,
        policy_version: str | None,
        canary_percent: float,
        previous_policy_version: str | None,
        note: str | None,
        policy_status_updates: list[tuple[str, str]],
        action: str,
        from_stage: str,
        from_policy_version: str | None,
        actor: str | None,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically persist a rollout transition and its audit record."""
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        try:
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("BEGIN IMMEDIATE")
            cursor = await conn.execute(
                "SELECT revision FROM rsl_rollout_state WHERE id=1"
            )
            row = await cursor.fetchone()
            actual_revision = int(row["revision"]) if row else 0
            if actual_revision != expected_revision:
                raise RolloutRevisionConflict(
                    f"Rollout changed from revision {expected_revision} to {actual_revision}"
                )

            updated_at = _utc_now()
            if row:
                await conn.execute(
                    """
                    UPDATE rsl_rollout_state
                    SET stage=?, policy_version=?, canary_percent=?,
                        previous_policy_version=?, updated_at=?, note=?,
                        revision=revision+1
                    WHERE id=1 AND revision=?
                    """,
                    (
                        stage,
                        policy_version,
                        float(canary_percent),
                        previous_policy_version,
                        updated_at,
                        note,
                        expected_revision,
                    ),
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO rsl_rollout_state (
                        id, stage, policy_version, canary_percent,
                        previous_policy_version, updated_at, note, revision
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        stage,
                        policy_version,
                        float(canary_percent),
                        previous_policy_version,
                        updated_at,
                        note,
                    ),
                )

            for version, status in policy_status_updates:
                await conn.execute(
                    "UPDATE rsl_policies SET status=? WHERE policy_version=?",
                    (status, version),
                )
            await conn.execute(
                """
                INSERT INTO rsl_audit_log (
                    created_at, action, policy_version, from_stage,
                    to_stage, actor, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    updated_at,
                    action,
                    policy_version or from_policy_version,
                    from_stage,
                    stage,
                    actor,
                    json.dumps(details, ensure_ascii=False, sort_keys=True),
                ),
            )
            await conn.commit()
            return {
                "id": 1,
                "stage": stage,
                "policy_version": policy_version,
                "canary_percent": float(canary_percent),
                "previous_policy_version": previous_policy_version,
                "updated_at": updated_at,
                "note": note,
                "revision": expected_revision + 1,
            }
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close()

    async def list_audit(self, *, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._require_connection()
        cursor = await conn.execute(
            "SELECT * FROM rsl_audit_log ORDER BY id DESC LIMIT ?",
            (max(1, min(500, int(limit))),),
        )
        rows = []
        for row in await cursor.fetchall():
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            rows.append(item)
        return rows

    async def decision_summary(
        self,
        *,
        policy_version: str | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Operational counts per mode; not a causal quality comparison."""
        conn = self._require_connection()
        clauses = []
        params: list[Any] = []
        if policy_version:
            clauses.append("d.policy_version=?")
            params.append(policy_version)
        if since:
            clauses.append("d.created_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"""
            SELECT
                d.mode,
                d.applied,
                COUNT(*) AS decisions,
                SUM(CASE WHEN d.proposed_model IS NOT NULL
                          AND d.proposed_model != d.baseline_model THEN 1 ELSE 0 END)
                    AS proposed_changes,
                SUM(CASE WHEN o.request_id IS NOT NULL THEN 1 ELSE 0 END) AS outcomes,
                SUM(CASE WHEN o.status = 'success' THEN 1 ELSE 0 END) AS successes,
                SUM(CASE WHEN o.fallback_reason IS NOT NULL THEN 1 ELSE 0 END) AS fallbacks,
                AVG(o.latency_ms) AS avg_latency_ms,
                AVG(o.actual_cost) AS avg_actual_cost
            FROM rsl_decisions AS d
            LEFT JOIN rsl_outcomes AS o ON o.request_id = d.request_id
            {where}
            GROUP BY d.mode, d.applied
            ORDER BY d.mode, d.applied
            """,
            params,
        )
        groups = []
        for row in await cursor.fetchall():
            item = dict(row)
            item["applied"] = bool(item["applied"])
            outcomes = item.get("outcomes") or 0
            item["failure_rate"] = (
                (outcomes - (item.get("successes") or 0)) / outcomes if outcomes else None
            )
            groups.append(item)
        return {"policy_version": policy_version, "since": since, "groups": groups}

    def _require_connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("RSLStore is not connected")
        return self._conn
