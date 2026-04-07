"""Typed output models for every supervisor graph node.

Each model defines the exact subset of SupervisorState fields that a node
writes. ``extra="forbid"`` prevents writes to fields the node doesn't own.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthCheckOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    health_status: dict = {}
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> HealthCheckOutput:
        return cls(health_status={"overall": "unknown", "error": "node_unavailable"})


class DiagnoseIssuesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosed_issues: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> DiagnoseIssuesOutput:
        return cls(diagnosed_issues=[])


class ExecuteRecoveryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recovery_actions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ExecuteRecoveryOutput:
        return cls(recovery_actions=[])


class StrategyPipelineOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_pipeline_report: dict = {}
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> StrategyPipelineOutput:
        return cls(strategy_pipeline_report={"skipped": True, "reason": "node_unavailable"})


class StrategyLifecycleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_lifecycle_actions: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> StrategyLifecycleOutput:
        return cls(strategy_lifecycle_actions=[])


class ScheduledTasksOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduled_task_results: list[dict] = []
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> ScheduledTasksOutput:
        return cls(scheduled_task_results=[])


class EodDataSyncOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eod_refresh_summary: dict = {}
    errors: list[str] = []

    @classmethod
    def safe_default(cls) -> EodDataSyncOutput:
        return cls(eod_refresh_summary={"skipped": True, "reason": "node_unavailable"})
