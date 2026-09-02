"""QC stage — run quality checks on final render."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

from .contracts import PipelineStage, StageConfig, StageResult, STAGE_CONTRACTS
from ..manifest import ProjectManifest


class QCStage(PipelineStage):
    """Run quality checks on pipeline outputs."""
    
    def run(self, stage_config: StageConfig) -> StageResult:
        self.mark_running()
        
        try:
            project_id = self.manifest.project_id
            project_dir = self.artifact_root / project_id
            
            # Run QC
            from qc.critic import run_qc
            
            report = run_qc(project_dir)
            print(f"QC checks: {report.get('checks')}")
            
            # Save pipeline status
            from quality.pipeline_status import save_pipeline_status
            
            pstatus = save_pipeline_status(project_dir)
            print(f"Pipeline status: {pstatus.status}")
            if pstatus.reasons:
                for reason in pstatus.reasons:
                    print(f"  - {reason}")
            
            # Register artifacts
            artifact_ids = []
            
            qc_report_path = project_dir / "reports" / "qc_report.json"
            if qc_report_path.exists():
                artifact_id = self.register_output(
                    artifact_type="qc_report",
                    relative_path=f"{project_id}/reports/qc_report.json",
                )
                artifact_ids.append(artifact_id)
            
            pipeline_status_path = project_dir / "reports" / "pipeline_status.json"
            if pipeline_status_path.exists():
                artifact_id = self.register_output(
                    artifact_type="pipeline_status",
                    relative_path=f"{project_id}/reports/pipeline_status.json",
                )
                artifact_ids.append(artifact_id)
            
            return StageResult(
                success=True,
                stage_name=self.contract.name,
                output_artifact_ids=artifact_ids,
                metrics={
                    "status": pstatus.status,
                    "checks": report.get("checks"),
                    "reasons": pstatus.reasons,
                },
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                stage_name=self.contract.name,
                error=str(e),
            )


QCStage = QCStage