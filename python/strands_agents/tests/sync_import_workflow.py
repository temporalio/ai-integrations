from strands import Agent

from temporalio import workflow
from tests.mock_model import MockModel

# Module-level sync call: the sandbox runs it again when it imports this module
# for each Workflow run, before the Workflow runtime is attached.
try:
    import_result = str(
        Agent(model=MockModel(["import answer"]), callback_handler=None)("Hello")
    ).strip()
except RuntimeError as err:
    import_result = f"refused: {err}"


@workflow.defn
class SyncImportWorkflow:
    @workflow.run
    async def run(self) -> str:
        return import_result
