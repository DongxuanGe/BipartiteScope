import tempfile
import unittest
from pathlib import Path

from bipartite_scope.config import ServiceSettings
from bipartite_scope.database import ServiceDatabase
from bipartite_scope.storage import init_workspace
from bipartite_scope.tasks import InlineDispatcher


class InlineTaskTests(unittest.TestCase):
    def test_inline_validation_uses_durable_job_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspaces"
            root.mkdir()
            workspace = root / "demo"
            init_workspace(workspace)
            (workspace / "data" / "edges.csv").write_text(
                "u_id,v_id\nu1,i1\nu1,i2\nu2,i1\n", encoding="utf-8"
            )
            (workspace / "data" / "features.csv").write_text(
                "u_id,f1,f2\nu1,1,0\nu2,0,1\n", encoding="utf-8"
            )
            settings = ServiceSettings.model_validate(
                {
                    "service": {"workspace_root": str(root)},
                    "database": {
                        "url": f"sqlite+pysqlite:///{Path(directory) / 'service.sqlite3'}"
                    },
                    "redis": {
                        "broker_url": "redis://127.0.0.1:1/0",
                        "result_url": "redis://127.0.0.1:1/1",
                        "progress_url": "redis://127.0.0.1:1/2",
                    },
                    "worker": {"always_eager": True},
                }
            )
            database = ServiceDatabase(settings)
            database.initialize()
            try:
                database.register_workspace("demo", workspace)
                job, created = database.create_job("demo", "validate", {})
                self.assertTrue(created)
                task_id = InlineDispatcher(settings, database).submit(job["job_id"])
                self.assertTrue(task_id.startswith("inline-"))
                completed = database.get_job(job["job_id"])
                self.assertEqual(completed["status"], "succeeded")
                self.assertEqual(completed["progress"], 100)
                self.assertTrue(completed["result"]["valid"])
                self.assertEqual(completed["attempt_count"], 1)
                self.assertEqual(completed["celery_task_id"], task_id)
                events = database.events_after(job["job_id"])
                self.assertEqual(events[0]["event"], "queued")
                self.assertEqual(events[-1]["event"], "succeeded")
            finally:
                database.dispose()


if __name__ == "__main__":
    unittest.main()
