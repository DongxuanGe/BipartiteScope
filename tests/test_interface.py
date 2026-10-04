import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from bipartite_scope.interface import build_parser, run


class InterfaceTests(unittest.TestCase):
    def execute(self, *arguments: str) -> tuple[int, dict]:
        stream = StringIO()
        with redirect_stdout(stream):
            code = run(build_parser().parse_args(list(arguments)))
        return code, json.loads(stream.getvalue())

    def test_complete_cli_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            code, initialized = self.execute("init", str(workspace))
            self.assertEqual(code, 0)
            self.assertEqual(initialized["workspace"], str(workspace.resolve()))
            (workspace / "data" / "edges.csv").write_text(
                "u_id,v_id\nu1,i1\nu1,i2\nu1,i3\nu2,i1\nu2,i3\nu2,i4\nu3,i2\nu3,i4\n",
                encoding="utf-8",
            )
            (workspace / "data" / "features.csv").write_text(
                "u_id,f1,f2\nu1,1,0\nu2,1,0\nu3,0,1\n",
                encoding="utf-8",
            )
            config = workspace / "bipartitescope.toml"
            text = config.read_text(encoding="utf-8")
            text = text.replace("hidden_dim = 64", "hidden_dim = 4")
            text = text.replace("layers = 2", "layers = 1")
            text = text.replace("latent_groups = 8", "latent_groups = 2")
            text = text.replace("epochs = 100", "epochs = 1")
            text = text.replace("warm_start_epochs = 20", "warm_start_epochs = 1")
            config.write_text(text, encoding="utf-8")
            code, validation = self.execute("validate", "--workspace", str(workspace))
            self.assertEqual(code, 0)
            self.assertTrue(validation["valid"])
            code, built = self.execute("build", "--workspace", str(workspace))
            self.assertEqual(code, 0)
            initial_id = built["snapshot_id"]
            self.assertTrue(built["active"])
            code, query = self.execute(
                "query",
                "--workspace",
                str(workspace),
                "--snapshot",
                initial_id,
                "--entity",
                "u1",
                "--size",
                "2",
            )
            self.assertEqual(code, 0)
            self.assertIn("u1", query["members"])
            code, exported = self.execute(
                "export",
                "--workspace",
                str(workspace),
                "--entity",
                "u1",
                "--name",
                "result.json",
            )
            self.assertEqual(code, 0)
            self.assertTrue(Path(exported["output"]).is_file())
            code, feedback = self.execute(
                "feedback",
                "--workspace",
                str(workspace),
                "--user",
                "u1",
                "--item",
                "i4",
                "--event-type",
                "dislike",
                "--event-id",
                "feedback-1",
                "--event-time",
                "2026-09-03T12:00:00+08:00",
            )
            self.assertEqual(code, 0)
            self.assertTrue(feedback["accepted"])
            events = Path(directory) / "events.jsonl"
            records = [
                ("e1", "u1", "i1", "2026-09-01T00:00:00Z"),
                ("e2", "u1", "i2", "2026-09-02T00:00:00Z"),
                ("e3", "u1", "i3", "2026-09-03T00:00:00Z"),
                ("e4", "u2", "i1", "2026-09-01T00:00:00Z"),
                ("e5", "u2", "i3", "2026-09-02T00:00:00Z"),
                ("e6", "u2", "i4", "2026-09-03T00:00:00Z"),
            ]
            events.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "event_id": event_id,
                            "user_id": user,
                            "item_id": item,
                            "event_type": "click",
                            "event_time": timestamp,
                        }
                    )
                    for event_id, user, item, timestamp in records
                )
                + "\n",
                encoding="utf-8",
            )
            code, updated = self.execute(
                "update",
                "--workspace",
                str(workspace),
                "--events",
                str(events),
            )
            self.assertEqual(code, 0)
            candidate_id = updated["snapshot_id"]
            self.assertFalse(updated["active"])
            code, recommendation = self.execute(
                "recommend",
                "--workspace",
                str(workspace),
                "--user",
                "u1",
                "--top-n",
                "2",
            )
            self.assertEqual(code, 0)
            self.assertEqual(recommendation["snapshot_id"], initial_id)
            code, evaluation = self.execute("evaluate", "--workspace", str(workspace), "--k", "2")
            self.assertEqual(code, 0)
            self.assertTrue(Path(evaluation["output"]).is_dir())
            code, snapshots = self.execute("snapshot", "list", "--workspace", str(workspace))
            self.assertEqual(code, 0)
            self.assertEqual(len(snapshots["snapshots"]), 2)
            code, verified = self.execute(
                "snapshot",
                "verify",
                "--workspace",
                str(workspace),
                "--snapshot",
                candidate_id,
            )
            self.assertEqual(code, 0)
            self.assertTrue(verified["verified"])
            self.execute(
                "snapshot",
                "activate",
                "--workspace",
                str(workspace),
                "--snapshot",
                candidate_id,
            )
            code, rollback = self.execute(
                "snapshot",
                "rollback",
                "--workspace",
                str(workspace),
                "--snapshot",
                initial_id,
            )
            self.assertEqual(code, 0)
            self.assertTrue(rollback["rollback"])

    def test_generated_sparse_benchmark(self) -> None:
        code, result = self.execute(
            "benchmark",
            "--users",
            "4",
            "--items",
            "5",
            "--events",
            "8",
            "--delta-ratio",
            "0.1",
        )
        self.assertEqual(code, 0)
        self.assertFalse(result["dense_user_by_user_matrix_materialized"])
        self.assertGreaterEqual(result["full_build_seconds"], 0)
        self.assertGreaterEqual(result["incremental_build_seconds"], 0)

    def test_scheduler_uses_the_celery_beat_command(self) -> None:
        application = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "workspaces"
            with (
                patch("bipartite_scope.tasks.configure_worker", return_value=application),
                patch.dict(
                    "os.environ",
                    {
                        "BIPARTITE_SCOPE_WORKSPACE_ROOT": str(root),
                        "BIPARTITE_SCOPE_DATABASE_URL": f"sqlite+pysqlite:///{Path(directory).resolve() / 'service.sqlite3'}",
                    },
                ),
            ):
                code = run(build_parser().parse_args(["scheduler"]))
            self.assertEqual(code, 0)
            application.start.assert_called_once_with(
                [
                    "beat",
                    "--loglevel=INFO",
                    f"--schedule={root / '.operations' / 'celerybeat-schedule'}",
                ]
            )

    def test_operations_and_backup_cli_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspaces"
            workspace = root / "demo"
            configuration = base / "service.toml"
            database_url = f"sqlite+pysqlite:///{base / 'service.sqlite3'}"
            configuration.write_text(
                f"[service]\nworkspace_root={json.dumps(str(root))}\n"
                f"[database]\nurl={json.dumps(database_url)}\n"
                f"[backup]\nroot={json.dumps(str(base / 'backups'))}\n"
                "[resources]\nmin_free_bytes=0\n",
                encoding="utf-8",
            )
            self.execute("init", str(workspace))
            self.execute(
                "workspace",
                "--config",
                str(configuration),
                "register",
                "--workspace",
                str(workspace),
                "--id",
                "demo",
            )
            code, summary = self.execute("operations", "--config", str(configuration), "summary")
            self.assertEqual(code, 0)
            self.assertEqual(summary["workspace_count"], 1)
            code, backup = self.execute(
                "backup", "--config", str(configuration), "create", "--workspace", str(workspace)
            )
            self.assertEqual(code, 0)
            self.assertEqual(backup["status"], "succeeded")
            code, listed = self.execute("backup", "--config", str(configuration), "list")
            self.assertEqual(code, 0)
            self.assertEqual(listed["backups"][0]["backup_id"], backup["backup_id"])
            for operation in ("verify", "drill"):
                code, checked = self.execute(
                    "backup", "--config", str(configuration), operation, "--backup", backup["path"]
                )
                self.assertEqual(code, 0)
                self.assertTrue(checked["verified"])
            code, restored = self.execute(
                "backup",
                "--config",
                str(configuration),
                "restore",
                "--backup",
                backup["path"],
                "--target",
                str(base / "restored"),
            )
            self.assertEqual(code, 0)
            self.assertTrue(restored["restored"])
            self.assertTrue((base / "restored" / "bipartitescope.toml").is_file())
            code, audit = self.execute("operations", "--config", str(configuration), "audit")
            self.assertEqual(code, 0)
            actions = {entry["payload"]["action"] for entry in audit["records"]}
            self.assertTrue(
                {"backup.create", "backup.verify", "backup.drill", "backup.restore"} <= actions
            )


if __name__ == "__main__":
    unittest.main()
