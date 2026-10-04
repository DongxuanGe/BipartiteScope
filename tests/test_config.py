import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from bipartite_scope.config import ServiceSettings, load_service_settings


class ServiceSettingsTests(unittest.TestCase):
    def test_configuration_precedence_and_secret_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "service.toml"
            dotenv = root / ".env"
            config.write_text(
                """
[service]
port = 7001
workspace_root = "from-toml"

[database]
url = "postgresql+psycopg://service:toml-secret@db:5432/scope"
""".strip(),
                encoding="utf-8",
            )
            dotenv.write_text(
                "BIPARTITE_SCOPE_PORT=7002\nBIPARTITE_SCOPE_WORKER_CONCURRENCY=3\n",
                encoding="utf-8",
            )
            settings = load_service_settings(
                config,
                env_path=dotenv,
                environment={
                    "BIPARTITE_SCOPE_PORT": "7003",
                    "BIPARTITE_SCOPE_DATABASE_URL": (
                        "postgresql+psycopg://service:environment-secret@db:5432/scope"
                    ),
                },
                overrides={"service": {"port": 7004}},
            )

            self.assertEqual(settings.service.port, 7004)
            self.assertEqual(settings.worker.concurrency, 3)
            self.assertEqual(settings.service.workspace_root, Path("from-toml"))
            public = settings.public_dict()
            self.assertEqual(
                public["database"]["url"],
                "postgresql+psycopg://service:***@db:5432/scope",
            )
            self.assertNotIn("environment-secret", repr(settings))
            self.assertEqual(len(settings.fingerprint()), 64)

    def test_remote_bind_requires_explicit_insecure_opt_in(self) -> None:
        with self.assertRaisesRegex(ValidationError, "allow_insecure_remote"):
            ServiceSettings.model_validate({"service": {"host": "0.0.0.0"}})

        settings = ServiceSettings.model_validate(
            {
                "service": {
                    "host": "0.0.0.0",
                    "allow_insecure_remote": True,
                }
            }
        )
        self.assertEqual(settings.service.host, "0.0.0.0")

    def test_invalid_urls_and_unknown_keys_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ServiceSettings.model_validate({"database": {"url": "mysql://db/scope"}})
        with self.assertRaises(ValidationError):
            ServiceSettings.model_validate({"service": {"unknown": True}})

    def test_v4_operation_configuration_precedence_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = root / "service.toml"
            configuration.write_text(
                "[resources]\nmax_running_heavy_jobs=2\n"
                "[retention]\ncandidate_snapshots=8\n"
                "[backup]\ninclude_reports=true\n"
                "[observability]\nbackup_max_age_seconds=3600\n",
                encoding="utf-8",
            )
            settings = load_service_settings(
                configuration,
                environment={
                    "BIPARTITE_SCOPE_RESOURCES__MAX_RUNNING_HEAVY_JOBS": "3",
                    "BIPARTITE_SCOPE_BACKUP__INCLUDE_REPORTS": "false",
                    "BIPARTITE_SCOPE_WORKER__HEARTBEAT_SECONDS": "2",
                },
                overrides={"retention": {"candidate_snapshots": 4}},
            )
            self.assertEqual(settings.resources.max_running_heavy_jobs, 3)
            self.assertEqual(settings.retention.candidate_snapshots, 4)
            self.assertFalse(settings.backup.include_reports)
            self.assertEqual(settings.observability.backup_max_age_seconds, 3600)
            self.assertEqual(settings.worker.heartbeat_seconds, 2)
            with self.assertRaises(ValidationError):
                settings.resources.max_running_heavy_jobs = 5
        for payload in (
            {"resources": {"max_running_heavy_jobs": 0}},
            {"retention": {"backups_keep": 0}},
            {"backup": {"wait_seconds": 0}},
            {"observability": {"log_max_bytes": 1}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                ServiceSettings.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
