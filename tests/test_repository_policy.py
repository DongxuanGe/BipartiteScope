import unittest
from pathlib import Path
from subprocess import check_output


class RepositoryPolicyTests(unittest.TestCase):
    def test_consolidated_source_and_documentation_layout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = root / "src" / "bipartite_scope"
        self.assertEqual(
            {path.relative_to(source).as_posix() for path in source.rglob("*.py")},
            {
                "__init__.py",
                "api.py",
                "config.py",
                "core.py",
                "database.py",
                "interface.py",
                "maintenance.py",
                "observability.py",
                "policies.py",
                "recommendation.py",
                "reliability.py",
                "storage.py",
                "tasks.py",
            },
        )
        self.assertFalse((root / "docs").exists())
        self.assertTrue((root / "DOCUMENTATION.md").is_file())
        self.assertEqual(len(list(root.rglob("DOCUMENTATION.md"))), 1)

    def test_repository_text_is_english_and_contains_no_dataset_files(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tracked = check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
        ).splitlines()
        datasets = [name for name in tracked if Path(name).suffix in {".csv", ".jsonl", ".ndjson"}]
        self.assertEqual(datasets, [])
        offenders = []
        for name in tracked:
            path = root / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(
                0x3400 <= ord(character) <= 0x4DBF
                or 0x4E00 <= ord(character) <= 0x9FFF
                or 0xF900 <= ord(character) <= 0xFAFF
                or 0x20000 <= ord(character) <= 0x323AF
                for character in text
            ):
                offenders.append(name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
