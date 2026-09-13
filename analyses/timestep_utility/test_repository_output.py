import argparse
import unittest

from analyses.timestep_utility.repository_output import (
    PROJECT_ROOT,
    repository_output_dir,
)


class RepositoryOutputDirTests(unittest.TestCase):
    def test_accepts_git_ignored_directories_inside_the_repository(self):
        for relative in (
            "outputs/unit-test-run",
            "analyses/archvied_analyses/unit-test-run",
        ):
            with self.subTest(relative=relative):
                path = PROJECT_ROOT / relative
                self.assertEqual(repository_output_dir(str(path)), path)

    def test_rejects_directories_outside_the_repository(self):
        for value in (
            "/tmp/unit-test-run",
            str(PROJECT_ROOT),
            str(PROJECT_ROOT.parent / "unit-test-run"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError,
                    "inside the repository",
                ):
                    repository_output_dir(value)

    def test_rejects_tracked_directories_inside_the_repository(self):
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "git-ignored"):
            repository_output_dir(str(PROJECT_ROOT / "analyses" / "unit-test-run"))


if __name__ == "__main__":
    unittest.main()
