"""Regression checks for missing inputs and failed environment activation."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hommi_repro"))


class InputValidationTests(unittest.TestCase):
    def test_laundry_requires_recorded_depth(self):
        from laundry_episodes import load_laundry_episodes
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("laundry_episodes.load_plan_episodes", return_value=[]):
            with self.assertRaisesRegex(FileNotFoundError, "replay_buffer"):
                load_laundry_episodes(directory)

    def test_video_requires_every_specified_clip(self):
        path = ROOT / "scripts/make_demo_video.py"
        spec = importlib.util.spec_from_file_location("make_demo_video", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "spec.txt"
            missing = Path(directory) / "missing.mp4"
            config.write_text(f"clip|{missing}|required clip|\n")
            with mock.patch.object(sys, "argv", [str(path), "--spec", str(config),
                                                "--out", str(Path(directory) / "out.mp4")]), \
                    mock.patch.object(module.imageio, "get_writer"), \
                    mock.patch.object(module.subprocess, "run"):
                with self.assertRaisesRegex(FileNotFoundError, "missing.mp4"):
                    module.main()

    def test_failed_conda_activation_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "etc/profile.d/conda.sh"
            profile.parent.mkdir(parents=True)
            profile.write_text(":\n")
            script = '''conda() {
if [ "$1" = info ]; then printf '%s\\n' "$EYECODE_TEST_CONDA_BASE"; else return 42; fi
}
source "$1"
'''
            env = dict(os.environ, EYECODE_TEST_CONDA_BASE=directory)
            result = subprocess.run(["bash", "--noprofile", "--norc", "-c", script, "_",
                                     str(ROOT / "scripts/hommi_env.sh")], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0, "Failed activation must not look successful")

    def test_failed_ros_setup_is_reported(self):
        script = 'source() { return 42; }; builtin source "$1"'
        result = subprocess.run(["bash", "--noprofile", "--norc", "-c", script, "_",
                                 str(ROOT.parent / "task1_rosorin/scripts/env.sh")], capture_output=True)
        self.assertNotEqual(result.returncode, 0, "Failed ROS setup must not look successful")


if __name__ == "__main__":
    unittest.main()
