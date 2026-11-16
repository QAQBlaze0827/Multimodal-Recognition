import contextlib
import io
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from start_sender import load_or_setup, main, validate

TOKEN = "a" * 64  # Synthetic fixture, not a real credential.


class ValidationTests(unittest.TestCase):
    def test_valid(self):
        validate({"laptop_ip": "100.74.11.81", "port": 8765, "token": TOKEN})

    def test_invalid_settings_never_echo_secret(self):
        cases = [None, [], {}, {"laptop_ip": "192.168.1.1", "port": 8765, "token": TOKEN},
                 {"laptop_ip": "100.74.11.81", "port": 8000, "token": TOKEN},
                 {"laptop_ip": "100.74.11.81", "port": 8765, "token": "private-secret"}]
        for settings in cases:
            with self.subTest(settings_type=type(settings).__name__), self.assertRaises(ValueError) as error:
                validate(settings)
            self.assertNotIn(TOKEN, str(error.exception))
            self.assertNotIn("private-secret", str(error.exception))


@unittest.skipUnless(sys.platform == "linux", "Actual POSIX ownership/mode tests run under Linux/WSL")
class SenderLauncherTests(unittest.TestCase):
    def setUp(self):
        # Preserve synthetic fixtures, per this project's no-delete constraint.
        self.home = Path(tempfile.mkdtemp(prefix="voice-launcher-test-"))
        self.path = self.home / ".config" / "multimodal-voice" / "sender.json"
        self.environment = patch.dict(os.environ, {"COMPANION_LAPTOP_IP": "100.74.11.81", "COMPANION_AUDIO_TOKEN": TOKEN})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_reuses_environment_and_saves_private_permissions(self):
        with patch("builtins.input", side_effect=AssertionError("unexpected prompt")), patch("getpass.getpass", side_effect=AssertionError("unexpected token prompt")), contextlib.redirect_stdout(io.StringIO()) as output:
            settings = load_or_setup(self.path)
        self.assertEqual(settings["token"], TOKEN)
        self.assertNotIn(TOKEN, output.getvalue())
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)

    def test_second_launch_uses_saved_values_without_prompts_or_overwrite(self):
        load_or_setup(self.path)
        before = self.path.read_bytes()
        with patch.dict(os.environ, {"COMPANION_LAPTOP_IP": "100.99.1.2", "COMPANION_AUDIO_TOKEN": "b" * 64}), patch("builtins.input", side_effect=AssertionError), patch("getpass.getpass", side_effect=AssertionError):
            settings = load_or_setup(self.path)
        self.assertEqual(settings["token"], TOKEN)
        self.assertEqual(settings["laptop_ip"], "100.74.11.81")
        self.assertEqual(self.path.read_bytes(), before)

    def test_only_missing_values_prompt(self):
        with patch.dict(os.environ, {"COMPANION_LAPTOP_IP": "", "COMPANION_AUDIO_TOKEN": ""}), patch("builtins.input", return_value="100.74.11.81") as address, patch("getpass.getpass", return_value=TOKEN) as token:
            load_or_setup(self.path)
        address.assert_called_once()
        token.assert_called_once()

    def test_invalid_environment_creates_no_file(self):
        with patch.dict(os.environ, {"COMPANION_AUDIO_TOKEN": "wrong"}), self.assertRaises(ValueError):
            load_or_setup(self.path)
        self.assertFalse(self.path.exists())

    def test_corrupt_file_preserved_without_secret_in_error(self):
        load_or_setup(self.path)
        self.path.write_text(TOKEN, encoding="utf-8")
        with self.assertRaises(ValueError) as error:
            load_or_setup(self.path)
        self.assertNotIn(TOKEN, str(error.exception))
        self.assertEqual(self.path.read_text(), TOKEN)

    def test_unsafe_file_mode_rejected_without_chmod(self):
        load_or_setup(self.path)
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):
            load_or_setup(self.path)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o644)

    def test_symlinks_rejected(self):
        self.path.parent.mkdir(parents=True, mode=0o700)
        target = self.home / "untouched"
        target.write_text("unchanged", encoding="utf-8")
        self.path.symlink_to(target)
        with self.assertRaises(ValueError):
            load_or_setup(self.path)
        self.assertEqual(target.read_text(), "unchanged")

    def test_setup_only_opens_no_process(self):
        with patch("start_sender.Path.home", return_value=self.home), patch("start_sender.os.execve") as execute:
            self.assertEqual(main(["--setup-only"]), 0)
        execute.assert_not_called()

    def test_exec_uses_project_root_and_no_token_in_arguments(self):
        with patch("start_sender.Path.home", return_value=self.home), patch("start_sender.os.execve") as execute, patch("start_sender.os.chdir") as chdir:
            main([])
        executable, arguments, env = execute.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(env["COMPANION_AUDIO_TOKEN"], TOKEN)
        self.assertNotIn(TOKEN, " ".join(arguments))
        self.assertIn("--audio-only", arguments)
        self.assertEqual(arguments[-1], "http://100.74.11.81:8765")
        self.assertEqual(Path(arguments[2]).parent, chdir.call_args.args[0])


@unittest.skipUnless(sys.platform == "linux", "Source wrapper test requires bash")
class ShellLauncherTests(unittest.TestCase):
    def test_source_inherits_unexported_ip_and_does_not_exit_or_move_parent_shell(self):
        root = Path(tempfile.mkdtemp(prefix="voice-shell-test-")) / "project with spaces"
        tool = root / "tools" / "voice_companion"
        tool.mkdir(parents=True)
        binary = root / ".venv" / "bin"
        binary.mkdir(parents=True)
        (binary / "python").symlink_to(sys.executable)
        wrapper = tool / "start_sender.sh"
        shutil.copyfile(Path(__file__).with_name("start_sender.sh"), wrapper)
        # A synthetic child checks the launch contract instead of opening a mic.
        (tool / "start_sender.py").write_text(
            "import os, sys\n"
            "assert os.environ['COMPANION_LAPTOP_IP'] == '100.74.11.81'\n"
            "assert os.environ['COMPANION_AUDIO_TOKEN'] == 'a' * 64\n"
            "assert sys.argv[1:] == ['--setup-only']\n"
            "print('CHILD_OK')\n", encoding="utf-8")
        command = 'before="$PWD"; laptop_ip=100.74.11.81; . "$1" --setup-only; result=$?; test "$PWD" = "$before" || exit 9; printf "PARENT_ALIVE\\n"; exit "$result"'
        result = subprocess.run(["bash", "-c", command, "test", str(wrapper)],
                                env=dict(os.environ, COMPANION_AUDIO_TOKEN=TOKEN),
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CHILD_OK", result.stdout)
        self.assertIn("PARENT_ALIVE", result.stdout)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
