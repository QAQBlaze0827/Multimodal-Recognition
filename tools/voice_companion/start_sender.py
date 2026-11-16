"""Pi-only launcher; preserve the existing audio-only command and private settings."""

import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import sys
import warnings


def validate(settings):
    try:
        address = ipaddress.IPv4Address(settings["laptop_ip"])
        if str(address) != settings["laptop_ip"] or address not in ipaddress.ip_network("100.64.0.0/10"):
            raise ValueError
        if settings["port"] != 8765 or isinstance(settings["port"], bool):
            raise ValueError
        if not re.fullmatch(r"[0-9a-fA-F]{64}", settings["token"]):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid settings: require a laptop Tailscale IPv4, port 8765 and the same 64-hex audio token") from None
    return settings


def check_private(path, directory=False):
    info = path.lstat()
    expected = 0o700 if directory else 0o600
    correct_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not correct_type or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != expected:
        raise ValueError(f"Settings must be owned by this user, not a link, with mode {expected:o}; stop and review permissions")


def load_or_setup(path):
    if path.parent.parent.is_symlink():
        raise ValueError("The config parent must not be a symbolic link")
    if path.parent.exists() or path.parent.is_symlink():
        check_private(path.parent, directory=True)
    if path.exists() or path.is_symlink():
        check_private(path)
        try:
            with path.open(encoding="utf-8") as file:
                settings = json.load(file)
        except (ValueError, OSError):
            raise ValueError("Cannot read saved settings; they were not overwritten") from None
        print("[start] Using saved settings; existing terminal values are not replacing them.", flush=True)
        return validate(settings)

    address = os.environ.get("COMPANION_LAPTOP_IP") or input("Laptop Tailscale IPv4: ").strip()
    token = os.environ.get("COMPANION_AUDIO_TOKEN")
    if not token:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                token = getpass.getpass("Paste the SAME laptop audio token (hidden): ")
            except getpass.GetPassWarning:
                raise ValueError("Hidden token input is unavailable; use an interactive SSH terminal") from None
    settings = validate({"laptop_ip": address, "port": 8765, "token": token})
    parent = path.parent.parent
    if parent.is_symlink():
        raise ValueError("The config parent must not be a symbolic link")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_info = parent.stat()
    if parent_info.st_uid != os.getuid() or parent_info.st_mode & 0o022:
        raise ValueError("The config parent must belong to this user and not be writable by other users")
    path.parent.mkdir(mode=0o700, exist_ok=True)
    check_private(path.parent, directory=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
        file.write("\n")
    check_private(path)
    print(f"[start] Saved settings: {path} (mode 600; token is NOT encrypted at rest).", flush=True)
    return settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true", help="Save/check settings without opening microphone or network")
    args = parser.parse_args(argv)
    try:
        if sys.platform != "linux":
            raise ValueError("Run the sender launcher on the Pi, not Windows")
        path = Path.home() / ".config" / "multimodal-voice" / "sender.json"
        settings = load_or_setup(path)
        print(f"[start] Destination: {settings['laptop_ip']}:8765. Token is not displayed.", flush=True)
        if args.setup_only:
            print("[start] Setup checked. Nothing was started.", flush=True)
            return 0
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, COMPANION_AUDIO_TOKEN=settings["token"])
        print("[start] Starting Pi microphone (audio-only). Ctrl+C stops capture and transmission.", flush=True)
        os.chdir(root)
        os.execve(sys.executable, [sys.executable, "-B", str(root / "app.py"), "--audio-only",
                                  "--stt-url", f"http://{settings['laptop_ip']}:8765"], env)
    except (EOFError, KeyboardInterrupt):
        print("\n[start] Cancelled; no microphone started.", flush=True)
        return 1
    except ValueError as exc:
        print(f"[start] {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("[start] Could not access settings or start Python. Check paths and permissions; settings were not deleted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
