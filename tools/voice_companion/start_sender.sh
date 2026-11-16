#!/usr/bin/env bash
# Source this file to reuse laptop_ip from the existing SSH shell.
(
    set +x
    tool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
    project_root="$(cd -- "$tool_dir/../.." && pwd)" || exit 1
    python="$project_root/.venv/bin/python"
    if [ ! -x "$python" ]; then
        printf '%s\n' '[start] Project .venv/bin/python is missing. Stop and check the Pi environment.' >&2
        exit 1
    fi
    export COMPANION_LAPTOP_IP="${laptop_ip:-${COMPANION_LAPTOP_IP:-}}"
    cd -- "$project_root" || exit 1
    exec "$python" -B "$tool_dir/start_sender.py" "$@"
)
