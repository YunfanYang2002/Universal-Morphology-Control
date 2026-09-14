#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
keep_open=1
if [[ "${1:-}" == "--no-keep-open" ]]; then
    keep_open=0
    shift
fi
finish() {
    code=$?
    trap - EXIT
    printf 'HD0_LAUNCHER_EXIT=%s\n' "$code"
    if [[ "$keep_open" == 1 && -t 0 ]]; then
        printf '日志已保留；输入 exit 关闭终端。\n'
        bash --noprofile --norc -i || true
    fi
    exit "$code"
}
trap finish EXIT
# Use the active, explicitly verified environment; never activate a guessed one.
python tools/run_hyperdistill_hd0.py "$@"
