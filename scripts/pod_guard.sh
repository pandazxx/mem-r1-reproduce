#!/usr/bin/env bash
# Cost guard for RunPod: run a command under a time limit, then stop the pod
# no matter how it ended (success, failure, or timeout) — an unattended pod
# should never keep billing. Outputs live on the network volume, which
# survives the stop.
#
# Usage: pod_guard.sh <time-limit> <command...>     e.g. pod_guard.sh 3.5h uv run ...
set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <time-limit, e.g. 90m or 3.5h> <command...>" >&2
    exit 2
fi

limit="$1"
shift

# --foreground keeps the command attached to the terminal (live logs, Ctrl-C);
# SIGTERM first so trainers can flush a checkpoint, SIGKILL 60s later.
timeout --foreground --signal=TERM --kill-after=60 "$limit" "$@"
status=$?

if [ "$status" -eq 0 ]; then
    echo "pod-guard: command finished successfully" >&2
elif [ "$status" -eq 124 ]; then
    echo "pod-guard: time limit ($limit) exceeded" >&2
else
    echo "pod-guard: command failed (exit $status)" >&2
fi

if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    echo "pod-guard: stopping pod $RUNPOD_POD_ID to cap cost" >&2
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "pod-guard: not on RunPod (RUNPOD_POD_ID or runpodctl missing) — pod NOT stopped" >&2
fi
exit "$status"
