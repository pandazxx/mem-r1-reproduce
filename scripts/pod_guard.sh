#!/usr/bin/env bash
# Cost guard for RunPod: run a command under a time limit, then stop the pod
# no matter how it ended (success, failure, or timeout) — an unattended pod
# should never keep billing. Outputs live on the network volume, which
# survives the stop.
#
# Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to get a Telegram message when
# the command finishes (skipped silently when unset).
#
# Usage: pod_guard.sh <time-limit> <command...>     e.g. pod_guard.sh 3.5h uv run ...
set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 <time-limit, e.g. 90m or 3.5h> <command...>" >&2
    exit 2
fi

limit="$1"
shift

notify() {
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -fsS --max-time 15 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" --data-urlencode text="$1" >/dev/null \
            || echo "pod-guard: telegram notification failed" >&2
    fi
}

start=$(date +%s)
# --foreground keeps the command attached to the terminal (live logs, Ctrl-C);
# SIGTERM first so trainers can flush a checkpoint, SIGKILL 60s later.
timeout --foreground --signal=TERM --kill-after=60 "$limit" "$@"
status=$?
elapsed=$(( $(date +%s) - start ))
runtime="$(( elapsed / 3600 ))h$(printf '%02d' $(( elapsed % 3600 / 60 )))m"

if [ "$status" -eq 0 ]; then
    outcome="finished successfully"
elif [ "$status" -eq 124 ]; then
    outcome="hit the $limit time limit"
else
    outcome="failed (exit $status)"
fi
echo "pod-guard: command $outcome after $runtime" >&2

if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    echo "pod-guard: stopping pod $RUNPOD_POD_ID to cap cost" >&2
    notify "pod-guard: \`$*\` $outcome after $runtime — stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "pod-guard: not on RunPod (RUNPOD_POD_ID or runpodctl missing) — pod NOT stopped" >&2
    notify "pod-guard: \`$*\` $outcome after $runtime (not on RunPod, nothing stopped)"
fi
exit "$status"
