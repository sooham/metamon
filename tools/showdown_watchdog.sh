#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="${METAMON_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

HOST="${METAMON_SHOWDOWN_HOST:-127.0.0.1}"
PORT="${METAMON_SHOWDOWN_PORT:-8000}"
HEALTH_URL="${METAMON_SHOWDOWN_HEALTH_URL:-http://${HOST}:${PORT}/}"
INTERVAL_SECONDS="${METAMON_SHOWDOWN_WATCH_INTERVAL:-15}"
SESSION="${METAMON_SHOWDOWN_TMUX_SESSION:-metamon-showdown}"
STATE_DIR="${METAMON_SHOWDOWN_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/metamon}"

LOCK_FILE="$STATE_DIR/showdown-watchdog.lock"
PID_FILE="$STATE_DIR/showdown.pid"
WATCHDOG_PID_FILE="$STATE_DIR/showdown-watchdog.pid"
SERVER_LOG="$STATE_DIR/showdown.log"
WATCHDOG_LOG="$STATE_DIR/showdown-watchdog.log"

mkdir -p "$STATE_DIR"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$WATCHDOG_LOG" >&2
}

is_healthy() {
    curl -sS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1
}

pid_alive() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1
}

resolve_node() {
    if [[ -n "${METAMON_NODE:-}" && -x "${METAMON_NODE}" ]]; then
        printf '%s\n' "$METAMON_NODE"
        return 0
    fi

    local node_bin
    node_bin="$(command -v node 2>/dev/null || true)"
    if [[ -n "$node_bin" ]]; then
        printf '%s\n' "$node_bin"
        return 0
    fi

    shopt -s nullglob
    local candidates=(
        /opt/nvm/versions/node/*/bin/node
        "$HOME"/.nvm/versions/node/*/bin/node
        /usr/local/bin/node
        /usr/bin/node
    )
    shopt -u nullglob

    local idx
    for ((idx=${#candidates[@]} - 1; idx >= 0; idx--)); do
        if [[ -x "${candidates[$idx]}" ]]; then
            printf '%s\n' "${candidates[$idx]}"
            return 0
        fi
    done

    return 1
}

start_showdown() {
    if is_healthy; then
        log "Showdown is already reachable at $HEALTH_URL"
        return 0
    fi

    if pid_alive "$PID_FILE"; then
        log "Showdown process $(cat "$PID_FILE") is running, but $HEALTH_URL is not reachable yet"
        return 0
    fi

    local node_bin
    if ! node_bin="$(resolve_node)"; then
        log "ERROR: node executable not found. Set METAMON_NODE=/path/to/node and retry."
        return 1
    fi

    log "Starting Showdown with make showdown on $HEALTH_URL (node: $node_bin)"
    setsid bash -c '
        set -euo pipefail
        repo_root="$1"
        node_dir="$2"
        node_bin="$3"
        printf "\\n[%s] starting make showdown\\n" "$(date -Is)"
        cd "$repo_root"
        PATH="$node_dir:$PATH" exec make showdown NODE="$node_bin"
    ' bash "$REPO_ROOT" "$(dirname "$node_bin")" "$node_bin" >>"$SERVER_LOG" 2>&1 &
    printf '%s\n' "$!" >"$PID_FILE"
}

stop_showdown() {
    if pid_alive "$PID_FILE"; then
        local pid
        pid="$(cat "$PID_FILE")"
        log "Stopping tracked Showdown process group $pid"
        kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
        sleep 1
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    fi
    rm -f "$PID_FILE"
}

watch() {
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "Showdown watchdog is already running"
        exit 0
    fi

    printf '%s\n' "$$" >"$WATCHDOG_PID_FILE"
    trap 'log "Stopping Showdown watchdog"; stop_showdown; rm -f "$WATCHDOG_PID_FILE"; exit 0' INT TERM HUP

    log "Showdown watchdog started for $HEALTH_URL"
    while true; do
        if ! is_healthy; then
            log "$HEALTH_URL is down; ensuring Showdown is running"
            start_showdown || true
        fi
        sleep "$INTERVAL_SECONDS"
    done
}

start_daemon() {
    if command -v tmux >/dev/null 2>&1; then
        if tmux has-session -t "$SESSION" >/dev/null 2>&1; then
            printf 'Showdown watchdog tmux session already exists: %s\n' "$SESSION"
        else
            tmux new-session -d -s "$SESSION" "$SCRIPT_PATH watch"
            printf 'Started Showdown watchdog tmux session: %s\n' "$SESSION"
        fi
    else
        if pid_alive "$WATCHDOG_PID_FILE"; then
            printf 'Showdown watchdog process already exists: %s\n' "$(cat "$WATCHDOG_PID_FILE")"
        else
            nohup "$SCRIPT_PATH" watch >>"$WATCHDOG_LOG" 2>&1 &
            printf '%s\n' "$!" >"$WATCHDOG_PID_FILE"
            printf 'Started Showdown watchdog process: %s\n' "$!"
        fi
    fi

    sleep 2
    status
}

status() {
    if is_healthy; then
        printf 'Showdown: reachable at %s\n' "$HEALTH_URL"
    else
        printf 'Showdown: not reachable at %s\n' "$HEALTH_URL"
    fi

    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" >/dev/null 2>&1; then
        printf 'Watchdog: tmux session %s is running\n' "$SESSION"
    elif pid_alive "$WATCHDOG_PID_FILE"; then
        printf 'Watchdog: process %s is running\n' "$(cat "$WATCHDOG_PID_FILE")"
    else
        printf 'Watchdog: not running\n'
    fi

    if pid_alive "$PID_FILE"; then
        printf 'Tracked Showdown pid: %s\n' "$(cat "$PID_FILE")"
    else
        printf 'Tracked Showdown pid: none\n'
    fi

    printf 'Server log: %s\n' "$SERVER_LOG"
    printf 'Watchdog log: %s\n' "$WATCHDOG_LOG"
}

stop_daemon() {
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" >/dev/null 2>&1; then
        tmux kill-session -t "$SESSION"
        printf 'Stopped Showdown watchdog tmux session: %s\n' "$SESSION"
    fi

    if pid_alive "$WATCHDOG_PID_FILE"; then
        kill "$(cat "$WATCHDOG_PID_FILE")" >/dev/null 2>&1 || true
        rm -f "$WATCHDOG_PID_FILE"
    fi

    stop_showdown
    status
}

install_cron() {
    if ! command -v crontab >/dev/null 2>&1; then
        printf 'ERROR: crontab is not available on this machine.\n' >&2
        return 1
    fi

    local tmp_in tmp_out begin end
    tmp_in="$(mktemp)"
    tmp_out="$(mktemp)"
    begin="# metamon-showdown-watchdog BEGIN $REPO_ROOT"
    end="# metamon-showdown-watchdog END $REPO_ROOT"

    crontab -l >"$tmp_in" 2>/dev/null || true
    awk -v begin="$begin" -v end="$end" '
        $0 == begin { skip = 1; next }
        $0 == end { skip = 0; next }
        skip != 1 { print }
    ' "$tmp_in" >"$tmp_out"

    {
        printf '%s\n' "$begin"
        printf '@reboot cd %q && %q start >/dev/null 2>&1\n' "$REPO_ROOT" "$SCRIPT_PATH"
        printf '* * * * * cd %q && %q start >/dev/null 2>&1\n' "$REPO_ROOT" "$SCRIPT_PATH"
        printf '%s\n' "$end"
    } >>"$tmp_out"

    crontab "$tmp_out"
    rm -f "$tmp_in" "$tmp_out"
    printf 'Installed Showdown watchdog cron entry for %s\n' "$REPO_ROOT"
}

uninstall_cron() {
    if ! command -v crontab >/dev/null 2>&1; then
        printf 'ERROR: crontab is not available on this machine.\n' >&2
        return 1
    fi

    local tmp_in tmp_out begin end
    tmp_in="$(mktemp)"
    tmp_out="$(mktemp)"
    begin="# metamon-showdown-watchdog BEGIN $REPO_ROOT"
    end="# metamon-showdown-watchdog END $REPO_ROOT"

    crontab -l >"$tmp_in" 2>/dev/null || true
    awk -v begin="$begin" -v end="$end" '
        $0 == begin { skip = 1; next }
        $0 == end { skip = 0; next }
        skip != 1 { print }
    ' "$tmp_in" >"$tmp_out"

    crontab "$tmp_out"
    rm -f "$tmp_in" "$tmp_out"
    printf 'Removed Showdown watchdog cron entry for %s\n' "$REPO_ROOT"
}

usage() {
    cat <<EOF
Usage: $0 {start|watch|status|stop|install-cron|uninstall-cron}

Environment:
  METAMON_SHOWDOWN_PORT          default: 8000
  METAMON_SHOWDOWN_HOST          default: 127.0.0.1
  METAMON_SHOWDOWN_TMUX_SESSION  default: metamon-showdown
  METAMON_SHOWDOWN_STATE_DIR     default: \$XDG_STATE_HOME/metamon or \$HOME/.local/state/metamon
  METAMON_NODE                   optional absolute path to node
EOF
}

cmd="${1:-start}"
case "$cmd" in
    start)
        start_daemon
        ;;
    watch)
        watch
        ;;
    status)
        status
        ;;
    stop)
        stop_daemon
        ;;
    install-cron)
        install_cron
        ;;
    uninstall-cron)
        uninstall_cron
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
