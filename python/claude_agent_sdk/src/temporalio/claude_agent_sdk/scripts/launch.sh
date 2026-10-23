# Start the CLI once, detached, by running $run (run_cli.sh) with the arguments
# after it. Prints "<pid> <boot ID>".
# - A retry finds $pid and starts no second CLI.
# - $pid is written to $new_pid and moved into place, so no reader sees half.
# - The CLI's output goes to files, so the exec returns without waiting on it.
stdin=$1 stdout=$2 log=$3 pid=$4 new_pid=$5 exited=$6 run=$7
shift 7
if [ ! -e "$pid" ]; then
  mkdir -p "$stdin" || exit 1
  sh -c "$run" sh "$stdin" "$exited" "$@" >>"$stdout" 2>"$log" &
  echo "$! $(boot_id)" > "$new_pid" && mv "$new_pid" "$pid"
fi
cat "$pid"
