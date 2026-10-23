# Print the stdout file from byte $from as it grows, until the CLI exits: $exited
# appears or shell $pid is gone. If $boot is not this boot, the pid is stale:
# print what is there and stop.
# A portable `tail --pid`, which BSD and BusyBox tail lack. exec makes tail the
# command, so stopping the exec stops everything.
stdout=$1 from=$2 exited=$3 pid=$4 boot=$5
[ "$(boot_id)" = "$boot" ] || exec tail -c +"$from" "$stdout"
(while [ ! -e "$exited" ] && kill -0 "$pid" && kill -0 $$; do sleep 0.2; done; kill $$) >/dev/null 2>&1 &
exec tail -c +"$from" -f "$stdout"
