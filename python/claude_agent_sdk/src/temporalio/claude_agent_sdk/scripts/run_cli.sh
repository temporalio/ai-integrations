# Run the CLI (the arguments after $2), feeding it the files in the stdin
# directory $1 in order, 1.jsonl, 2.jsonl, ..., up to an $end_of_input line.
# Each file appears whole. On exit, write the CLI's status to $exited, which
# also stops the feeding.
stdin=$1 exited=$2
shift 2
n=1
while [ ! -e "$exited" ]; do
  if [ -e "$stdin/$n.jsonl" ]; then
    while IFS= read -r line; do
      [ "$line" = "$end_of_input" ] && exit
      printf '%s\n' "$line"
    done < "$stdin/$n.jsonl"
    n=$((n + 1))
  else
    sleep 0.05
  fi
done | { "$@"; echo $? > "$exited"; }
