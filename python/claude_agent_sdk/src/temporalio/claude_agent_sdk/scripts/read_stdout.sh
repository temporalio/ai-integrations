# Print the stdout file from byte $from, or nothing if it is missing (a CLI from
# before a reset).
stdout=$1 from=$2
[ ! -e "$stdout" ] || tail -c +"$from" "$stdout"
