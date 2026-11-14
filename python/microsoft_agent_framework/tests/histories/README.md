# Recorded histories

`tests/test_replay.py` replays every `*.json` file in this directory through a `Replayer` configured with the plugin. The files were recorded by `tests/record_histories.py` from the workflows the suite runs, with scripted chat clients and the in-process MCP server.

A recording made before a release is evidence that runs started on that release still replay on the current code, so it stays as it is. A change that needs a new recording (a new scenario, or a new shape of an Activity input or result that a released worker could still be producing) adds a file next to the old ones, for example with `--suffix`, and leaves the old file in place; re-recording every file on each change would hide exactly the break the replay test exists to catch, a workflow in flight that the new worker code can no longer continue.

Before the first release the files may be re-recorded freely. After it, `record_histories.py` refuses to overwrite an existing file unless `--overwrite "<reason>"` is given; the reason is printed next to each file it replaces.

`streaming_unmerged_record.json` is such an older shape: it was recorded when the streaming Activity returned one entry per provider update, including the provider layer's duplicate of each call and one entry per text chunk. It cannot be recorded again and it keeps replaying.
