# Import log

Append-only record of every history import and re-sync performed with
`scripts/migrate/extract-sdk-python.sh`. The merge SHA is the `--allow-unrelated-histories`
merge commit on `main`.

| Date (UTC) | Plugin | Source repo @ SHA | git-filter-repo | Merge SHA | Notes |
|---|---|---|---|---|---|
| 2026-09-02 | openai_agents | temporalio/sdk-python @ d0e075c1e25b3371f0e0f5116d9c6626892e3eab | 2.47.0 | 685eb28a2af86eff456793647fbc04afeacddb09 | initial import: 101 commits, 18 identities, root 53d9ace6 upstream |
