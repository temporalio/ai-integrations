# Import log

Append-only record of every history import and re-sync performed with
`scripts/migrate/extract-sdk-python.sh`. The merge SHA is the `--allow-unrelated-histories`
merge commit on `main`.

| Date (UTC) | Plugin | Source repo @ SHA | git-filter-repo | Merge SHA | Notes |
|---|---|---|---|---|---|
| 2026-09-02 | openai_agents | temporalio/sdk-python @ d0e075c1e25b3371f0e0f5116d9c6626892e3eab | 2.47.0 | 685eb28a2af86eff456793647fbc04afeacddb09 | initial import: 101 commits, 18 identities, root 53d9ace6 upstream |
| 2026-09-03 | openai_agents | temporalio/sdk-python#1793 @ dcb4e8ed6503d45874f22e97296854822d7fe6bc | 2.47.0 | 827e59f154deba40ae79b4504ee0fdc4bbec1f79 | re-sync: 7 MCP v2 adapter commits; depends on temporalio-mcp 0.1.x |
