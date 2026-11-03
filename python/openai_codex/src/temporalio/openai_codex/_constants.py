"""Stable Activity names and connection failure text."""

CONNECT = "openai-codex-connect"
WRITE = "openai-codex-write"
READ = "openai-codex-read"
CLOSE = "openai-codex-close"
CONNECTION_LOST = (
    "Lost the connection to Codex: the Worker connected to it shut down or went "
    "away. To continue, start a new client and call thread_resume with the saved "
    "thread ID, using the same Codex home and workspace."
)
