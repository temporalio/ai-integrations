"""Stable Activity names and connection failure text."""

CONNECT = "openai-codex-connect"
RESERVE = "openai-codex-reserve"
WRITE = "openai-codex-write"
READ = "openai-codex-read"
CLOSE = "openai-codex-close"
CONNECTION_LOST = (
    "Lost the connection to Codex: the Worker connected to it shut down or went "
    "away. Start a new client and resume the persisted thread by its ID."
)
