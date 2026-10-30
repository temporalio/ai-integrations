# Binds run_async when imported, before any Worker installs the guard.
from strands._async import run_async as run_async
