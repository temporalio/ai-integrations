"""Install registry bytes in a clean consumer and verify distribution provenance."""

import os
import subprocess
import tempfile
import time
from pathlib import Path

from uv_tool import uv_binary


def main():
    with tempfile.TemporaryDirectory(prefix="index-smoke-") as directory:
        project = Path(directory)
        coordinate, version, api, requires = [
            os.environ[key]
            for key in ["COORDINATE", "VERSION", "ROOT_API", "REQUIRES_PYTHON"]
        ]
        text = f'[project]\nname="smoke"\nversion="0"\nrequires-python="{requires}"\ndependencies=["{coordinate}=={version}"]\n'
        if os.environ.get("INDEX_URL"):
            text += f'\n[[tool.uv.index]]\nname="release-index"\nurl="{os.environ["INDEX_URL"]}"\nexplicit=true\n\n[tool.uv.sources]\n"{coordinate}"={{index="release-index"}}\n'
        (project / "pyproject.toml").write_text(text)
        for attempt in range(10):
            result = subprocess.run(
                [uv_binary(), "sync", "--no-cache"],
                cwd=project,
                env={**os.environ, "UV_NO_EDITABLE": "1", "UV_LINK_MODE": "copy"},
            )
            if result.returncode == 0:
                break
            if attempt == 9:
                result.check_returncode()
            time.sleep(30)
        smoke = Path(__file__).parents[1] / "ci/smoke.py"
        subprocess.run(
            [
                uv_binary(),
                "run",
                "--no-sync",
                "python",
                str(smoke),
                "--in-env",
                "--coordinate",
                coordinate,
                "--root-api",
                api,
                "--imports",
                os.environ.get("SMOKE_IMPORTS", api),
            ],
            cwd=project,
            env={**os.environ, "EXPECTED_VERSION": version},
            check=True,
        )


if __name__ == "__main__":
    main()
