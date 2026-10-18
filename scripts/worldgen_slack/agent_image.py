"""The agents' Docker image: the pinned base every agent's sandbox runs, with the rlm harness already installed where
the harness looks for it, so a container starts in seconds rather than installing git, uv and nano-rlm first (7
minutes on the research VM's link, for every container: the author's and each review's judge).

uv run --frozen python scripts/worldgen_slack/agent_image.py [--tag zawarudo-agent] [--print]

It builds the image on this machine's Docker and prints its content id (sha256:...), which a run config names as
`[env] docker_image` beside `sandbox = "docker"`: the id is a hash of the image's content, so a run is pinned to these
exact bytes, as a Prime VM is to its digest. --print shows the Dockerfile and builds nothing."""

import argparse
import hashlib
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from verifiers.v1.harnesses.rlm.harness import RLM_CACHE_DIR, RLM_REPO
from worldgen_slack.config import IMAGE, RLM_REVISION


def dockerfile(base: str = IMAGE, version: str = RLM_REVISION) -> str:
    """The harness's own install (verifiers' RLMHarness.setup), run at build time into the directory it checks for
    `.ready`: a container of this image skips it."""
    directory = f"{RLM_CACHE_DIR}-{hashlib.sha256(version.encode()).hexdigest()}"
    checkout = f"{directory}/checkout"
    install = (
        "apt-get update -qq && apt-get install -y -qq git && "
        f"git clone https://{RLM_REPO} {checkout} && "
        f"git -C {checkout} checkout {shlex.quote(version)} && "
        # the extra uv args the harness gives install.sh (RLMHarness.setup)
        f"RLM_EXTRA_UV_ARGS='--with mcp~=1.28' UV_INSTALL_DIR={directory}/bin UV_TOOL_BIN_DIR={directory}/bin "
        f"RLM_CHECKOUT_PATH={checkout} bash {checkout}/install.sh && "
        f"touch {directory}/.ready"
    )
    return f"FROM {base}\nRUN {install}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="zawarudo-agent")
    parser.add_argument("--print", action="store_true", help="show the Dockerfile; build nothing")
    args = parser.parse_args()
    text = dockerfile()
    if args.print:
        print(text)
        return 0
    tag = f"{args.tag}:{RLM_REVISION[:12]}"
    with tempfile.TemporaryDirectory() as context:
        (Path(context) / "Dockerfile").write_text(text)
        subprocess.run(["docker", "build", "--tag", tag, context], check=True, stdout=sys.stderr)
    image = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f'{tag}\n[env]\nsandbox = "docker"\ndocker_image = "{image}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
