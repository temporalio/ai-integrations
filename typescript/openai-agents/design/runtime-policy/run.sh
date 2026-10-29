#!/bin/sh
set -eu

# Run inside the existing evaluation image, with the checkout and evidence mounted.
proof_plugin=${1:?Pass the disposable typescript/openai-agents directory}
proof_evidence=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$proof_plugin"

python3 "$proof_evidence/prepare.py" "$proof_plugin" baseline
make sync
pnpm run build
node boundary-upstream.cjs > "$proof_evidence/upstream.txt" 2>&1
node boundary-proof.cjs baseline "$proof_evidence/histories-baseline" > "$proof_evidence/baseline.txt" 2>&1
python3 "$proof_evidence/prepare.py" "$proof_plugin" policy
pnpm run build
node boundary-proof.cjs policy "$proof_evidence/histories-policy" > "$proof_evidence/policy.txt" 2>&1
make test 'TEST_ARGS=--match=*defaults* --match=*options*name* --match=*handoff*asTool* --match=*saved*function*resumes* --match=*approval* --match=*stream*handlers* --match=*handler*registered* --match=*Cancelling*nested* --match=*stops*tool* --match=*listeners*context* --match=*callback*returns* --match=*sandbox*lifecycle* --match=*sandbox*replaces* --match=*own*model*upstream*' > "$proof_evidence/controls.txt" 2>&1
node boundary-replay.cjs "$proof_evidence" > "$proof_evidence/replay.txt" 2>&1
echo 'PASS: baseline, runtime policy, existing controls, replay, and Worker replacement'
