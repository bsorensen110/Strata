#!/usr/bin/env bash
# Start one campaign arm as a transient user unit (escapes the 4 GiB Hermes cgroup cap).
set -euo pipefail
name=$1
root=/home/soren/Strata-latest-20261112-182859/bench/ports-ab-2026-11-12
cfg="$root/configs/$name.json"
[[ -f "$cfg" ]]
port=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$cfg")
unit="portsab-$name"
scratch_env="$root/configs/$name/arm-env.sh"
python3 - "$cfg" "$scratch_env" <<'PYENV'
import json, sys, shlex
d = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'w') as f:
    for k, v in d.get('env', {}).items():
        f.write('export ' + k + '=' + shlex.quote(str(v)) + '\n')
PYENV
chmod 600 "$scratch_env"
systemd-run --user --unit="$unit" --collect --no-block \
  --property=MemoryMax=infinity --property=MemorySwapMax=infinity --property=TasksMax=infinity \
  -- /usr/bin/bash -c 'set -a; source /home/soren/.config/strata-upstream-v0.2.10/production.env; source "$1"/baseline-env.sh; source "$4"; cd "$5"; exec "$6" -u -m serve.server --engine strata --config "$1/configs/$2.json" --host 127.0.0.1 --port "$3" > "$1/configs/$2/serve.out" 2>&1' \
  _ "$root" "$name" "$port" "$scratch_env" /home/soren/Strata-latest-20261112-182859 \
  /home/soren/.local/share/strata-upstream-v0.2.10/terminal/0.1.56/python/bin/python3
