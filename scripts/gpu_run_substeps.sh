#!/bin/bash
# Run the same substeps twice from one restart and compare the two runs.
#
#   gpu_run_substeps.sh <binary A> <binary B> <prefix.in> <nsteps> <outdir> [envA] [envB]
#
# Copies <prefix.in>, its dump and any .discparams/.setup beside it into <outdir>/A and
# <outdir>/B, sets nmax=<nsteps> so the run stops after that many steps (substeps of the
# finest bin under individual timesteps) and writes a full dump there, runs each binary
# with its environment string (default: A with COSMO_GPU=1, B with COSMO_GPU=0, i.e. one
# GPU binary on both paths), then runs gpu_compare_substeps.py A B.
# Threads: OMP_NUM_THREADS from the caller's environment.
set -uo pipefail
BINA=${1:?binary A}; BINB=${2:?binary B}; IN=${3:?prefix.in}; N=${4:?nsteps}; OUT=${5:?outdir}
ENVA=${6:-COSMO_GPU=1}; ENVB=${7:-COSMO_GPU=0}
SRC=$(cd "$(dirname "$IN")" && pwd); P=$(basename "$IN" .in)
DUMP=$(sed -n 's/^ *dumpfile *= *\([^ !]*\).*/\1/p' "$IN")
HERE=$(cd "$(dirname "$0")" && pwd)
for R in A B; do
  D=$OUT/$R; rm -rf "$D"; mkdir -p "$D"
  cp "$IN" "$D/"; cp "$SRC/$DUMP" "$D/"
  cp "$SRC"/*.discparams "$SRC"/*.setup "$D/" 2>/dev/null
  sed -i -e "s/^\( *nmax *= *\)[^ !]*/\1$N/" "$D/$P.in"
done
run() {  # run <dir> <binary> <env>
  ( cd "$1" && t0=$(date +%s) && env $3 COSMO_DENS_STATS=${COSMO_DENS_STATS:-1} "$2" $P.in > run.log 2> run.err
    echo "$(basename "$1"): exit $? in $(( $(date +%s) - t0 )) s, $(grep -c '^> step' run.log) substep lines" )
}
run "$OUT/A" "$BINA" "$ENVA"
run "$OUT/B" "$BINB" "$ENVB"
python3 -I "$HERE/gpu_compare_substeps.py" "$OUT/A" "$OUT/B" --prefix "$P"
