#!/usr/bin/env bash
set -uo pipefail

CTM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTM_REPO_DIR="$(cd "${CTM_SCRIPT_DIR}/../.." && pwd)"
CTM_STEPS="${CTM_ABLATION_STEPS:-300000}"
CTM_SEED="${CTM_ABLATION_SEED:-1}"
CTM_PREFETCH="${CTM_ABLATION_PREFETCH:-2}"
CTM_ENVS="${CTM_ABLATION_ENVS:-1}"
CTM_OUTPUT_ROOT="${CTM_ABLATION_LOGDIR:-${CTM_REPO_DIR}/logdir/ablations/walker_300k_seed${CTM_SEED}}"

mkdir -p "${CTM_OUTPUT_ROOT}"

run_variant() {
  local name="$1"
  local replay_capacity="$2"
  local batch_size="$3"
  local logdir="${CTM_OUTPUT_ROOT}/${name}"

  mkdir -p "${logdir}"
  echo
  echo "=== ${name}: replay_capacity=${replay_capacity}, batch_size=${batch_size} ==="
  if bash "${CTM_REPO_DIR}/run_wsl.sh" -u "${CTM_REPO_DIR}/dreamer.py" \
      --logdir "${logdir}" \
      --task dmc_walker_walk \
      --steps "${CTM_STEPS}" \
      --seed "${CTM_SEED}" \
      --precision 32 \
      --envs "${CTM_ENVS}" \
      --batch_size "${batch_size}" \
      --replay_capacity "${replay_capacity}" \
      --dataset_prefetch "${CTM_PREFETCH}" \
      --log_images False \
      2>&1 | tee -a "${logdir}/console.log"; then
    echo "${name}: COMPLETE" | tee "${logdir}/status.txt"
    return 0
  else
    local code=${PIPESTATUS[0]}
    echo "${name}: FAILED (exit ${code})" | tee "${logdir}/status.txt"
    return "${code}"
  fi
}

declare -A CTM_REPLAY=(
  [control]=100000
  [replay_all]=0
  [replay_all_batch50]=0
)
declare -A CTM_BATCH=(
  [control]=32
  [replay_all]=32
  [replay_all_batch50]=50
)

if (($#)); then
  CTM_VARIANTS=("$@")
else
  CTM_VARIANTS=(control replay_all replay_all_batch50)
fi

CTM_FAILED=0
for variant in "${CTM_VARIANTS[@]}"; do
  if [[ -z "${CTM_REPLAY[$variant]+x}" ]]; then
    echo "Unknown variant: ${variant}" >&2
    CTM_FAILED=1
    continue
  fi
  run_variant "${variant}" "${CTM_REPLAY[$variant]}" "${CTM_BATCH[$variant]}" || CTM_FAILED=1
done

echo
echo "=== Comparison summary ==="
bash "${CTM_REPO_DIR}/run_wsl.sh" -u \
  "${CTM_SCRIPT_DIR}/compare_results.py" \
  --logroot "${CTM_OUTPUT_ROOT}" \
  --scores "${CTM_REPO_DIR}/scores/dreamer.json" || CTM_FAILED=1

echo
echo "Results: ${CTM_OUTPUT_ROOT}"
exit "${CTM_FAILED}"
