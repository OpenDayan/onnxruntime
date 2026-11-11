#!/usr/bin/env bash
# Dayan 算子回归。
#
# run_dayan_tests.sh [--report] [--direct-ops] [--out DIR] [--conv SPEC] [op ...]
#   --report      输出张量尺寸、仿真周期和 CSV 汇总。
#   --direct-ops  测试代表算子并外推整网周期，需要 TestCluster 的 tmux 会话。
#   --conv SPEC   IN:W:STRIDE:PAD，例如 1,3,15,17:5,3,3,3:2,2:1,1。
#   --out DIR     direct-ops 结果目录。
#   op            只运行指定算子，如 add、matmul、yolov8n-official。
#
# DAYAN_SERVER_SOCKET 指定 dayan_server socket；RVNPU_YOLO_IMAGE 指定测试图片。
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/build"
TEST_LOG_DIR="${DAYAN_TEST_LOG_DIR:-${ROOT_DIR}/build/dayan-test-logs}"
TC_SESSION="${DAYAN_TC_SESSION:-sim}"
CYCLE_CSV="${DAYAN_CYCLE_CSV:-${ROOT_DIR}/build/dayan-op-cycles.csv}"

REPORT=0
[[ "${DAYAN_REPORT:-0}" == "1" ]] && REPORT=1
DIRECT_OPS=0
CONV_SPEC="1,3,15,17:5,3,3,3:2,2:1,1"
OUT_DIR=""

POS=()
while (($#)); do
  case "$1" in
    --report)     REPORT=1 ;;
    --no-report)  REPORT=0 ;;
    --direct-ops) DIRECT_OPS=1 ;;
    --conv)       [[ $# -ge 2 ]] || { echo "--conv 需要参数" >&2; exit 2; }
                  CONV_SPEC="$2"; shift ;;
    --out)        [[ $# -ge 2 ]] || { echo "--out 需要参数" >&2; exit 2; }
                  OUT_DIR="$2"; shift ;;
    -h|--help)    sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            POS+=("$1") ;;
  esac
  shift
done
set -- ${POS[@]+"${POS[@]}"}

export LD_LIBRARY_PATH="${ROOT_DIR}/build/ort-sdk/lib:${LD_LIBRARY_PATH:-}"
export DAYAN_SERVER_SOCKET="${DAYAN_SERVER_SOCKET:-${TMPDIR:-/tmp}/dayan-server.sock}"
export BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/dayan-migration}"
export RVNPU_KERNEL_ELF="${RVNPU_KERNEL_ELF:-${ROOT_DIR}/build/dummy-kernel-elf}"

mkdir -p "${TEST_LOG_DIR}"
[[ -e "${RVNPU_KERNEL_ELF}" ]] || : > "${RVNPU_KERNEL_ELF}"

tc_available() {
  command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TC_SESSION}" 2>/dev/null
}

tc_cycles() {
  tmux capture-pane -t "${TC_SESSION}" -p 2>/dev/null \
    | grep -a -o -E 'Cycl[a-z]*[0-9]+/' | head -1 | tr -dc '0-9'
}

model_sizes() {
  local models=("${ROOT_DIR}/build/operators/$1"/*/model.onnx)
  [[ -e "${models[0]}" ]] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  python3 - "${ROOT_DIR}/build/operators/" "${models[@]}" <<'PY' 2>/dev/null | paste -sd'|' -
import os, sys
try:
    import onnx
except Exception:
    sys.exit(0)
prefix = sys.argv[1]
for path in sys.argv[2:]:
    try:
        model = onnx.load(path)
    except Exception:
        continue
    def fmt(value_info):
        tensor = value_info.type.tensor_type
        dims = "x".join(
            str(d.dim_value) if d.HasField("dim_value") and d.dim_value else (d.dim_param or "?")
            for d in tensor.shape.dim
        )
        return "%s[%s]" % (value_info.name, dims)
    name = os.path.relpath(path, prefix).replace("/model.onnx", "")
    print("%s: %s" % (name, ",".join(fmt(v) for v in model.graph.input)))
PY
}

op_sizes() {
  local op="$1" rs="${ROOT_DIR}/tests/operators/${op}/run.sh" out=""
  if [[ -f "${rs}" ]]; then
    out="$(grep -o -E \
        -- '--(shape|rhs-shape|input-shape|weight-shape) [0-9,x]+|--m [0-9]+ --n [0-9]+ --k [0-9]+|--rows [0-9]+ --columns [0-9]+ --k [0-9]+' \
        "${rs}" 2>/dev/null | paste -sd'|' -)"
  fi
  [[ -z "${out}" ]] && out="$(model_sizes "${op}")"
  [[ -z "${out}" ]] && out="-"
  if (( ${#out} > 150 )); then out="${out:0:147}..."; fi
  printf '%s' "${out}"
}

ROWS=()

run_op() {
  local op="$1"
  local log="${TEST_LOG_DIR}/${op}.log"
  local sizes="-" c0="" c1="" cycles="-" t0 t1 wall rc

  if [[ ${REPORT} == 1 ]]; then sizes="$(op_sizes "${op}")"; fi

  echo "########## ${op}"
  [[ ${REPORT} == 1 ]] && echo "  sizes: ${sizes}"

  if [[ -f "${ROOT_DIR}/tests/operators/${op}/prepare.sh" ]]; then
    if ! bash "${ROOT_DIR}/tests/operators/${op}/prepare.sh" >>"${log}.prepare" 2>&1; then
      echo "PREPARE FAIL ${op}"
      ROWS+=("${op}|${sizes}|-|-|PREPARE_FAIL")
      return 1
    fi
  fi

  if [[ ${REPORT} == 1 ]] && tc_available; then c0="$(tc_cycles)"; fi
  t0=$(date +%s)
  if bash "${ROOT_DIR}/tests/operators/${op}/run.sh" >"${log}" 2>&1; then rc=0; else rc=1; fi
  t1=$(date +%s)
  wall=$((t1 - t0))
  if [[ -n "${c0}" ]] && tc_available; then
    c1="$(tc_cycles)"
    [[ -n "${c1}" ]] && cycles=$((c1 - c0))
  fi

  if [[ ${rc} -eq 0 ]]; then
    echo "PASS ${op}  cycles=${cycles}  wall=${wall}s"
  else
    echo "FAIL ${op}  cycles=${cycles}  wall=${wall}s  (log ${log})"
    tail -30 "${log}"
  fi
  ROWS+=("${op}|${sizes}|${cycles}|${wall}s|$([[ ${rc} -eq 0 ]] && echo PASS || echo FAIL)")
  return ${rc}
}

print_summary() {
  local total=0
  echo
  echo "======== SUMMARY fail=${1}"
  if [[ ${REPORT} != 1 ]]; then return; fi
  if ! tc_available; then
    echo "（未找到 tmux 会话 ${TC_SESSION}，没有采集周期）"
    return
  fi
  {
    printf '%-12s %12s %8s  %s\n' op cycles wall sizes
    printf '%-12s %12s %8s  %s\n' ------------ ------------ -------- -----
    for r in "${ROWS[@]}"; do
      IFS='|' read -r o s c w res <<<"${r}"
      [[ "${c}" =~ ^[0-9]+$ ]] && total=$((total + c))
      printf '%-12s %12s %8s  %s\n' "${o}" "${c}" "${w}" "${s}"
    done
    printf '%-12s %12s\n' TOTAL "${total}"
  } | tee "${CYCLE_CSV}.txt"

  {
    echo "op,cycles,wall,result,sizes"
    for r in "${ROWS[@]}"; do
      IFS='|' read -r o s c w res <<<"${r}"
      echo "${o},${c},${w},${res},${s}"
    done
  } > "${CYCLE_CSV}"
  echo "（周期明细已写入 ${CYCLE_CSV}）"
}

DAYAN_FIXED_CYCLES=22000

run_direct_sample() {
  local name="$1" tool="$2"; shift 2
  local args="$*" c0 c1 raw net rc attempt
  echo "---------- sample ${name}: ${tool} ${args}"
  for attempt in 1 2 3; do
    c0="$(tc_cycles)"
    if timeout 600 "${BUILD_DIR}/${tool}" \
          --kernel-elf "${RVNPU_KERNEL_ELF}" ${args} >"${TEST_LOG_DIR}/direct-${name}.log" 2>&1; then
      rc=0
    else
      rc=1
    fi
    sleep 1  # TUI 刷新粒度 ~1s，等最后一帧稳定
    c1="$(tc_cycles)"
    raw=$(( ${c1:-0} - ${c0:-0} ))
    net=$(( raw - DAYAN_FIXED_CYCLES ))
    (( net < 0 )) && net=0
    if [[ ${rc} -eq 0 ]]; then
      echo "PASS ${name}  net_cycles=${net}  raw=${raw}  (attempt ${attempt})"
      break
    fi
    if [[ ${attempt} -lt 3 ]]; then
      echo "  retry ${name}: attempt ${attempt} FAIL，重试"
    else
      echo "FAIL ${name}  net_cycles=${net}  raw=${raw}  (log ${TEST_LOG_DIR}/direct-${name}.log)"
      tail -2 "${TEST_LOG_DIR}/direct-${name}.log" | sed 's/^/    /'
    fi
  done
  echo "${name},${tool},\"${args// /_}\",${rc},${raw},${net}" >>"${DIRECT_CSV}"
}

direct_ops_summarize() {
  python3 - "${OUT_DIR:-${ROOT_DIR}/build/yolo_representative/yolov8n_64_dayan}" "${ROOT_DIR}" <<'PY'
import csv, json, math, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2]) / "build/yolov8n_64/perf_manifest.json"
if not manifest_path.exists():
    print(f"skip network extrapolation: missing {manifest_path}")
    sys.exit(0)
manifest = json.loads(manifest_path.read_text())
entries = manifest['graph_entries']
rows = list(csv.DictReader((out_dir / 'samples_dayan.csv').open()))
samples = {r['name']: r for r in rows}

def tokens(args): return args.split('_')
def elems(args):
    t = tokens(args)
    for i, x in enumerate(t):
        if x in ('--shape', '--input-shape'):
            return math.prod(int(d) for d in t[i+1].split(','))
    return 1
def conv_macs(args):
    t = tokens(args); d = {}
    for i, x in enumerate(t):
        if x in ('--input-shape', '--weight-shape', '--stride', '--pad'):
            d[x] = [int(v) for v in t[i+1].split(',')]
    i, w = d['--input-shape'], d['--weight-shape']
    s, p = d['--stride'][0], d['--pad'][0]
    oh = (i[2] + 2*p - w[2]) // s + 1
    ow = (i[3] + 2*p - w[3]) // s + 1
    return w[0] * w[2] * w[3] * i[1] * oh * ow

conv_m = conv_macs(samples['conv']['args'])
conv_rate = int(samples['conv']['net_cycles']) / conv_m
print(f"conv 样本 MACs={conv_m}  conv_rate={conv_rate:.4f} cyc/MAC")

def ew_rate(op):
    cands = [r for n, r in samples.items() if n.split('_mid')[0] == op]
    best = max(cands, key=lambda r: elems(r['args']))
    return int(best['net_cycles']) / elems(best['args'])
rates = {op: ew_rate(op) for op in ('add', 'mul', 'div', 'sigmoid') if any(n.split('_mid')[0] == op for n in samples)}
for op, r in rates.items():
    print(f"{op}_rate={r:.3f} cyc/elem")
post = {op: int(samples[op]['net_cycles']) for op in
        ('maxpool', 'softmax', 'reduce_max', 'resize') if op in samples}
nms_net = int(samples['nms']['net_cycles']) if 'nms' in samples else 0
LEGACY_ARGMAX = 64406
nms_scale = (84 * 80) / 3  # nonoverlap 样本 3 boxes/1 class，比例外推到 84x80

def shape_text(e):
    op = e['op']
    if op == 'conv': return f"input={e['input']} weight={e['weight']} stride={e['stride']} pad={e['pad']}"
    if op == 'maxpool': return f"input={e['input']} kernel={e['kernel']} stride={e['stride']} pad={e['pad']}"
    if op == 'transpose': return f"input={e['input']} perm={e['perm']} output={e['output']}"
    if op in ('add', 'mul', 'div'): return f"lhs={e.get('lhs')} rhs={e.get('rhs')} output={e.get('output')}"
    if op == 'resize': return f"input={e['input']} output={e['output']}"
    if op in ('softmax', 'argmax', 'reduce_max'): return f"input={e['input']} output={e['output']}"
    if op == 'nms': return 'boxes=N,4 scores=C,N max_output=300'
    return str(e)

def estimate(group):
    op, n = group[0]['op'], len(group)
    if op == 'conv':
        return sum(e['m'] * e['n'] * e['k'] for e in group) * conv_rate, 'measured'
    if op in rates:
        return sum(math.prod(e.get('input') or e.get('lhs') or [1]) for e in group) * rates[op], 'measured'
    if op in post:
        return n * post[op], 'measured(shape不同,按节点数)'
    if op == 'argmax':
        return LEGACY_ARGMAX, 'legacy占位(无直测)'
    if op == 'transpose':
        return 0, '未测(整网占比~0,legacy也为0)'
    if op == 'nms':
        return nms_net * nms_scale, 'measured比例外推(3boxes→84x80)'
    return 0, 'skipped'

groups = {}
for e in entries:
    groups.setdefault((e['op'], shape_text(e)), []).append(e)
by_op, rows_by_op = {}, []
for (op, shape), g in groups.items():
    est, src = estimate(g)
    by_op.setdefault(op, [0, 0, ''])
    by_op[op][0] += est; by_op[op][1] += len(g); by_op[op][2] = src
    rows_by_op.append({'op': op, 'shape': shape, 'count': len(g),
                       'estimated_cycles': round(est, 2), 'source': src})
total = sum(v[0] for v in by_op.values())
op_rows = [{'op': op, 'shape': f'count={v[1]}', 'count': v[1],
            'estimated_cycles': round(v[0], 2), 'source': v[2]}
           for op, v in sorted(by_op.items(), key=lambda kv: -kv[1][0])]
op_rows.append({'op': 'TOTAL', 'shape': '', 'count': sum(r['count'] for r in op_rows),
                'estimated_cycles': round(total, 2), 'source': ''})
rows_by_op.append(op_rows[-1])

with (out_dir / 'estimated_total_cycles.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['op', 'shape', 'count', 'estimated_cycles', 'source'])
    w.writeheader(); w.writerows(op_rows)
with (out_dir / 'estimated_operator_cycles.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['op', 'shape', 'count', 'estimated_cycles', 'source'])
    w.writeheader(); w.writerows(rows_by_op)

print(f"\n{'op':<12} {'count':>6} {'est_cycles':>16}  source")
for r in op_rows:
    print(f"{r['op']:<12} {r['count']:>6} {r['estimated_cycles']:>16}  {r['source']}")
print(f"\nTOTAL estimated_cycles={total:.0f}")
print(out_dir / 'estimated_total_cycles.csv')
print(out_dir / 'estimated_operator_cycles.csv')
PY
}

direct_ops() {
  OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/yolo_representative/yolov8n_64_dayan}"
  mkdir -p "${OUT_DIR}"
  DIRECT_CSV="${OUT_DIR}/samples_dayan.csv"
  echo "name,tool,args,pass,raw_cycles,net_cycles" >"${DIRECT_CSV}"

  IFS=':' read -r CONV_IN CONV_W CONV_S CONV_P <<<"${CONV_SPEC}"
  local conv_args="--input-shape ${CONV_IN} --weight-shape ${CONV_W} --stride ${CONV_S} --pad ${CONV_P} --bias"

  # 大 conv 的中断积聚会污染后续样本的周期读数，必须最后运行。
  local samples=(
    "add|test_rvnpu_elementwise|--op add --shape 2,3,5 --rhs-shape 2,3,5"
    "add_mid|test_rvnpu_elementwise|--op add --shape 1,64,2,2 --rhs-shape 1,64,2,2"
    "div|test_rvnpu_elementwise|--op div --shape 1,2,16 --rhs-shape 1"
    "sigmoid|test_rvnpu_elementwise|--op sigmoid --shape 2,3,5"
    "sigmoid_mid|test_rvnpu_elementwise|--op sigmoid --shape 1,64,2,2"
    "mul|test_rvnpu_elementwise|--op mul --shape 2,3,5 --rhs-shape 2,3,5"
    "mul_mid|test_rvnpu_elementwise|--op mul --shape 1,64,2,2 --rhs-shape 1,64,2,2"
    "maxpool|test_rvnpu_maxpool| --input-shape 1,3,8,9 --kernel 2,2 --stride 2,2 --pad 0,0"
    "softmax|test_rvnpu_transform|--op softmax"
    "reduce_max|test_rvnpu_transform|--op reduce_max"
    "resize|test_rvnpu_transform|--op resize"
    "nms|test_rvnpu_nms|--case nonoverlap"
    "conv|test_rvnpu_conv|${conv_args}"
  )

  for s in "${samples[@]}"; do
    IFS='|' read -r name tool args <<<"$s"
    run_direct_sample "${name}" "${tool}" ${args}
  done

  echo
  direct_ops_summarize
}


if [[ ${DIRECT_OPS} == 1 ]]; then
  if ! tc_available; then
    echo "错误：--direct-ops 需要 tmux 会话 ${TC_SESSION} 里在跑 TestCluster（周期从 TUI 读）。" >&2
    exit 2
  fi
  direct_ops
  exit ${?}
fi

if [[ $# -gt 0 ]]; then
  OPS=("$@")
else
  OPS=(add concat conv div matmul maxpool mul nms reduce_max resize
       selection sigmoid silu slice softmax split sub topk transpose
       yolov8n)
fi

if [[ ${REPORT} == 1 ]] && ! tc_available; then
  echo "警告：--report 需要 tmux 会话 ${TC_SESSION} 里在跑 TestCluster，当前没找到，只输出尺寸。"
fi

fail=0
for op in "${OPS[@]}"; do
  run_op "${op}" || fail=1
done
print_summary "${fail}"
exit ${fail}
