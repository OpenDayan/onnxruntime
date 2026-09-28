#!/usr/bin/env bash
# 迁移链路（dayan）的算子回归：把 build/linux 指到 dayan-migration 产物后，
# 原样执行 tests/operators/<op>/run.sh。
#
# 前置：dayan_server 已在跑（DAYAN_SERVER_SOCKET，默认 /tmp/dayan-server.sock）。
# 用法：run_dayan_tests.sh [--report] [op ...]
#       不带参数跑除 yolov8n* 外的全部算子（yolo 设备超时，速度问题解决前默认跳过）。
#
#   --report   打印每个算子的测试张量尺寸，并测出该算子消耗的仿真周期，末尾汇总成表。
#              等价于 DAYAN_REPORT=1。
#              周期读数取自 TestCluster 的 TUI（tmux 会话 DAYAN_TC_SESSION，默认 sim），
#              刷新粒度约 1 秒，因此耗时很短的算子可能读到 0，只能算上界。
#              汇总同时写入 ${DAYAN_CYCLE_CSV}（默认 /tmp/dy-op-cycles.csv）。
#
#   --direct-ops  dayan 版 test_direct_ops：跑一组小尺寸代表算子（直测工具，自带数值
#                 校验），每个样本用 TUI Cycle 差值测周期（扣 22k 进程固定开销），再按
#                 legacy test_direct_ops.sh 的速率模型外推 yolov8n_64 整网总周期。
#                 结果写 build/yolo_representative/yolov8n_64_dayan/。
#     --conv S    覆盖 conv 样本尺寸，格式 "IN:W:STRIDE:PAD"，如
#                 "1,3,15,17:5,3,3,3:2,2:1,1"（默认即此值；换大尺寸可提高 conv 速率
#                 精度，conv 占整网估算 ~84%，是主要误差源）。分隔符用冒号，分号在
#                 ssh/zsh 多层引号下会被当命令分隔。
#     --out DIR   结果目录（默认 build/yolo_representative/yolov8n_64_dayan）。
#
# 注意：RTL 目前没有实现周期时间戳寄存器（device/include/platform_regs.h 里的
# DAYAN_PCU_CYCLE 只是预留槽位），npu_profile_pcu_read() 会返回 -ENODATA，
# 所以周期只能这样从 TestCluster 侧量，走不了 dyperf --pcu。
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHIM=/tmp/dy-tests
TC_SESSION="${DAYAN_TC_SESSION:-sim}"
CYCLE_CSV="${DAYAN_CYCLE_CSV:-/tmp/dy-op-cycles.csv}"

REPORT=0
[[ "${DAYAN_REPORT:-0}" == "1" ]] && REPORT=1
DIRECT_OPS=0
CONV_SPEC="1,3,15,17;5,3,3,3;2,2;1,1"
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
    -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            POS+=("$1") ;;
  esac
  shift
done
set -- ${POS[@]+"${POS[@]}"}

export LD_LIBRARY_PATH="${ROOT_DIR}/build/ort-sdk/lib:${LD_LIBRARY_PATH:-}"
export DAYAN_SERVER_SOCKET="${DAYAN_SERVER_SOCKET:-/tmp/dayan-server.sock}"
export RVNPU_KERNEL_ELF=/tmp/dummy-kernel-elf

setup_shim() {
  rm -rf "${SHIM}"
  mkdir -p "${SHIM}/build"
  ln -s "${ROOT_DIR}/tests" "${SHIM}/tests"
  ln -s "${ROOT_DIR}/tools" "${SHIM}/tools" 2>/dev/null
  ln -s "${ROOT_DIR}/scripts" "${SHIM}/scripts" 2>/dev/null
  ln -s "${ROOT_DIR}/build/dayan-migration" "${SHIM}/build/linux"
  ln -s "${ROOT_DIR}/build/models" "${SHIM}/build/models"
  ln -s "${ROOT_DIR}/build/yolov8n" "${SHIM}/build/yolov8n" 2>/dev/null
  for d in operators yolo_representative; do
    ln -s "${ROOT_DIR}/build/${d}" "${SHIM}/build/${d}" 2>/dev/null
  done
  : > /tmp/dummy-kernel-elf
}

# ---------- TestCluster 周期读数 ----------

tc_available() {
  command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TC_SESSION}" 2>/dev/null
}

# TUI 里的 `Cycle249688/20000000` 是当前仿真周期，取斜杠前的数字。
# 窄窗口下会截断成 `Cycl1291665/49`，两种形态都要匹配。
tc_cycles() {
  tmux capture-pane -t "${TC_SESSION}" -p 2>/dev/null \
    | grep -a -o -E 'Cycl[a-z]*[0-9]+/' | head -1 | tr -dc '0-9'
}

# ---------- 测试张量尺寸 ----------

# 形状写在 onnx 模型里的算子：直接读模型，打印每个测试模型的输入张量形状。
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

# 优先用 run.sh 里直接调用写死的形状；没有就读 onnx 模型。
op_sizes() {
  local op="$1" rs="${SHIM}/tests/operators/${op}/run.sh" out=""
  if [[ -f "${rs}" ]]; then
    out="$(grep -o -E \
        -- '--(shape|rhs-shape|input-shape|weight-shape) [0-9,x]+|--m [0-9]+ --n [0-9]+ --k [0-9]+|--rows [0-9]+ --columns [0-9]+ --k [0-9]+' \
        "${rs}" 2>/dev/null | paste -sd'|' -)"
  fi
  [[ -z "${out}" ]] && out="$(model_sizes "${op}")"
  [[ -z "${out}" ]] && out="-"
  # 太长的截断，表格里放不下
  if (( ${#out} > 150 )); then out="${out:0:147}..."; fi
  printf '%s' "${out}"
}

# ---------- 跑算子 ----------

ROWS=()

run_op() {
  local op="$1"
  local log="/tmp/dy-op-${op}.log"
  local sizes="-" c0="" c1="" cycles="-" t0 t1 wall rc

  if [[ ${REPORT} == 1 ]]; then sizes="$(op_sizes "${op}")"; fi

  echo "########## ${op}"
  [[ ${REPORT} == 1 ]] && echo "  sizes: ${sizes}"

  if [[ -f "${SHIM}/tests/operators/${op}/prepare.sh" ]]; then
    if ! bash "${SHIM}/tests/operators/${op}/prepare.sh" >>"${log}.prepare" 2>&1; then
      echo "PREPARE FAIL ${op}"
      ROWS+=("${op}|${sizes}|-|-|PREPARE_FAIL")
      return 1
    fi
  fi

  if [[ ${REPORT} == 1 ]] && tc_available; then c0="$(tc_cycles)"; fi
  t0=$(date +%s)
  if bash "${SHIM}/tests/operators/${op}/run.sh" >"${log}" 2>&1; then rc=0; else rc=1; fi
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
  local total=0 ok=1
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

# ---------- dayan 版 test_direct_ops（--direct-ops） ----------

# 每个直测样本一次进程：TUI Cycle 前后差值，再扣掉进程固定开销（约 22k，含固件
# 上下文加载/启动），得到该算子样本的净周期。TUI 刷新 ~1s，短样本读数偏粗。
DAYAN_FIXED_CYCLES=22000

run_direct_sample() {
  local name="$1" tool="$2"; shift 2
  local args="$*" c0 c1 raw net rc attempt
  echo "---------- sample ${name}: ${tool} ${args}"
  # 冷启动竞态是概率性的（warmup 压不住），改为 FAIL 才重试：
  # 正常路径只跑一次，数值错时最多补到 3 次，取最后一次通过的读数。
  for attempt in 1 2 3; do
    c0="$(tc_cycles)"
    if timeout 600 "${ROOT_DIR}/build/dayan-migration/${tool}" \
          --kernel-elf /tmp/dummy-kernel-elf ${args} >"/tmp/dy-do-${name}.log" 2>&1; then
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
      echo "FAIL ${name}  net_cycles=${net}  raw=${raw}  (log /tmp/dy-do-${name}.log)"
      tail -2 "/tmp/dy-do-${name}.log" | sed 's/^/    /'
    fi
  done
  echo "${name},${tool},\"${args// /_}\",${rc},${raw},${net}" >>"${DIRECT_CSV}"
}

direct_ops_summarize() {
  python3 - "${OUT_DIR:-${ROOT_DIR}/build/yolo_representative/yolov8n_64_dayan}" <<'PY'
import csv, json, math, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
manifest = json.loads(Path('/workspace/onnxruntime-rvnpu/build/yolov8n_64/perf_manifest.json').read_text())
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
    # 取元素数最大的样本（mid 档）：小样本的净周期被固定开销残差主导，
    # cycles/elems 会虚高一个数量级。
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
  local conv_args="--op conv --input-shape ${CONV_IN} --weight-shape ${CONV_W} --stride ${CONV_S} --pad ${CONV_P} --bias"

  # 对齐 legacy samples.csv 的算子覆盖。silu 在 yolo 图里表达为 sigmoid+mul，
  # 不单测；transpose 整网占比 ~0 且 legacy 计数也是 0，不测；argmax 无直测入口。
  # conv 放最后：大 conv 会让设备进入中断堆积的劣化态（后续样本周期被空转
  # 计数污染，maxpool 曾虚高 11 倍），先测小样本再测 conv，劣化只影响它自己。
  local samples=(
    "add|test_rvnpu_elementwise|--op add --shape 2,3,5 --rhs-shape 2,3,5"
    "add_mid|test_rvnpu_elementwise|--op add --shape 1,64,2,2 --rhs-shape 1,64,2,2"
    "div|test_rvnpu_elementwise|--op div --shape 1,2,16 --rhs-shape 1"
    "sigmoid|test_rvnpu_elementwise|--op sigmoid --shape 2,3,5"
    "sigmoid_mid|test_rvnpu_elementwise|--op sigmoid --shape 1,64,2,2"
    "mul|test_rvnpu_elementwise|--op mul --shape 2,3,5 --rhs-shape 2,3,5"
    "mul_mid|test_rvnpu_elementwise|--op mul --shape 1,64,2,2 --rhs-shape 1,64,2,2"
    "maxpool|test_rvnpu_conv_maxpool|--op maxpool --input-shape 1,3,8,9 --kernel 2,2 --stride 2,2 --pad 0,0"
    "softmax|test_rvnpu_transform|--op softmax"
    "reduce_max|test_rvnpu_transform|--op reduce_max"
    "resize|test_rvnpu_transform|--op resize"
    "nms|test_rvnpu_nms|--case nonoverlap"
    "conv|test_rvnpu_conv_maxpool|${conv_args}"
  )

  for s in "${samples[@]}"; do
    IFS='|' read -r name tool args <<<"$s"
    run_direct_sample "${name}" "${tool}" ${args}
  done

  echo
  direct_ops_summarize
}

setup_shim

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
       selection sigmoid silu slice softmax split sub topk transpose)
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
