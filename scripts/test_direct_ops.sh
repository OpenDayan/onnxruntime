#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ELF="${RVNPU_KERNEL_ELF:-/workspace/npu-setup/workspace/npu-kernels/bin/kernels_pack.elf}"
PERF="${ROOT_DIR}/build/linux/test_rvnpu_yolo_perf"
YOLO_SIZE=64

usage() {
  cat >&2 <<'EOF'
用法:
  test_direct_ops.sh [--yolo-size 64|96|160] profile

说明:
  只保留 YOLO 代表算子 profile。脚本不会启动或重启 TestCluster；
  请先手动启动 TestCluster 并输入 c，再运行本脚本。

输出:
  build/yolo_representative/yolov8n_<size>/group_*/
  build/yolo_representative/yolov8n_<size>/estimated_total_cycles.csv
EOF
}

while (($#)); do
  case "$1" in
    --yolo-size)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      YOLO_SIZE="$2"
      shift 2
      ;;
    profile) shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "${YOLO_SIZE}" == 64 || "${YOLO_SIZE}" == 96 || "${YOLO_SIZE}" == 160 ]] || {
  echo "不支持的 YOLO 尺寸: ${YOLO_SIZE}" >&2
  exit 2
}
[[ -x "${PERF}" ]] || { echo "缺少 runner: ${PERF}" >&2; exit 2; }
[[ -e "${ELF}" ]] || { echo "缺少 kernel ELF: ${ELF}" >&2; exit 2; }

MODEL_DIR="${ROOT_DIR}/build/yolov8n_${YOLO_SIZE}"
MANIFEST="${MODEL_DIR}/perf_manifest.json"
MODEL="${MODEL_DIR}/yolov8n_${YOLO_SIZE}.onnx"
OUT="${ROOT_DIR}/build/yolo_representative/yolov8n_${YOLO_SIZE}"
mkdir -p "${OUT}"

if [[ ! -e "${MODEL}" ]]; then
  mkdir -p "${MODEL_DIR}"
  python3 - "${ROOT_DIR}" "${YOLO_SIZE}" <<'PY'
from pathlib import Path
from ultralytics import YOLO
import shutil
import sys
root = Path(sys.argv[1])
size = int(sys.argv[2])
outdir = root / "build" / f"yolov8n_{size}"
target = outdir / f"yolov8n_{size}.onnx"
model = YOLO(str(root / "build/yolov8n/yolov8n.pt"))
exported = model.export(
    format="onnx", imgsz=size, batch=1, dynamic=False, simplify=False,
    opset=13, nms=True, conf=.25, iou=.7, max_det=300,
    agnostic_nms=False, device="cpu")
shutil.move(str(exported), str(target))
print(f"generated {target}")
PY
fi
if [[ ! -s "${MANIFEST}" || "${MODEL}" -nt "${MANIFEST}" ]]; then
  python3 "${ROOT_DIR}/tools/extract_yolo_perf_manifest.py" "${MODEL}" -o "${MANIFEST}" || exit 2
fi


write_samples() {
  cat > "${OUT}/samples.csv" <<'EOF'
op,input,weight,stride,pad,rhs,count
add,1x2x16,,,,1x2x16,1
add,1x64x2x2,,,,1x64x2x2,1
div,1,,,,1,1
maxpool,1x128x2x2,5x5,1x1,2x2x2x2,,1
sigmoid,1x2x16,,,,,1
sigmoid,1x64x2x2,,,,,1
mul,1x2x16,,,,1x2x16,1
mul,1x64x2x2,,,,1x64x2x2,1
softmax,1x16x4x84,,,,,1
argmax,1x84x80,,,,,1
reduce_max,1x84x80,,,,,1
conv,1x64x8x8,64x64x3x3,1x1,1x1x1x1,,1
EOF

  case "${YOLO_SIZE}" in
    64)
      cat >> "${OUT}/samples.csv" <<'EOF'
resize,1x256x2x2,,,,,1
resize,1x128x4x4,,,,,1
transpose,1x4x16x84,,,,0x2x1x3,1
transpose,1x84x84,,,,0x2x1,1
nms,16,16,16,45,25,1
EOF
      ;;
    96)
      cat >> "${OUT}/samples.csv" <<'EOF'
resize,1x256x3x3,,,,,1
resize,1x128x6x6,,,,,1
transpose,1x4x16x189,,,,0x2x1x3,1
transpose,1x84x189,,,,0x2x1,1
nms,16,16,16,45,25,1
EOF
      ;;
    160)
      cat >> "${OUT}/samples.csv" <<'EOF'
resize,1x256x5x5,,,,,1
resize,1x128x10x10,,,,,1
transpose,1x4x16x525,,,,0x2x1x3,1
transpose,1x84x525,,,,0x2x1,1
nms,16,16,16,45,25,1
EOF
      ;;
  esac
}

make_group_csv() {
  local start="$1" end="$2" dst="$3"
  python3 - "${OUT}/samples.csv" "${start}" "${end}" "${dst}" <<'PY'
import csv
import sys
src,start,end,dst=sys.argv[1:]
start,end=int(start),int(end)
rows=list(csv.DictReader(open(src)))
with open(dst,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['op','input','weight','stride','pad','rhs','count'])
  w.writeheader(); w.writerows(rows[start:end+1])
PY
}

summarize() {
  python3 - "${OUT}" "${YOLO_SIZE}" <<'PY'
import csv, json, math, sys
from pathlib import Path
out=Path(sys.argv[1]); size=int(sys.argv[2])
manifest=json.loads(Path(f'/workspace/onnxruntime-rvnpu/build/yolov8n_{size}/perf_manifest.json').read_text())
entries=manifest['graph_entries']
rows=[]
for p in sorted(out.glob('group_*/operators.csv')):
    rows += list(csv.DictReader(p.open()))

def elems(text): return math.prod(map(int,text.split('x'))) if text else 1
def shape_text(e):
    op=e['op']
    if op=='conv': return f"input={e['input']} weight={e['weight']} stride={e['stride']} pad={e['pad']}"
    if op=='maxpool': return f"input={e['input']} kernel={e['kernel']} stride={e['stride']} pad={e['pad']}"
    if op=='transpose': return f"input={e['input']} perm={e['perm']} output={e['output']}"
    if op in ('add','mul','div'): return f"lhs={e.get('lhs')} rhs={e.get('rhs')} output={e.get('output')}"
    if op=='resize': return f"input={e['input']} output={e['output']}"
    if op in ('softmax','argmax','reduce_max'): return f"input={e['input']} output={e['output']}"
    if op=='nms': return 'boxes=N,4 scores=C,N max_output=300'
    return str(e)

conv=next(r for r in rows if r['op']=='conv')
conv_macs=8*8*64*3*3*64
conv_rate=int(conv['cycles'])/conv_macs
rates={}
for op in ['add','mul','sigmoid','div']:
    candidates=[r for r in rows if r['op']==op]
    if candidates:
        sample=max(candidates,key=lambda r:int(r['cycles']))
        rates[op]=int(sample['cycles'])/elems(sample['input'])
post_rates={}
for op in ['maxpool','softmax','argmax','reduce_max']:
    candidates=[r for r in rows if r['op']==op]
    if candidates: post_rates[op]=int(max(candidates,key=lambda r:int(r['cycles']))['cycles'])
resize_candidates=[r for r in rows if r['op']=='resize']
resize_rate=sum(int(r['cycles']) for r in resize_candidates)/max(1,len(resize_candidates))
transpose_candidates=[r for r in rows if r['op']=='transpose']
transpose_cycles=int(max(transpose_candidates,key=lambda r:int(r['cycles']))['cycles']) if transpose_candidates else 0
nms_sample=next(r for r in rows if r['op']=='nms')
nms_cycles=int(nms_sample['cycles'])*((84*80)/(16*16))

result=[]
for op in sorted({e['op'] for e in entries}):
    group=[e for e in entries if e['op']==op]
    if op=='conv': est=sum(e['m']*e['n']*e['k'] for e in group)*conv_rate
    elif op in rates: est=sum(elems('x'.join(map(str,e.get('input') or e.get('lhs')))) for e in group)*rates[op]
    elif op in post_rates: est=len(group)*post_rates[op]
    elif op=='resize': est=len(group)*resize_rate
    elif op=='transpose': est=len(group)*transpose_cycles
    elif op=='nms': est=len(group)*nms_cycles
    else: continue
    result.append({'op':op,'shape':f'count={len(group)}','count':len(group),'estimated_cycles':round(est,2)})
result.append({'op':'TOTAL','shape':'','count':sum(r['count'] for r in result),'estimated_cycles':round(sum(float(r['estimated_cycles']) for r in result),2)})
with (out/'estimated_total_cycles.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['op','shape','count','estimated_cycles']); w.writeheader(); w.writerows(result)

shape_rows=[]
groups={}
for e in entries:
    groups.setdefault((e['op'],shape_text(e)),[]).append(e)
for (op,shape),group in groups.items():
    if op=='conv': est=sum(e['m']*e['n']*e['k'] for e in group)*conv_rate
    elif op in rates: est=sum(elems('x'.join(map(str,e.get('input') or e.get('lhs')))) for e in group)*rates[op]
    elif op in post_rates: est=len(group)*post_rates[op]
    elif op=='resize': est=len(group)*resize_rate
    elif op=='transpose': est=len(group)*transpose_cycles
    elif op=='nms': est=len(group)*nms_cycles
    else: continue
    shape_rows.append({'op':op,'shape':shape,'count':len(group),'estimated_cycles':round(est,2)})
shape_rows.append({'op':'TOTAL','shape':'','count':sum(r['count'] for r in shape_rows),'estimated_cycles':round(sum(float(r['estimated_cycles']) for r in shape_rows),2)})
with (out/'estimated_operator_cycles.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['op','shape','count','estimated_cycles']); w.writeheader(); w.writerows(shape_rows)
print(f'TOTAL input_size={size} estimated_cycles={sum(float(r["estimated_cycles"]) for r in result[:-1]):.0f}')
print(out/'estimated_total_cycles.csv')
print(out/'estimated_operator_cycles.csv')
PY
}

run_group() {
  local name="$1" start="$2" end="$3"
  local dir="${OUT}/group_${name}"
  local log="${dir}/run.log"
  mkdir -p "${dir}"
  make_group_csv "${start}" "${end}" "${dir}/samples.csv"

  if [[ -s "${dir}/operators.csv" && -s "${dir}/cycles.json" ]]; then
    echo "SKIP ${name}: existing result found"
    return 0
  fi

  echo "---------- profile-yolo-${YOLO_SIZE}-${name} ----------"
  "${PERF}" "${ELF}" "${dir}/samples.csv" \
    "${dir}/cycles.json" "${dir}/operators.csv" \
    --proxy-elements 4096 2>&1 | tee "${log}"
}

write_samples
run_group add 0 0
run_group add_mid 1 1
run_group div 2 2
run_group maxpool 3 3
run_group sigmoid 4 4
run_group sigmoid_mid 5 5
run_group mul 6 6
run_group mul_mid 7 7
run_group post 8 10
run_group conv 11 11
run_group resize 12 13
run_group transpose 14 15
run_group nms 16 16
summarize
