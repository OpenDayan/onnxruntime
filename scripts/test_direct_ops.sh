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
manifest_path=Path(f'/workspace/onnxruntime-rvnpu/build/yolov8n_{size}/perf_manifest.json')
manifest=json.loads(manifest_path.read_text())
entries=manifest['graph_entries']
rows=[]
for p in sorted(out.glob('group_*/operators.csv')):
    rows += list(csv.DictReader(p.open()))

def elems(text): return math.prod(map(int,text.split('x'))) if text else 1
result=[]; grand=0.0
conv=next(r for r in rows if r['op']=='conv')
conv_macs=8*8*64*3*3*64
conv_rate=int(conv['cycles'])/conv_macs
conv_total_macs=sum(e['m']*e['n']*e['k'] for e in entries if e['op']=='conv')
conv_est=conv_total_macs*conv_rate; grand+=conv_est
result.append({'op':'conv','sample_input':conv['input'],'sample_weight':conv['weight'],'sample_cycles':conv['cycles'],'graph_count':sum(1 for e in entries if e['op']=='conv'),'estimated_cycles':conv_est})
for op in ['add','mul','sigmoid','div']:
    candidates=[r for r in rows if r['op']==op]
    if not candidates: continue
    sample=max(candidates,key=lambda r:int(r['cycles']))
    rate=int(sample['cycles'])/elems(sample['input'])
    total_elems=sum(elems('x'.join(map(str,e.get('input') or e.get('lhs')))) for e in entries if e['op']==op)
    est=rate*total_elems; grand+=est
    result.append({'op':op,'sample_input':sample['input'],'sample_weight':sample['weight'],'sample_cycles':sample['cycles'],'graph_count':sum(1 for e in entries if e['op']==op),'estimated_cycles':est})
for op in ['maxpool','resize','transpose','nms','softmax','argmax','reduce_max']:
    candidates=[r for r in rows if r['op']==op]
    if not candidates: continue
    total_est=sum(int(r['cycles']) for r in candidates)
    grand+=total_est
    sample=max(candidates,key=lambda r:int(r['cycles']))
    result.append({'op':op,'sample_input':sample['input'],'sample_weight':sample['weight'],'sample_cycles':sample['cycles'],'graph_count':sum(1 for e in entries if e['op']==op),'estimated_cycles':total_est})
fields=['op','sample_input','sample_weight','sample_cycles','graph_count','estimated_cycles']
with (out/'estimated_total_cycles.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(result)
print(f'TOTAL input_size={size} estimated_cycles={grand:.0f}')
print(out/'estimated_total_cycles.csv')
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