#!/usr/bin/env python3
import csv, re, sys

def nk(op, i, w, s, p):
    s2 = ",".join(x.strip() for x in s.split(",")[:2])
    p2 = ",".join(x.strip() for x in p.split(",")[:2])
    return (op, i.replace(" ", ""), w.replace(" ", ""), s2, p2)

done = {}
for r in csv.reader(open("/workspace/onnxruntime-rvnpu/build/operator_matrix/results.csv")):
    if len(r) < 5 or r[0] == "net":
        continue
    net, tag, args, rc, cyc = r[0], r[1], r[2], r[3], r[4]
    m = re.search(r"input-shape ([0-9,]+) --weight-shape ([0-9,]+) --stride ([0-9,]+) --pad ([0-9,]+)", args)
    if m:
        done[(net, nk("conv", m.group(1), m.group(2), m.group(3), m.group(4)))] = (rc, cyc, tag)
    m2 = re.search(r"--op (sigmoid|mul|add|div) --shape ([0-9,x]+)", args)
    if m2:
        rhs = re.search(r"--rhs-shape ([0-9,x]+)", args)
        sh = m2.group(2).replace("x", ",")
        done[(net, (m2.group(1), sh, rhs.group(1).replace("x", ",") if rhs else ""))] = (rc, cyc, tag)

rows64, order64 = {}, []
for r in csv.DictReader(open("/workspace/onnxruntime-rvnpu/build/yolo_representative/yolov8n_64_dayan/estimated_operator_cycles.csv")):
    if r["op"] == "TOTAL":
        continue
    mi = re.search(r"input=\[([0-9, ]+)\] weight=\[([0-9, ]+)\] stride=\[([0-9, ]+)\] pad=\[([0-9, ]+)\]", r["shape"])
    ms = re.search(r"input.: \[([0-9, ]+)\]", r["shape"])
    mm = re.search(r"lhs=\[([0-9, ]+)\]", r["shape"])
    if r["op"] == "conv" and mi:
        key = nk("conv", mi.group(1), mi.group(2), mi.group(3), mi.group(4))
        disp = "%s×%s s%s p%s" % (mi.group(1).replace(" ", ""), mi.group(2).replace(" ", ""),
                                  ",".join(mi.group(3).split()[:1]) or mi.group(3).replace(" ", ""),
                                  mi.group(4).replace(" ", "")[:3])
        disp = "%s×%s" % (mi.group(1).replace(" ", ""), mi.group(2).replace(" ", ""))
        st = mi.group(3).replace(" ", "")
        if st == "2,2":
            disp += " s2"
        if "3, 3" in r["shape"] and mi.group(2).replace(" ", "").endswith("3,3"):
            pass
    elif r["op"] == "sigmoid" and ms:
        key = ("sigmoid", ms.group(1).replace(" ", ""), "")
        disp = ms.group(1).replace(" ", "")
    elif r["op"] in ("mul", "add") and mm:
        sh = mm.group(1).replace(" ", "")
        key = (r["op"], sh, sh)
        disp = sh
    elif r["op"] == "div":
        key = ("div", "1", "1")
        disp = "标量"
    else:
        key = (r["op"], r["shape"][:20], "")
        disp = r["shape"][:44].replace("'", "").replace("{", "").replace("}", "")
    if key not in rows64:
        rows64[key] = [r["op"], disp, int(r["count"])]
        order64.append(key)

out = []
out.append("## 表一:64 网(yolov8n_64)全部 %d 个不同形状" % len(order64))
out.append("| # | 算子 | 形状(输入×权重 / 张量) | 网内节点数 | 状态 | 实测周期 |")
out.append("|---|---|---|---|---|---|")
missing64 = []
for i, key in enumerate(order64, 1):
    op, disp, cnt = rows64[key]
    hit = done.get(("y64", key))
    if hit:
        st = {"0": "PASS", "9": "跳过"}.get(hit[0], "FAIL")
        out.append("| %d | %s | %s | %d | %s | %s |" % (i, op, disp, cnt, st, hit[1]))
    else:
        out.append("| %d | %s | %s | %d | **待补** | - |" % (i, op, disp, cnt))
        missing64.append((op, key, cnt, disp))

# 640 网
rows640, order640 = {}, []
for r in csv.reader(open("/workspace/onnxruntime-rvnpu/build/yolov8n/perf_samples.csv")):
    if r[0] == "op":
        continue
    op, inp, w, stride, pad, rhs, cnt = r[0], r[1], r[2], r[3], r[4], r[5], int(r[6])
    if op == "conv":
        key = nk("conv", inp.replace("x", ","), w.replace("x", ","), stride.replace("x", ","), pad.replace("x", ","))
        disp = "%s×%s" % (inp, w)
        if stride == "2x2":
            disp += " s2"
    elif op in ("sigmoid",):
        sh = inp.replace("x", ",")
        key = ("sigmoid", sh, "")
        disp = inp
    elif op in ("mul", "add"):
        sh = inp.replace("x", ",")
        rh = (rhs or sh).replace("x", ",")
        key = (op, sh, rh)
        disp = inp + ("×" + rhs if rhs and rhs != inp else "")
    elif op == "div":
        key = ("div", "1", "1")
        disp = "标量"
    else:
        key = (op, inp, "")
        disp = "%s %s %s" % (inp, w, stride)
    if key not in rows640:
        rows640[key] = [op, disp, cnt]
        order640.append(key)

out.append("")
out.append("## 表二:640 网(标准 yolov8n)全部 %d 个不同形状" % len(order640))
out.append("| # | 算子 | 形状(输入×权重 / 张量) | 网内节点数 | 状态 | 实测周期(缩放尺寸) |")
out.append("|---|---|---|---|---|---|")
missing640 = []
for i, key in enumerate(order640, 1):
    op, disp, cnt = rows640[key]
    hit = done.get(("y640", key))
    if hit:
        st = {"0": "PASS", "9": "跳过"}.get(hit[0], "FAIL")
        out.append("| %d | %s | %s | %d | %s | %s |" % (i, op, disp, cnt, st, hit[1]))
    else:
        out.append("| %d | %s | %s | %d | **待补** | - |" % (i, op, disp, cnt))
        missing640.append((op, key, cnt, disp))

open("/workspace/onnxruntime-rvnpu/build/operator_matrix/operator_matrix_table.md", "w").write("\n".join(out))
with open("/tmp/missing.txt", "w") as f:
    f.write("=== 64 网待补 %d 个 ===\n" % len(missing64))
    for op, key, cnt, disp in missing64:
        f.write("y64|%s|%s|%d|%s\n" % (op, "|".join(str(k) for k in key[1:]), cnt, disp))
    f.write("\n=== 640 网待补 %d 个 ===\n" % len(missing640))
    for op, key, cnt, disp in missing640:
        f.write("y640|%s|%s|%d|%s\n" % (op, "|".join(str(k) for k in key[1:]), cnt, disp))
print("table:", len(order64), "+", len(order640), "rows; missing:", len(missing64), "+", len(missing640))

import re as _re
_p = "/workspace/onnxruntime-rvnpu/build/operator_matrix/operator_matrix_table.md"
_fixed = {"softmax": "15.5万(工具固定)", "transpose": "38.3万(工具固定)",
          "resize": "4.8万(工具固定)", "reduce_max": "3.8万(工具固定)",
          "argmax": "38.7万(topk近似)", "nms": "2.6万(预设case)"}
_lines = open(_p).read().splitlines()
_out = []
for _ln in _lines:
    _m = _re.match(r"\| (\d+) \| ([a-z_]+) \|", _ln)
    if _m and "**待补**" in _ln and _m.group(2) in _fixed:
        _ln = _ln.replace("**待补** | - |", "工具固定形状 | " + _fixed[_m.group(2)] + " |")
    _out.append(_ln)
open(_p, "w").write("\n".join(_out) + "\n")
