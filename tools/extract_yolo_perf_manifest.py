#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
from collections import Counter
import onnx
from onnx import shape_inference, numpy_helper, helper

parser=argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('-o','--output',required=True)
args=parser.parse_args()

m=shape_inference.infer_shapes(onnx.load(args.model))
shapes={}
for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output):
    dims=[]
    for d in v.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField('dim_value') else d.dim_param)
    if all(isinstance(x,int) for x in dims): shapes[v.name]=dims
inits={x.name:numpy_helper.to_array(x).shape for x in m.graph.initializer}

entries=[]
def attrs(node):
    return {a.name:helper.get_attribute_value(a) for a in node.attribute}

for n in m.graph.node:
    op=n.op_type
    if op=='Conv':
        inp=shapes.get(n.input[0]); w=inits.get(n.input[1]); out=shapes.get(n.output[0])
        if not inp or not w or not out: continue
        a=attrs(n)
        strides=a.get('strides',[1,1]); pads=a.get('pads',[0,0,0,0])
        entries.append(dict(op='conv', input=inp, weight=list(w), stride=list(strides), pad=list(pads), output=out,
                            m=out[2]*out[3], n=w[0], k=w[1]*w[2]*w[3]))
    elif op in ('Add','Mul','Sub'):
        out=shapes.get(n.output[0]); ins=[shapes.get(x) or list(inits.get(x,[])) for x in n.input]
        if out and all(ins): entries.append(dict(op=op.lower(), lhs=ins[0], rhs=ins[1], output=out))
    elif op=='Sigmoid':
        out=shapes.get(n.output[0]); inp=shapes.get(n.input[0])
        if inp: entries.append(dict(op='sigmoid', input=inp, output=out or inp))
    elif op=='Div':
        ins=[shapes.get(x) or list(inits.get(x,[])) for x in n.input]
        if all(ins): entries.append(dict(op='div', lhs=ins[0], rhs=ins[1], output=shapes.get(n.output[0])))
    elif op=='MaxPool':
        inp=shapes.get(n.input[0]); out=shapes.get(n.output[0]); a=attrs(n)
        if inp: entries.append(dict(op='maxpool', input=inp, output=out, kernel=list(a.get('kernel_shape',[])), stride=list(a.get('strides',[])), pad=list(a.get('pads',[]))))
    elif op=='Resize':
        inp=shapes.get(n.input[0]); out=shapes.get(n.output[0])
        if inp and out: entries.append(dict(op='resize', input=inp, output=out))
    elif op=='Transpose':
        inp=shapes.get(n.input[0]); out=shapes.get(n.output[0]); a=attrs(n)
        if inp: entries.append(dict(op='transpose', input=inp, output=out, perm=list(a.get('perm',[]))))
    elif op=='ReduceMax':
        inp=shapes.get(n.input[0]); out=shapes.get(n.output[0]); a=attrs(n)
        if inp: entries.append(dict(op='reduce_max', input=inp, output=out, axes=[int(a.get('axes',[-1]))] if isinstance(a.get('axes',[-1]), int) else list(a.get('axes',[])), keepdims=int(a.get('keepdims',1))))
    elif op=='ArgMax':
        inp=shapes.get(n.input[0]); out=shapes.get(n.output[0]); a=attrs(n)
        if inp: entries.append(dict(op='argmax', input=inp, output=out, axis=int(a.get('axis',-1)), keepdims=int(a.get('keepdims',1))))
    elif op=='Softmax':
        inp=shapes.get(n.input[0]); out=shapes.get(n.output[0]); a=attrs(n)
        if inp: entries.append(dict(op='softmax', input=inp, output=out, axis=int(a.get('axis',1))))
    elif op=='MatMul':
        ins=[shapes.get(x) or list(inits.get(x,[])) for x in n.input]; out=shapes.get(n.output[0])
        if all(ins) and out: entries.append(dict(op='matmul', lhs=ins[0], rhs=ins[1], output=out))
    elif op=='TopK':
        ins=[shapes.get(x) or list(inits.get(x,[])) for x in n.input]; out=shapes.get(n.output[0]); a=attrs(n)
        if all(ins) and out: entries.append(dict(op='topk', input=ins[0], k=int(numpy_helper.to_array(next(x for x in m.graph.initializer if x.name==n.input[1])).item()) if len(ins[1])==1 else 1, output=out, axis=int(a.get('axis',-1))))
    elif op=='NonMaxSuppression':
        ins=[shapes.get(x) or list(inits.get(x,[])) for x in n.input]
        entries.append(dict(op='nms', inputs=ins))

# Deduplicate supported performance samples and preserve graph count.
counts=Counter()
for e in entries:
    key=json.dumps(e,sort_keys=True)
    counts[key]+=1

# Filter only kinds our perf runner currently supports.
supported_kinds={'conv','add','mul','sub','div','sigmoid','maxpool','reduce_max','argmax','softmax'}
samples=[]
for key,count in counts.items():
    e=json.loads(key)
    if e['op'] in supported_kinds:
        e['graph_count']=count
        samples.append(e)

summary=Counter(e['op'] for e in entries)
result=dict(model=args.model, graph_entries=entries, samples=samples,
            graph_operator_counts=dict(sorted(summary.items())),
            total_supported=sum(summary[x] for x in supported_kinds))
with open(args.output,'w') as f: json.dump(result,f,indent=2)

# Compatibility sidecar consumed by test_rvnpu_yolo_perf. Keep the format fixed:
# op,input,weight,stride,pad,rhs,count; dimensions use x separators.
def shape_text(value):
    if not value: return ''
    if isinstance(value[0], list): value=value[0]
    return 'x'.join(map(str,value))

csv_path=Path(args.output).with_name(Path(args.output).stem + '_samples.csv')
with csv_path.open('w') as f:
    f.write('op,input,weight,stride,pad,rhs,count\n')
    # Put affordable non-conv probes first so small --limit smoke runs do not
    # start with the largest input convolution.
    def sample_rank(e):
        dims=e.get('input') or e.get('lhs') or [1]
        if dims and isinstance(dims[0],list): dims=dims[0]
        n=1
        for d in dims: n*=max(1,int(d))
        return n,e['op']
    for e in sorted(samples,key=sample_rank):
        fields=[e['op'],shape_text(e.get('input') or e.get('lhs')),shape_text(e.get('weight') or e.get('kernel'))]
        if e['op'] in ('conv','maxpool'):
            fields += ['x'.join(map(str,e.get('stride',[]))),'x'.join(map(str,e.get('pad',[])))]
        else:
            fields += ['','']
        fields += [shape_text(e.get('rhs')),str(e['graph_count'])]
        f.write(','.join(fields)+'\n')

print('graph operator counts:',result['graph_operator_counts'])
print('total supported:',result['total_supported'])
print('unique supported samples:',len(samples))
print('wrote',args.output)
print('wrote',csv_path)
