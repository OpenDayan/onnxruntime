#include "runtime/device_context.h"
#include "runtime/tensor_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>

namespace {

std::vector<float> Values(size_t count) {
  std::vector<float> v(count);
  for (size_t i=0;i<count;++i)
    v[i]=static_cast<float>((static_cast<int>(i*17+3)%37)-18)*0.03125f;
  return v;
}

std::vector<int64_t> Shape(const std::vector<float>& s) {
  std::vector<int64_t> r(s.size());
  std::transform(s.begin(),s.end(),r.begin(),[](float x){return static_cast<int64_t>(x);});
  return r;
}

size_t Elements(const std::vector<int64_t>& s) {
  size_t n=1; for(auto d:s) n*=static_cast<size_t>(d); return n;
}

void RunSample(rvnpu::DeviceContext& ctx,const std::string& op,
               const std::vector<int64_t>& input,
               const std::vector<int64_t>& weight,
               const std::vector<int64_t>& stride,
               const std::vector<int64_t>& pad,
               const std::vector<int64_t>& rhs={}) {
  if(op=="sigmoid") {
    auto x=Values(Elements(input)); auto y=x;
    ctx.RunSigmoid(x.data(),y.data(),rvnpu::ValidateUnaryTensorShape(input,"perf"));
  } else if(op=="silu" || op=="mul" || op=="add" || op=="sub" || op=="div") {
    auto a=Values(Elements(input)); auto b=Values(rhs.empty()?1:Elements(rhs)); auto y=a;
    if(op=="add") ctx.RunAdd(a.data(),b.data(),y.data(),rvnpu::ValidateBinaryElementwiseShapes(input,rhs,"perf"));
    else if(op=="mul") ctx.RunMul(a.data(),b.data(),y.data(),rvnpu::ValidateBinaryElementwiseShapes(input,rhs,"perf"));
    else if(op=="sub") ctx.RunSub(a.data(),b.data(),y.data(),rvnpu::ValidateBinaryElementwiseShapes(input,rhs,"perf"));
    else ctx.RunDiv(a.data(),b.data(),y.data(),rvnpu::ValidateBinaryElementwiseShapes(input,rhs,"perf"));
  } else if(op=="maxpool") {
    rvnpu::MaxPoolAttributes a; a.kernel_shape={weight[0],weight[1]}; a.strides={stride[0],stride[1]}; a.pads={pad[0],pad[1],pad[0],pad[1]};
    auto shape=rvnpu::ValidateMaxPoolShape(input,a); auto x=Values(shape.input_elements); auto y=x;
    ctx.RunMaxPool(x.data(),y.data(),shape);
  } else if(op=="reduce_max") {
    auto shape=rvnpu::ValidateReduceMaxShape(input,{-1},0); auto x=Values(shape.input_elements); auto y=std::vector<float>(shape.output_elements);
    ctx.RunReduceMax(x.data(),y.data(),shape);
  } else if(op=="argmax") {
    rvnpu::ArgMaxAttributes a; a.axis=-1; a.keepdims=0;
    auto shape=rvnpu::ValidateArgMaxShape(input,a); auto x=Values(shape.input_elements); auto y=std::vector<int64_t>(shape.output_elements);
    ctx.RunArgMax(x.data(),y.data(),shape);
  } else if(op=="softmax") {
    auto shape=rvnpu::ValidateSoftmaxAxis1Shape(input,1); auto x=Values(shape.elements); auto y=x;
    ctx.RunSoftmax(x.data(),y.data(),shape);
  } else if(op=="conv") {
    rvnpu::ConvAttributes a; a.kernel_shape={weight[2],weight[3]}; a.strides={stride[0],stride[1]}; a.pads={pad[0],pad[1],pad[0],pad[1]};
    auto shape=rvnpu::ValidateConvShapes(input,weight,nullptr,a);
    auto x=Values(shape.input_elements); auto w=Values(shape.weight_elements); auto y=std::vector<float>(shape.output_elements);
    ctx.RunConv(x.data(),w.data(),nullptr,y.data(),shape);
  } else if(op=="resize") {
    auto output=input;
    output[2]*=2; output[3]*=2;
    auto shape=rvnpu::ValidateResizeShape(input,output);
    auto x=Values(shape.input_elements); auto y=std::vector<float>(shape.output_elements);
    ctx.RunResize(x.data(),y.data(),shape);
  } else if(op=="transpose") {
    // Some transposes in the official postprocess are INT64 (for example
    // Transpose over box indices). 64-element tensors encode the dtype by
    // placing 32/64 in the CSV weight column, matching DeviceContext dispatch.
    if(weight.size()==1 && weight[0]==64) {
      auto shape=rvnpu::ValidateTransposeShape(input,stride);
      std::vector<int64_t> x(shape.elements),y(shape.elements);
      ctx.RunTranspose(x.data(),y.data(),shape);
    } else {
      auto shape=rvnpu::ValidateTransposeShape(input,stride);
      auto x=Values(shape.elements); auto y=std::vector<float>(shape.elements);
      ctx.RunTranspose(x.data(),y.data(),shape);
    }
  } else if(op=="nms") {
    // CSV encodes: input=1xN, weight=C, stride=max_output,
    // pad=iou_threshold_percent, rhs=score_threshold_percent.
    const int64_t boxes_count=input[0];
    const int64_t classes=weight[0];
    const int64_t max_output=std::min<int64_t>(stride[0], boxes_count);
    const float iou_threshold=static_cast<float>(pad[0])/100.0f;
    const float score_threshold=static_cast<float>(rhs[0])/100.0f;
    rvnpu::NmsAttributes attrs;
    auto shape=rvnpu::ValidateNmsShapes(
        {1,boxes_count,4},{1,classes,boxes_count},max_output,attrs);
    std::vector<float> boxes(shape.boxes_elements),scores(shape.scores_elements);
    for(size_t i=0;i<boxes.size();i+=4) {
      boxes[i+0]=static_cast<float>(i/4);
      boxes[i+1]=static_cast<float>(i/4);
      boxes[i+2]=static_cast<float>(i/4+1);
      boxes[i+3]=static_cast<float>(i/4+1);
    }
    for(size_t i=0;i<scores.size();++i)
      scores[i]=0.1f+static_cast<float>(i%17)*0.05f;
    (void)ctx.RunNms(boxes.data(),scores.data(),iou_threshold,score_threshold,shape);
  } else {
    throw std::runtime_error("unsupported sample op "+op);
  }
}

} // namespace

int main(int argc,char**argv) {
  if(argc<3) { std::cerr<<"usage: test_rvnpu_yolo_perf KERNEL_ELF MANIFEST [OUTPUT_JSON] [OUTPUT_CSV] [--limit N]\n"; return 2; }
  const std::string elf=argv[1], manifest=argv[2], outjson=argc>3?argv[3]:"yolo_perf_results.json", outcsv=argc>4?argv[4]:"yolo_perf_results.csv";
  size_t limit=SIZE_MAX; double spatial_scale=1.0; size_t proxy_elements=0;
  for(int i=5;i<argc;i+=2) {
    if(std::string(argv[i])=="--limit") limit=std::stoull(argv[i+1]);
    else if(std::string(argv[i])=="--scale") spatial_scale=std::stod(argv[i+1]);
    else if(std::string(argv[i])=="--proxy-elements") proxy_elements=std::stoull(argv[i+1]);
  }
  // Minimal manifest parser: only parses the compact sidecar emitted by wrapper.
  std::ifstream in(manifest); if(!in) throw std::runtime_error("cannot open manifest");
  std::vector<std::vector<std::string>> rows; std::string line;
  std::getline(in,line); // header
  while(std::getline(in,line)) { if(line.empty()) continue; std::stringstream ss(line); std::vector<std::string> f; std::string x; while(std::getline(ss,x,',')) f.push_back(x); rows.push_back(f); }

  rvnpu::DeviceContext ctx(elf);
  nlohmann_json_marker:
  std::ofstream json(outjson),csv(outcsv); csv<<"index,op,input,weight,stride,pad,rhs,graph_count,cycles,wall_ms,total_cycles\n";
  uint64_t grand=0; double wall=0; size_t idx=0;
  auto records=ctx.profile_records();
  for(const auto& row:rows) {
    if(idx>=limit) break;
    // row: op,input,weight,stride,pad,rhs,count
    auto parse=[](const std::string& s){ std::vector<int64_t> v; size_t p=0; while((p=s.find('x',p))!=std::string::npos) {} return v; };
    (void)parse;
    // Format is fixed decimal with x separators.
    auto dims=[](const std::string& s){ std::vector<int64_t> v; size_t b=0; while(true){size_t e=s.find('x',b); v.push_back(std::stoll(s.substr(b,e-b))); if(e==std::string::npos)break;b=e+1;} return v; };
    std::vector<int64_t> input=dims(row[1]),weight={1,1},stride={1,1},pad={0,0},rhs={1};
    // Scale trailing H/W dimensions for affordable sampling. H/W are indices 2/3
    // for rank 4 and index 1 for rank 3.
    auto scale_dims=[&](std::vector<int64_t>& v) {
      if(v.size()==4) { v[2]=std::max<int64_t>(1,llround(v[2]*spatial_scale)); v[3]=std::max<int64_t>(1,llround(v[3]*spatial_scale)); }
      else if(v.size()==3) v[1]=std::max<int64_t>(1,llround(v[1]*spatial_scale));
    };
    if(spatial_scale!=1.0) {
      scale_dims(input); scale_dims(rhs);
      if(row[0]=="conv") {
        // Weight channel/hardware topology remains unchanged. Pad and stride scale together.
        for(auto& x:stride) x=std::max<int64_t>(1,llround(x*spatial_scale));
        for(auto& x:pad) x=std::max<int64_t>(0,llround(x*spatial_scale));
      }
    }
    // Elementwise/SFU paths currently hang on large RTL shapes. Measure a small
    // proxy and linearly extrapolate cycles by element count.
    bool proxy=false; size_t proxy_count=0;
    if(proxy_elements>0 && row[0]!="conv" && row[0]!="maxpool") {
      proxy_count=Elements(input);
      if(proxy_count>proxy_elements) {
        const auto scale_vector=[&](std::vector<int64_t>&v,double ratio) {
          if(v.size()==4) {
            v[2]=std::max<int64_t>(1,llround(v[2]*ratio));
            v[3]=std::max<int64_t>(1,llround(v[3]*ratio));
          } else if(v.size()==3) {
            v[1]=std::max<int64_t>(1,llround(v[1]*ratio));
          } else if(v.size()<=2) {
            v[0]=std::max<int64_t>(1,llround(v[0]*ratio));
          }
        };
        const double ratio=static_cast<double>(proxy_elements)/
                           static_cast<double>(proxy_count);
        scale_vector(input,ratio);
        scale_vector(rhs,ratio);
        proxy=true;
      }
    }
    if(row.size()>2 && !row[2].empty()) weight=dims(row[2]);
    if(row.size()>3 && !row[3].empty()) stride=dims(row[3]);
    if(row.size()>4 && !row[4].empty()) pad=dims(row[4]);
    if(row.size()>5 && !row[5].empty()) rhs=dims(row[5]);
    size_t count=std::stoull(row[6]);
    const size_t full_elements=Elements(input);
    ctx.ClearProfile(); records=ctx.profile_records();
    try { RunSample(ctx,row[0],input,weight,stride,pad,rhs); } catch(const std::exception& e) { std::cerr<<"FAIL sample "<<idx<<" "<<row[0]<<" "<<e.what()<<"\n"; }
    if(!ctx.profile_records().empty()) {
      uint64_t cycles=0; double ms=0; for(auto&r:ctx.profile_records()){cycles+=r.cycles;ms+=r.wall_ms;}
      double normalization=1.0;
      if(proxy) {
        normalization=static_cast<double>(proxy_count)/std::max<size_t>(1,Elements(input));
      }
      if(spatial_scale!=1.0 && (row[0]=="conv" || row[0]=="maxpool" || row[0]=="resize" ||
                                row[0]=="transpose" || row[0]=="softmax" || row[0]=="reduce_max" ||
                                row[0]=="argmax")) {
        normalization=1.0/(spatial_scale*spatial_scale);
      } else if(spatial_scale!=1.0) {
        // Elementwise cycles scale with element count.
        normalization=1.0/(spatial_scale*spatial_scale);
      }
      const uint64_t normalized_cycles=static_cast<uint64_t>(cycles*normalization);
      const double normalized_ms=ms*normalization;
      uint64_t total=normalized_cycles*count; grand+=total; wall+=normalized_ms*count;
      csv<<idx<<','<<row[0]<<','<<row[1]<<','<<(row.size()>2?row[2]:"")<<','<<(row.size()>3?row[3]:"")<<','<<(row.size()>4?row[4]:"")<<','<<(row.size()>5?row[5]:"")<<','<<count<<','<<normalized_cycles<<','<<normalized_ms<<','<<total<<'\n';
      std::cout<<"SAMPLE "<<idx<<" op="<<row[0]<<" input="<<row[1]<<" weight="<<(row.size()>2?row[2]:"")<<" count="<<count<<" measured_cycles="<<cycles<<" estimated_full_cycles="<<normalized_cycles<<" measured_wall_ms="<<ms<<" estimated_wall_ms="<<normalized_ms<<" total_cycles="<<total<<"\n";
    }
    ++idx;
  }
  json<<"{\"total_cycles\":"<<grand<<",\"summed_wall_ms\":"<<wall<<",\"samples\":"<<idx<<",\"spatial_scale\":"<<spatial_scale<<",\"proxy_elements\":"<<proxy_elements<<"}\n";
  std::cout<<"TOTAL estimated_full_cycles="<<grand<<" estimated_summed_wall_ms="<<wall<<" samples="<<idx<<" spatial_scale="<<spatial_scale<<"\n";
  return 0;
}
