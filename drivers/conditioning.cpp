// MIT; bounded adapters to existing pinned conditioning/NAF helpers.
#include "trellis_model.h"
#include "pixal3d_cond.h"
#include "pixal3d_input.h"
#include "image_preprocess.h"
#include "transforms_json.h"
#include "naf.h"
#include "proj_grid.h"
#include "geometry_checkpoint.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
namespace trellis { extern bool g_no_fa,g_require_gpu,g_gpu_auto; }
using namespace trellis;
using std::vector;using std::string;namespace fs=std::filesystem;
static void require(bool ok,const string& why){if(!ok)throw std::runtime_error(why);}
template<class V> static void save(const fs::path& p,const vector<V>& x){
    require(!fs::exists(p),"fresh output required");std::ofstream f(p,std::ios::binary);
    f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(V));require(bool(f),"output write failed");
}
static void finite(const vector<float>& a){for(float v:a)require(std::isfinite(v),"nonfinite result");}
static void fresh(const fs::path& p){require(!fs::exists(p),"fresh output directory required");fs::create_directories(p);}
static vector<std::array<int,3>> coords_file(const fs::path& p,int R){
    const size_t bytes=fs::file_size(p);require(bytes>0&&bytes%12==0,"coords require I32 Nx3");
    static_assert(sizeof(std::array<int,3>)==12,"coordinate ABI");
    vector<std::array<int,3>> a(bytes/12);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),bytes);require(bool(f),"coords read");
    for(const auto& c:a)for(int v:c)require(v>=0&&v<R,"coordinate outside grid");return a;
}
static Pixal3dInputViews original_views(const string& image,const fs::path& out,float fov){
    fs::create_directory(out/"views");string error;TransformsFile tf;
    require(preprocess_prematted_rgba_file(image,(out/"views/input.png").string(),error),error);
    require(synthesize_single_view_gauge("input.png",fov,tf,error),error);
    require(write_transforms_json((out/"views/transforms.json").string(),tf,error),error);
    Pixal3dInputViews input;require(pixal3d_load_input_views((out/"views").string(),input,error,1),error);return input;
}
static void geometry(const fs::path& out){
    fresh(out);vector<float> v={0,0,0,1,0,0,0,2,0,0,0,3};vector<int32_t> f={0,2,1,0,1,3,0,3,2,1,2,3};
    const auto oldv=v;const auto oldf=f;write_geometry_checkpoint((out/"geometry-only.ply").string().c_str(),v,f);
    require(v==oldv&&f==oldf,"serializer mutated live geometry");
    std::ifstream file(out/"geometry-only.ply",std::ios::binary);string line;bool end=false;
    while(std::getline(file,line)){if(line=="end_header"){end=true;break;}}require(end,"PLY header");
    vector<float> rv(v.size());file.read(reinterpret_cast<char*>(rv.data()),rv.size()*4);require(rv==v,"PLY positions");
    for(size_t i=0;i<f.size();i+=3){unsigned char n=0;int32_t tri[3]{};file.read(reinterpret_cast<char*>(&n),1);file.read(reinterpret_cast<char*>(tri),12);require(n==3&&std::equal(tri,tri+3,f.begin()+i),"PLY indices");}
    require(bool(file),"PLY truncated");bool rejected=false;try{write_geometry_checkpoint((out/"geometry-only.ply").string().c_str(),v,f);}catch(...){rejected=true;}
    require(rejected,"overwrite should be rejected");puts("CPU GEOMETRY CHECKPOINT PASS: 4 vertices, 4 faces, exact reopen, no mutation, no overwrite, zero backend graphs");
}
static ggml_context* meta(){return ggml_init({ggml_tensor_overhead()*8192+ggml_graph_overhead_custom(16384,false)+(1<<20),nullptr,true});}
static void support(ggml_backend_t b,ggml_cgraph* g){
    auto d=ggml_backend_get_device(b);require(d,"device missing");
    for(int i=0;i<ggml_graph_n_nodes(g);++i){auto n=ggml_graph_node(g,i);require(ggml_nbytes(n)<=2147483648ull,"individual logical/storage span too large");require(ggml_backend_dev_supports_op(d,n),"unsupported operation");}
}
static void graph_run(ggml_backend_t b,ggml_context* c,const vector<ggml_tensor*>& outputs,const std::function<void()>& upload){
    auto g=ggml_new_graph_custom(c,16384,false);for(auto t:outputs){ggml_set_output(t);ggml_build_forward_expand(g,t);}support(b,g);
    auto a=ggml_gallocr_new(ggml_backend_get_default_buffer_type(b));require(ggml_gallocr_alloc_graph(a,g),"graph allocation");upload();
    require(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS,"graph compute");ggml_backend_synchronize(b);
    // Callers read outputs before releasing this graph through the callback below.
    for(auto t:outputs)finite(tensor_to_f32(t));
    // The local fixture keeps its output snapshots in host vectors via a separate callback.
    // All reads happen here, then neither graph nor allocation survives into another call.
    for(auto t:outputs){const char* path=ggml_get_name(t);save(fs::path(path),tensor_to_f32(t));}
    ggml_gallocr_free(a);ggml_free(c);
}
static void kernels(const fs::path& out){
    fresh(out);ggml_backend_t backend=ggml_backend_vk_init(0);require(backend,"Vulkan0");
    const int T=20,h=10,TT=T*T,nb=h*h,d2=4,C=16;
    vector<float> pooled(TT*256),periods(16),cv,sv,values(C*nb);
    for(size_t i=0;i<pooled.size();++i)pooled[i]=float(int(i%97)-48)*.002f+float((i/97)%13)*.001f;
    for(int i=0;i<16;++i)periods[i]=.3f+.05f*i;
    for(size_t i=0;i<values.size();++i)values[i]=float(int(i%71)-35)*.01f;
    naf_rope_tables(T,periods,cv,sv);vector<int32_t> raster,inverse,windows;naf_block_order(T,h,h,raster,inverse);naf_window_index(T,h,h,windows);
    save(out/"pooled.f32",pooled);save(out/"cos.f32",cv);save(out/"sin.f32",sv);save(out/"values.f32",values);save(out/"raster.i32",raster);save(out/"window.i32",windows);
    auto qk=[&](int first,int rows,const string& label){
        int pixels=rows*2*T,blocks=rows*h,start=first*2*T;vector<float> cos(pixels*64),sin(cos.size());vector<int32_t> order(blocks*d2);
        for(int ch=0;ch<64;++ch){std::copy_n(cv.data()+ch*TT+start,pixels,cos.data()+ch*pixels);std::copy_n(sv.data()+ch*TT+start,pixels,sin.data()+ch*pixels);}
        for(int i=0;i<blocks*d2;++i)order[i]=raster[d2*first*h+i]-start;
        auto ctx=meta();auto p=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,T,T,256);auto rc=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,pixels,64);auto rs=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,pixels,64);auto bi=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,order.size());
        for(auto t:{p,rc,rs,bi})ggml_set_input(t);
        auto result=naf_build_qk_stripe(ctx,p,rc,rs,bi,T,h,h,first,rows);
        // GGML names have a fixed capacity; use short relative output names in this owned cwd.
        ggml_set_name(result.q_bm,(label+"-q.f32").c_str());ggml_set_name(result.k_rows,(label+"-k.f32").c_str());
        graph_run(backend,ctx,{result.q_bm,result.k_rows},[&](){ggml_backend_tensor_set(p,pooled.data(),0,pooled.size()*4);ggml_backend_tensor_set(rc,cos.data(),0,cos.size()*4);ggml_backend_tensor_set(rs,sin.data(),0,sin.size()*4);ggml_backend_tensor_set(bi,order.data(),0,order.size()*4);});
    };
    const auto before=fs::current_path();fs::current_path(out);qk(0,h,"qk-full");for(int first=0;first<h;first+=4)qk(first,std::min(4,h-first),"qk-"+std::to_string(first));
    vector<float> q(TT*256),k(nb*256);{std::ifstream f("qk-full-q.f32",std::ios::binary);f.read(reinterpret_cast<char*>(q.data()),q.size()*4);require(bool(f),"q input");}{std::ifstream f("qk-full-k.f32",std::ios::binary);f.read(reinterpret_cast<char*>(k.data()),k.size()*4);require(bool(f),"k input");}
    auto attn=[&](int first,int blocks,const string& label){
        auto ctx=meta();auto qt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,TT);auto kt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,nb);auto vt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,C,nb);auto wi=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,windows.size());for(auto t:{qt,kt,vt,wi})ggml_set_input(t);
        auto result=naf_build_attn_chunk(ctx,qt,kt,vt,wi,d2,nb,first,blocks);ggml_set_name(result,(label+".f32").c_str());
        graph_run(backend,ctx,{result},[&](){ggml_backend_tensor_set(qt,q.data(),0,q.size()*4);ggml_backend_tensor_set(kt,k.data(),0,k.size()*4);ggml_backend_tensor_set(vt,values.data(),0,values.size()*4);ggml_backend_tensor_set(wi,windows.data(),0,windows.size()*4);});
    };
    attn(0,nb,"attn-full");for(int first=0;first<nb;first+=17)attn(first,std::min(17,nb-first),"attn-"+std::to_string(first));
    fs::current_path(before);ggml_backend_free(backend);puts("SYNTHETIC kernels: 11 backend graph calls; T20/h10/C16, stripe4/4/2, chunks17/17/17/17/17/15");
}
static void condition(int argc,char** argv){
    require(argc==9,"cond DINO NAF IMAGE COORDS OUTPUT MODE FOV");
    const string mode=argv[7];require(mode=="small-dense"||mode=="small-chunked"||mode=="t1024","mode");
    const fs::path out=fs::absolute(argv[6]);fresh(out);auto input=original_views(argv[4],out,std::stof(argv[8]));
    const int S=mode=="t1024"?1024:160,R=mode=="t1024"?64:5;auto coords=coords_file(argv[5],R);save(out/"coords.i32",coords);
    vector<Pixal3dView> views;
    if(S==1024)views=input.views1024;
    else {
        // Explicit small synthetic images; same represented inputs in both invocations.
        // Preserve the real input's camera gauge, with a second deterministic view.
        for(int v=0;v<2;++v){Pixal3dView view=input.views512.at(0);view.rgb_premult.resize(3*S*S);for(size_t i=0;i<view.rgb_premult.size();++i)view.rgb_premult[i]=float((i*13+v*7)%257)/256.f;
            if(v==1){view.c2w[3]+=.15f;view.fov_x*=1.2f;}views.push_back(std::move(view));}
    }
    for(size_t i=0;i<views.size();++i){save(out/("view-"+std::to_string(i)+".f32"),views[i].rgb_premult);vector<float> camera(views[i].c2w,views[i].c2w+16);camera.push_back(views[i].fov_x);camera.push_back(input.mesh_scale);save(out/("camera-"+std::to_string(i)+".f32"),camera);}
    Model dino=Model::load(argv[2],0),naf=Model::load(argv[3],0);require(dino.on_gpu&&naf.on_gpu,"models must be on GPU");
    require(ggml_backend_get_device(dino.backend)==ggml_backend_get_device(naf.backend),"models must share physical device");
    Pixal3dSlatCondParams prm{S,R,S,input.mesh_scale};Pixal3dCond result;Pixal3dCondStats stats;
    if(mode=="small-dense") {result=pixal3d_cond_slat(dino,naf,views,prm);result.proj=pixal3d_gather_proj(result.proj,R,result.d_proj,coords);}
    else {prm.naf_block_chunk=S==160?31:0;result=pixal3d_cond_slat_gpu(dino,naf,views,prm,&stats,&coords);}
    require(result.n_global==5&&result.global.size()==5120&&result.d_proj==2048&&result.proj.size()==coords.size()*2048ull,"sparse output shape");finite(result.global);finite(result.proj);
    save(out/"global.f32",result.global);save(out/"projected.f32",result.proj);
    printf("COND_RESULT S=%d T=%d R=%d views=%zu N=%zu global=%zu projected=%zu weights=%zu persistent=%zu graph_peak=%zu reported_peak=%zu ms=%.3f\n",S,S,R,views.size(),coords.size(),result.global.size(),result.proj.size(),stats.weight_bytes,stats.cond_bytes,stats.view_alloc_bytes,stats.peak_bytes,stats.total_ms);
    dino.free();naf.free();
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try {
        require(argc>=2,"mode required");const string mode=argv[1];
        if(mode=="--help"){puts("geometry OUTPUT | kernels OUTPUT | cond DINO NAF IMAGE COORDS OUTPUT small-dense|small-chunked|t1024 FOV");return 0;}
        if(mode=="geometry"){require(argc==3,"geometry args");geometry(fs::absolute(argv[2]));return 0;}
        g_no_fa=true;g_require_gpu=true;g_gpu_auto=false;ggml_backend_load_all();
        if(mode=="kernels"){require(argc==3,"kernel args");kernels(fs::absolute(argv[2]));}
        else if(mode=="cond")condition(argc,argv);else throw std::runtime_error("unknown mode");return 0;
    }catch(const std::exception& e){fprintf(stderr,"conditioning driver rejected: %s\n",e.what());return 2;}
}
