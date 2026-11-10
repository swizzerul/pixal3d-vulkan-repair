// MIT. Narrow replay of a validated denormalized decoder boundary. No sampler or texture.
#include "decoder_checkpoint.h"
#include "trellis_model.h"
#include "shape_decoder.h"
#include "sparse.h"
#include "dual_grid.h"
#include "geometry_checkpoint.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <typeinfo>
namespace trellis {extern bool g_no_fa,g_require_gpu,g_gpu_auto;}
using namespace trellis;namespace dc=decoder_checkpoint;namespace fs=std::filesystem;
static void selftest(const fs::path& out){
    dc::require(!fs::exists(out),"selftest fresh path");fs::create_directories(out);
    std::vector<float> a={1.25f,-2.5f,0.f,3.75f};const auto original=a;
    dc::save(out/"tiny.bin",a,2,2,1);dc::require(dc::read<float>(out/"tiny.bin",2,1)==a&&a==original,"readback/no mutation");
    int rejected=0;try{dc::save(out/"tiny.bin",a,2,2,1);}catch(const std::exception&){++rejected;}
    try{dc::read<float>(out/"tiny.bin",3,1);}catch(const std::exception&){++rejected;}
    fs::copy_file(out/"tiny.bin",out/"truncated.bin");fs::resize_file(out/"truncated.bin",65);
    try{dc::read<float>(out/"truncated.bin",2,1);}catch(const std::exception&){++rejected;}
    fs::copy_file(out/"tiny.bin",out/"corrupt.bin");{std::fstream f(out/"corrupt.bin",std::ios::in|std::ios::out|std::ios::binary);f.seekp(64);f.put('\x01');}
    try{dc::read<float>(out/"corrupt.bin",2,1);}catch(const std::exception&){++rejected;}
    dc::require(rejected==4,"malformed/overwrite rejection");puts("DECODER CHECKPOINT SELFTEST PASS; 0 backend calls, exact readback and four rejection cases");
}
static std::vector<float> block(const Model& m,const std::vector<float>& input,const std::vector<std::array<int,3>>& coords,int index){
    constexpr size_t nodes=262144;const int N=(int)coords.size();dc::require(input.size()==size_t(N)*1024,"block input shape");auto nbr=build_neighbor_table(coords);
    auto c=ggml_init({ggml_tensor_overhead()*nodes+ggml_graph_overhead_custom(nodes,false)+(1<<20),nullptr,true});dc::require(c,"metadata allocation");
    auto f=ggml_new_tensor_2d(c,GGML_TYPE_F32,1024,N);auto nb=ggml_new_tensor_2d(c,GGML_TYPE_I32,N,27);ggml_set_input(f);ggml_set_input(nb);
    auto y=sparse_convnext(c,m,"blocks.0."+std::to_string(index),f,nb,N);ggml_set_output(y);auto g=ggml_new_graph_custom(c,nodes,false);ggml_build_forward_expand(g,y);
    auto dev=ggml_backend_get_device(m.backend);for(int i=0;i<ggml_graph_n_nodes(g);++i)dc::require(ggml_backend_dev_supports_op(dev,ggml_graph_node(g,i)),"unsupported block operation");
    auto al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));dc::require(ggml_gallocr_alloc_graph(al,g),"block allocation");
    ggml_backend_tensor_set(f,input.data(),0,input.size()*4);ggml_backend_tensor_set(nb,nbr.data(),0,nbr.size()*4);dc::require(ggml_backend_graph_compute(m.backend,g)==GGML_STATUS_SUCCESS,"block compute");
    auto out=tensor_to_f32(y);ggml_gallocr_free(al);ggml_free(c);return out;
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IONBF,0);
    try{
        dc::require(argc>=2,"mode required");std::string mode=argv[1];
        if(mode=="--help"){puts("selftest OUTPUT | exception-test | replay MODEL CAPTURE OUTPUT | stage0 MODEL CAPTURE OUTPUT");return 0;}
        if(mode=="exception-test")throw std::runtime_error("intentional model-free exception diagnostic");
        if(mode=="selftest"){dc::require(argc==3,"selftest arguments");selftest(argv[2]);return 0;}
        dc::require(argc==5&&(mode=="replay"||mode=="stage0"),"replay arguments");
        dc::require(!std::getenv("TRELLIS_DECODER_CAPTURE"),"replay must not overwrite prefix capture");
        fs::path cap=argv[3],out=argv[4];dc::require(!fs::exists(out),"fresh output");fs::create_directories(out);
        std::ifstream ctx(cap/"context.txt");std::string context((std::istreambuf_iterator<char>(ctx)),{});
        dc::require(context.find("normalization=denormalized_shape_decoder_input\n")!=std::string::npos&&context.find("resolution=1024\n")!=std::string::npos,"normalization/resolution contract");
        auto coords=dc::coordinates(cap/"coords.bin",64);auto latent=dc::read<float>(cap/"latent.bin",32,1);dc::require(latent.size()==coords.size()*32&&!dc::nonfinite(latent),"latent finite/shape");
        g_no_fa=true;g_require_gpu=true;g_gpu_auto=false;ggml_backend_load_all();Model m=Model::load(argv[2],0);
        dc::require(m.on_gpu&&std::string(ggml_backend_name(m.backend)).rfind("Vulkan",0)==0,"Vulkan0 required");
        if(mode=="stage0"){
            auto h=dc::read<float>(cap/"from-latent.bin",1024,1);dc::require(h.size()==coords.size()*1024&&!dc::nonfinite(h),"stage0 finite input");
            for(int j=0;j<4;++j){h=block(m,h,coords,j);dc::save(out/("block-"+std::to_string(j)+".bin"),h,coords.size(),1024,1);dc::summary(("block-"+std::to_string(j)).c_str(),h);if(dc::nonfinite(h)){m.free();return 3;}}
            m.free();return 0;
        }
        auto so=shape_decode(m,latent,coords,1024);m.free();dc::require(so.feats7.size()==so.coords.size()*7&&!dc::nonfinite(so.feats7),"ShapeOut finite/shape");
        dc::save(out/"shape-features.bin",so.feats7,so.coords.size(),7,1);dc::save(out/"shape-coords.bin",dc::flatten(so.coords),so.coords.size(),3,2);
        dc::require(so.subs.size()==4,"four masks");for(size_t i=0;i<so.subs.size();++i)dc::save(out/("mask-"+std::to_string(i)+".bin"),so.subs[i],so.subs[i].size()/8,8,3);
        auto mesh=dual_grid_to_mesh(so);dc::require(mesh.V()>0&&mesh.F()>0,"nonempty actual mesh");write_geometry_checkpoint((out/"GEOMETRY-ONLY-NOT-FINAL.ply").string().c_str(),mesh.verts,mesh.faces);
        printf("ACTUAL_DECODER_GEOMETRY vertices=%d faces=%d voxels=%zu\n",mesh.V(),mesh.F(),so.coords.size());return 0;
    }catch(const std::exception& e){fprintf(stderr,"[decoder-error] type=%s what=%s\n",typeid(e).name(),e.what());return 2;}
    catch(...){fputs("[decoder-error] non-std exception; pipeline stopped\n",stderr);return 4;}
}
