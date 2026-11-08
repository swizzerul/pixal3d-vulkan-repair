// MIT; metadata and tiny CPU checks only. Route mode never allocates a graph.
#include "trellis_model.h"
#include "naf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
using namespace trellis;
static void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
static void abi() {
    printf("ABI sizeof_tensor=%zu type=%zu buffer=%zu ne=%zu nb=%zu op=%zu op_params=%zu flags=%zu src=%zu view_src=%zu view_offs=%zu data=%zu name=%zu\n",sizeof(ggml_tensor),offsetof(ggml_tensor,type),offsetof(ggml_tensor,buffer),offsetof(ggml_tensor,ne),offsetof(ggml_tensor,nb),offsetof(ggml_tensor,op),offsetof(ggml_tensor,op_params),offsetof(ggml_tensor,flags),offsetof(ggml_tensor,src),offsetof(ggml_tensor,view_src),offsetof(ggml_tensor,view_offs),offsetof(ggml_tensor,data),offsetof(ggml_tensor,name));
    for(int op=0;op<GGML_OP_COUNT;op++)printf("OP %d %s\n",op,ggml_op_name(ggml_op(op)));
}
static void cpu() {
    auto b=ggml_backend_cpu_init(); require(b,"CPU backend");
    auto c=ggml_init({2*1024*1024,nullptr,true}); require(c,"context");
    auto a=ggml_new_tensor_1d(c,GGML_TYPE_F32,8),d=ggml_new_tensor_1d(c,GGML_TYPE_F32,8);
    auto out=ggml_add(c,a,d); auto g=ggml_new_graph(c); ggml_build_forward_expand(g,out);
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(b));require(ggml_gallocr_alloc_graph(alloc,g),"CPU allocation");
    float x[8],y[8],z[8];for(int i=0;i<8;i++){x[i]=float(i);y[i]=float(10+i);}
    ggml_backend_tensor_set(a,x,0,sizeof(x));ggml_backend_tensor_set(d,y,0,sizeof(y));
    require(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS,"CPU compute");ggml_backend_tensor_get(out,z,0,sizeof(z));
    for(int i=0;i<8;i++)require(z[i]==float(10+2*i),"CPU mismatch");
    ggml_gallocr_free(alloc);ggml_free(c);ggml_backend_free(b);puts("CPU ADD PASS");
}
static void route(int S,int T) {
    Model m; m.backend=ggml_backend_vk_init(0);require(m.backend,"Vulkan0 initialization");
    auto dev=ggml_backend_get_device(m.backend);require(dev,"device");
    auto c=ggml_init({32*1024*1024,nullptr,true});require(c,"metadata context");
    auto weight=[&](std::string name,int K,int ci,int co){auto w=ggml_new_tensor_4d(c,GGML_TYPE_F16,K,K,ci,co);ggml_set_name(w,(name+".weight").c_str());m.tensors[name+".weight"]=w;m.tensors[name+".bias"]=ggml_new_tensor_1d(c,GGML_TYPE_F32,co);};
    for(auto branch:{std::string("image_encoder.encoder"),std::string("image_encoder.sem_encoder")}){
        int K=branch=="image_encoder.encoder"?1:3;weight(branch+".0",K,3,128);
        for(int block=1;block<=2;block++)for(int n=1;n<=2;n++){
            std::string p=branch+"."+std::to_string(block);weight(p+".conv"+std::to_string(n),K,128,128);
            m.tensors[p+".norm"+std::to_string(n)+".weight"]=ggml_new_tensor_1d(c,GGML_TYPE_F32,128);
            m.tensors[p+".norm"+std::to_string(n)+".bias"]=ggml_new_tensor_1d(c,GGML_TYPE_F32,128);
        }
    }
    auto opts=naf_ggml_opts_for(m);NafGraphInputs inputs;
    auto lr=ggml_new_tensor_2d(c,GGML_TYPE_F32,1024,(S/16)*(S/16));
    auto out=naf_build(c,m,S,T,S/16,S/16,lr,inputs,opts);
    auto graph=ggml_new_graph_custom(c,16384,false);ggml_build_forward_expand(graph,out);
    int direct=0,col=0,unsupported=0;size_t largest=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);i++){
        auto n=ggml_graph_node(graph,i);bool supported=ggml_backend_dev_supports_op(dev,n);
        require(!n->data&&!n->buffer,"Metadata unexpectedly allocated");
        largest=std::max(largest,ggml_nbytes(n));direct+=n->op==GGML_OP_CONV_2D;col+=n->op==GGML_OP_IM2COL;unsupported+=!supported;
        printf("NODE %d %s %s type=%d ne=%lld,%lld,%lld,%lld bytes=%zu supported=%d\n",i,ggml_op_name(n->op),n->name,int(n->type),(long long)n->ne[0],(long long)n->ne[1],(long long)n->ne[2],(long long)n->ne[3],ggml_nbytes(n),int(supported));
    }
    printf("ROUTE S=%d T=%d generic=%d explicit_direct=%d nodes=%d direct=%d im2col=%d unsupported=%d largest_tensor=%zu allocations=0 computes=0\n",S,T,int(opts.generic_lowering),int(opts.direct_conv),ggml_graph_n_nodes(graph),direct,col,unsupported,largest);
    ggml_free(c);ggml_backend_free(m.backend);
}
int main(int argc,char** argv){try{abi();if(argc==2&&std::string(argv[1])=="--cpu")cpu();else if(argc==2&&std::string(argv[1])=="--route"){route(512,512);route(1024,512);route(1024,1024);}else if(argc!=2||std::string(argv[1])!="--help")throw std::runtime_error("Use --help, --cpu or --route");return 0;}catch(const std::exception& e){fprintf(stderr,"probe rejected: %s\n",e.what());return 2;}}
