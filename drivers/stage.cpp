// MIT; narrow adapters to the pinned implementation. No private inputs or network.
#include "trellis_model.h"
#include "dinov3.h"
#include "naf.h"
#include "pixal3d_input.h"
#include "image_preprocess.h"
#include "transforms_json.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace trellis { extern bool g_no_fa, g_require_gpu, g_gpu_auto; }
using namespace trellis;
static void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
static void write_f32(const std::filesystem::path& path,const std::vector<float>& v) {
    require(!std::filesystem::exists(path),"Output already exists");
    for(float x:v) require(std::isfinite(x),"Nonfinite stage output");
    std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));
    require(bool(f),"Output write failed");
}
static std::vector<float> read_f32(const std::filesystem::path& path,size_t n) {
    require(std::filesystem::file_size(path)==n*sizeof(float),"DINO size mismatch");
    std::vector<float> v(n); std::ifstream f(path,std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()),n*sizeof(float)); require(bool(f),"Read failed");
    for(float x:v)require(std::isfinite(x),"Nonfinite captured DINO input"); return v;
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if(argc==2 && std::string(argv[1])=="--help") {
        puts("dino IMAGE MODEL OUTPUT_DIR GPU FOV_RADIANS\nnaf STAGED_VIEWS MODEL DINO_F32 OUTPUT_DIR GPU TARGET_SIZE"); return 0;
    }
    try {
        require(argc>=2,"Missing stage"); const std::string mode=argv[1];
        require((mode=="dino"&&argc==7)||(mode=="naf"&&argc==8),"Invalid stage arguments");
        g_no_fa=true; g_require_gpu=true; g_gpu_auto=false;
        ggml_backend_load_all();
        const int gpu=std::stoi(argv[mode=="dino"?5:6]); require(gpu>=0,"GPU required");
        const std::filesystem::path out=argv[mode=="dino"?4:5];
        require(!std::filesystem::exists(out),"Fresh output directory required");
        std::filesystem::create_directories(out);
        std::string err,views=argv[2];
        if(mode=="dino") {
            views=(out/"views").string(); std::filesystem::create_directory(views);
            require(preprocess_prematted_rgba_file(argv[2],views+"/input.png",err),err);
            TransformsFile tf;
            require(synthesize_single_view_gauge("input.png",std::stof(argv[6]),tf,err),err);
            require(write_transforms_json(views+"/transforms.json",tf,err),err);
        }
        Pixal3dInputViews input;
        require(pixal3d_load_input_views(views,input,err,1),err);
        require(input.views1024.size()==1,"Exactly one input view required");
        const auto& rgb=input.views1024[0].rgb_premult;
        require(rgb.size()==3ull*1024*1024,"Image layout mismatch");
        write_f32(out/"image-rgb-chw.f32",rgb);
        Model model=Model::load(argv[3],gpu);
        require(model.on_gpu && std::string(ggml_backend_name(model.backend)).rfind("Vulkan",0)==0,"Vulkan GPU required");
        if(mode=="dino") {
            auto normalized=pixal3d_imagenet_normalize(rgb,1024);
            write_f32(out/"image-normalized-chw.f32",normalized);
            auto tokens=dinov3_encode(model,normalized,1024);
            require(tokens.size()==4101ull*1024,"DINO output layout mismatch");
            write_f32(out/"dino-token-major.f32",tokens);
            puts("DINO output: F32 ggml [1024,4101,1,1], contiguous token-major; 16797696 bytes");
        } else {
            const int T=std::stoi(argv[7]); require(T==512||T==1024,"Unsupported target");
            auto tokens=read_f32(argv[4],4101ull*1024);
            std::vector<float> chw(1024ull*64*64);
            for(int p=0;p<4096;p++)for(int c=0;c<1024;c++)chw[size_t(c)*4096+p]=tokens[size_t(p+5)*1024+c];
            write_f32(out/"dino-patches-chw.f32",chw);
            NafGgmlStats stats;
            auto result=naf_upsample_ggml(model,rgb.data(),1024,chw.data(),1024,64,64,T,nullptr,&stats);
            require(result.size()==1024ull*T*T,"NAF output layout mismatch");
            write_f32(out/"naf-chw.f32",result);
            printf("NAF output: F32 CHW [1024,%d,%d], nodes=%d allocation=%zu compute_ms=%.3f total_ms=%.3f\n",T,T,stats.n_nodes,stats.alloc_bytes,stats.compute_ms,stats.total_ms);
        }
        model.free(); return 0;
    } catch(const std::exception& e) { fprintf(stderr,"stage rejected: %s\n",e.what()); return 2; }
}
