// MIT. Opt-in, boundary-only decoder observations; never changes tensor values.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace decoder_checkpoint {
namespace fs=std::filesystem;
inline fs::path root;
inline bool active=false;
inline void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error("decoder checkpoint: "+s);}
inline uint64_t digest(const void* p,size_t n){
    uint64_t h=14695981039346656037ull;auto b=static_cast<const unsigned char*>(p);
    for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}return h;
}
// Eight LE u64 fields: magic, version, dtype (1=f32,2=i32,3=u8), rows,
// columns, element width, payload bytes, FNV1a64. SHA256 is recorded locally
// by the supervisor after readback; the checksum here detects truncation/corruption.
template<class V> inline void save(const fs::path& p,const std::vector<V>& a,uint64_t rows,uint64_t cols,uint64_t type){
    static_assert(sizeof(float)==4&&sizeof(int)==4,"32-bit numeric ABI required");
    require(rows&&cols&&rows<=UINT64_MAX/cols&&rows*cols==a.size(),"array dimensions");
    require(!fs::exists(p),"refuse overwrite");
    uint64_t header[8]={0x31504b4344455254ull,1,type,rows,cols,sizeof(V),a.size()*sizeof(V),digest(a.data(),a.size()*sizeof(V))};
    std::ofstream f(p,std::ios::binary);require(bool(f),"create array");
    f.write(reinterpret_cast<const char*>(header),sizeof(header));f.write(reinterpret_cast<const char*>(a.data()),header[6]);f.flush();require(bool(f),"write array");
}
template<class V> inline std::vector<V> read(const fs::path& p,uint64_t cols,uint64_t type){
    uint64_t h[8]{};std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(h),sizeof(h));
    require(bool(f)&&h[0]==0x31504b4344455254ull&&h[1]==1&&h[2]==type&&h[4]==cols&&h[5]==sizeof(V),"array header");
    require(h[3]>0&&cols>0&&h[3]<=UINT64_MAX/cols/sizeof(V)&&h[6]==h[3]*cols*sizeof(V)&&h[6]<=2147483648ull,"array byte count");
    require(fs::file_size(p)==sizeof(h)+h[6],"array file size");std::vector<V> a(h[3]*cols);
    f.read(reinterpret_cast<char*>(a.data()),h[6]);require(bool(f)&&digest(a.data(),h[6])==h[7],"array checksum");return a;
}
inline std::vector<int32_t> flatten(const std::vector<std::array<int,3>>& c){
    std::vector<int32_t> a; a.reserve(c.size()*3);for(const auto& v:c)for(int x:v)a.push_back(x);return a;
}
inline std::vector<std::array<int,3>> coordinates(const fs::path& p,int grid){
    auto a=read<int32_t>(p,3,2);std::vector<std::array<int,3>> c(a.size()/3);
    for(size_t i=0;i<c.size();++i)for(int j=0;j<3;++j){int x=a[i*3+j];require(x>=0&&x<grid,"coordinate range");c[i][j]=x;}return c;
}
inline size_t nonfinite(const std::vector<float>& a){size_t n=0;for(float x:a)n+=!std::isfinite(x);return n;}
inline void summary(const char* name,const std::vector<float>& a){
    size_t pos=0,bad=0;float lo=std::numeric_limits<float>::infinity(),hi=-lo;
    for(float x:a){if(!std::isfinite(x)){++bad;continue;}pos+=x>0;lo=std::min(lo,x);hi=std::max(hi,x);}
    fprintf(stderr,"[decoder-boundary] %s elements=%zu nonfinite=%zu positive=%zu finite_min=%.9g finite_max=%.9g\n",name,a.size(),bad,pos,lo,hi);fflush(stderr);
}
inline void begin(const std::vector<float>& latent,const std::vector<std::array<int,3>>& coords,int res,const std::string& arch){
    const char* dir=std::getenv("TRELLIS_DECODER_CAPTURE");if(!dir||!*dir)return;
    require(!active&&res==1024,"one final HR shape decode only");root=fs::absolute(dir);require(!fs::exists(root),"fresh capture directory");fs::create_directories(root);active=true;
    save(root/"latent.bin",latent,coords.size(),32,1);save(root/"coords.bin",flatten(coords),coords.size(),3,2);
    std::ofstream f(root/"context.txt");f<<"format=1\nresolution="<<res<<"\nnormalization=denormalized_shape_decoder_input\nlayout=row-major-NC_channels-fast\ncoordinate_order=unchanged\narchitecture="<<arch<<"\n";
    for(auto key:{"GGML_VK_DISABLE_F16","GGML_VK_DISABLE_COOPMAT"})f<<key<<'='<<(std::getenv(key)?std::getenv(key):"absent")<<'\n';
    f.flush();require(bool(f),"context write");summary("decoder_entry",latent);
}
inline void boundary(const char* name,const std::vector<float>& a,int C,size_t N){
    if(!active)return;save(root/(std::string(name)+".bin"),a,N,C,1);summary(name,a);
}
inline void subdivision(const std::string& prefix,const std::vector<float>& f,int C,const std::vector<std::array<int,3>>& c,const std::vector<float>& logits,int attempt){
    if(!active||prefix!="blocks.0.4")return;
    if(attempt==0){boundary("subdivision-input",f,C,c.size());save(root/"subdivision-coords.bin",flatten(c),c.size(),3,2);}
    boundary(("subdivision-logits-"+std::to_string(attempt)).c_str(),logits,8,c.size());
}
inline void children(const std::string& prefix,const std::vector<std::array<int,3>>& c,const std::vector<uint8_t>& mask,const std::vector<int32_t>& idx,const std::vector<int32_t>& starts){
    if(!active||prefix!="blocks.0.4")return;
    save(root/"mask.bin",mask,mask.size()/8,8,3);save(root/"parent-start.bin",starts,starts.size(),1,2);
    if(!c.empty()){save(root/"children.bin",flatten(c),c.size(),3,2);save(root/"gather.bin",idx,idx.size(),1,2);}
    fprintf(stderr,"[decoder-boundary] actual_child_count=%zu\n",c.size());
    // The diagnostic prefix cannot enter the remaining decoder or texture pipeline.
    throw std::runtime_error("diagnostic prefix stopped after first subdivision boundary; no repair claim");
}
}
