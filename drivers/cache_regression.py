"""One synthetic graph: noncontiguous RHS conversion, scratch writer, same-RHS reuse.

Requires NumPy, an exact bundle manifest, and an external supervisor's PID permit.
No model loading, network access, retries, or data files distributed with this test.
Run one baseline and one candidate per writer; inspect Vulkan debug dispatch logs.
"""
import ctypes as C,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
P=C.c_void_p; S=C.c_size_t; I=C.c_int; L=C.c_int64
class Init(C.Structure):_fields_=[('size',S),('buffer',P),('no_alloc',C.c_bool)]
class Tensor(C.Structure):
    _fields_=[('type',I),('buffer',P),('ne',L*4),('nb',S*4),('op',I),('params',I*16),('flags',I),('src',P*10),('view_src',P),('view_offs',S),('data',P)]
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def main():
    bundle,writer,output=sys.argv[1:];bundle=Path(bundle).resolve();out=Path(output)
    assert writer in ('softmax','flash') and out.is_dir()
    save(out/'worker-start.json',{'pid':os.getpid(),'writer':writer})
    assert os.environ.get('GGML_VK_DISABLE_F16')=='1'
    manifest=json.loads((bundle/'bundle-manifest.json').read_text())
    for n in ['ggml-base.dll','ggml.dll','ggml-cpu.dll','ggml-vulkan.dll']:
        assert sha((bundle/n).read_bytes())==manifest['files'][n]['sha256']
    directory=os.add_dll_directory(str(bundle));base=C.CDLL(str(bundle/'ggml-base.dll'))
    deadline=time.monotonic()+15
    while not (out/'permit.json').exists() and time.monotonic()<deadline:time.sleep(.02)
    assert json.loads((out/'permit.json').read_text())['pid']==os.getpid(),'External guard permit required'
    vk=C.CDLL(str(bundle/'ggml-vulkan.dll'))
    def api(n,r,a,lib=base):f=getattr(lib,n);f.restype=r;f.argtypes=a;return f
    init=api('ggml_init',P,[Init]);new=api('ggml_new_tensor_4d',P,[P,I,L,L,L,L]);setname=api('ggml_set_name',P,[P,C.c_char_p])
    inputflag=api('ggml_set_input',None,[P]);outputflag=api('ggml_set_output',None,[P])
    transpose=api('ggml_transpose',P,[P,P]);mulmat=api('ggml_mul_mat',P,[P,P,P])
    softmax=api('ggml_soft_max_ext',P,[P,P,P,C.c_float,C.c_float])
    flash=api('ggml_flash_attn_ext',P,[P,P,P,P,P,C.c_float,C.c_float,C.c_float])
    nbytes=api('ggml_nbytes',S,[P]);nodecount=api('ggml_graph_n_nodes',I,[P]);node=api('ggml_graph_node',P,[P,I]);opname=api('ggml_op_name',C.c_char_p,[I])
    backend=api('ggml_backend_vk_init',P,[S],vk)(0);assert backend
    device=api('ggml_backend_get_device',P,[P])(backend)
    description=api('ggml_backend_dev_description',C.c_char_p,[P])(device).decode()
    supports=api('ggml_backend_dev_supports_op',C.c_bool,[P,P])
    buft=api('ggml_backend_get_default_buffer_type',P,[P])(backend)
    maximum=api('ggml_backend_buft_get_max_size',S,[P])(buft)
    context=init(Init(4*1024*1024,None,True));assert context
    uploads=[]
    def tensor(name,array):
        array=np.ascontiguousarray(array); dims=list(reversed(array.shape))+[1]*(4-array.ndim)
        t=new(context,1 if array.dtype==np.float16 else 0,*dims);setname(t,name.encode());inputflag(t);outputflag(t)
        assert nbytes(t)==array.nbytes;uploads.append((t,array));return t
    K=M=N=64
    K=128
    x=(((np.arange(M*K).reshape(M,K)*13+3)%127-63)/256).astype('<f2')
    y=(((np.arange(K*N).reshape(K,N)*17+5)%251-125)/128).astype('<f4')
    X=tensor('cache_left_f16',x);Ybase=tensor('cache_rhs_storage',y)
    Y=transpose(context,Ybase);setname(Y,b'cache_rhs_noncontiguous')
    prime=mulmat(context,X,Y);setname(prime,b'cache_prime');outputflag(prime)
    writer_arrays={}
    if writer=='softmax':
        values=(((np.arange(4*32768).reshape(4,32768)*11+1)%257-128)/32).astype('<f4')
        scratch=softmax(context,tensor('large_softmax_input',values),None,1.0,0.0)
        writer_arrays={'values':values}
    else:
        # 4096 columns and 64 query rows force the pinned mask-optimization predicate.
        q=(((np.arange(64*64).reshape(64,64)*7+3)%127-63)/256).astype('<f4')
        k=(((np.arange(4096*64).reshape(4096,64)*11+5)%127-63)/256).astype('<f2')
        v=(((np.arange(4096*64).reshape(4096,64)*19+7)%251-125)/128).astype('<f2')
        mask=np.zeros((64,4096),dtype='<f2')
        for row in range(64):
            for block in range(256):
                if (block+row)%7==0:mask[row,block*16:(block+1)*16]=-np.inf
        scratch=flash(context,tensor('flash_q',q),tensor('flash_k',k),tensor('flash_v',v),tensor('flash_mask',mask),1/math.sqrt(64),0.0,0.0)
        writer_arrays={'q':q,'k':k,'v':v,'mask':mask}
    setname(scratch,('scratch_writer_'+writer).encode());outputflag(scratch)
    reuse=mulmat(context,X,Y);setname(reuse,b'cache_reuse');outputflag(reuse)
    assert Tensor.from_address(prime).src[1]==Tensor.from_address(reuse).src[1]==Y
    graph=api('ggml_new_graph',P,[P])(context);expand=api('ggml_build_forward_expand',None,[P,P])
    for t in (prime,scratch,reuse):expand(graph,t)
    nodes=[]
    for i in range(nodecount(graph)):
        p=node(graph,i);t=Tensor.from_address(p)
        info={'index':i,'op':opname(t.op).decode(),'shape':list(t.ne),'strides':list(t.nb),'bytes':nbytes(p),'supported':bool(supports(device,p))}
        assert info['supported'] and info['bytes']<=maximum and info['bytes']<16*1024*1024,info
        nodes.append(info)
    assert [n['op'] for n in nodes if n['op'] not in ('TRANSPOSE','VIEW','RESHAPE','PERMUTE')]==['MUL_MAT','SOFT_MAX' if writer=='softmax' else 'FLASH_ATTN_EXT','MUL_MAT']
    allocator=api('ggml_gallocr_new',P,[P])(buft)
    assert api('ggml_gallocr_alloc_graph',C.c_bool,[P,P])(allocator,graph)
    put=api('ggml_backend_tensor_set',None,[P,P,S,S]);get=api('ggml_backend_tensor_get',None,[P,P,S,S])
    storage=[]
    for t,array in uploads:
        put(t,array.ctypes.data,0,array.nbytes);check=np.empty_like(array);get(t,check.ctypes.data,0,check.nbytes)
        assert check.tobytes()==array.tobytes()
        storage.append({'shape':list(Tensor.from_address(t).ne),'strides':list(Tensor.from_address(t).nb),'dtype':str(array.dtype),'sha256':sha(check.tobytes()),'bytes':check.nbytes})
    save(out/'graph-entry.json',{'writer':writer,'device':description,'nodes':nodes,'verified_uploads':storage,'same_rhs_pointer':hex(Y),'compute_calls_planned':1,'atol':0.0002,'rtol':0.00005,'flash_atol':0.0005,'flash_rtol':0.002,'large_softmax_atol':0.000002,'large_softmax_rtol':0.0002})
    start=time.monotonic();status=api('ggml_backend_graph_compute',I,[P,P])(backend,graph);elapsed=time.monotonic()-start
    save(out/'compute-completion.json',{'status':status,'seconds':elapsed,'calls':1});assert status==0
    reference=np.empty((N,M),dtype=np.float64)
    yh=y.astype('<f2').astype('f8')
    for n in range(N):
        for m in range(M):reference[n,m]=math.fsum(float(x[m,k])*yh[k,n] for k in range(K))
    results={}
    for label,t in [('prime',prime),('reuse',reuse)]:
        data=np.empty((N,M),dtype='<f4');get(t,data.ctypes.data,0,data.nbytes);np.save(out/(label+'.npy'),data,allow_pickle=False)
        err=np.abs(data.astype('f8')-reference);finite=bool(np.isfinite(data).all())
        results[label]={'finite':finite,'max_abs':float(np.max(err)),'pass':finite and bool(np.all(err<=0.0002+0.00005*np.abs(reference))),'sha256':sha(data.tobytes())}
    if writer=='softmax':
        values=writer_arrays['values'].astype('f8');exp=np.exp(values-values.max(axis=1,keepdims=True));ref=exp/exp.sum(axis=1,keepdims=True)
        data=np.empty((4,32768),dtype='<f4');get(scratch,data.ctypes.data,0,data.nbytes)
        results['writer']={'finite':bool(np.isfinite(data).all()),'max_abs':float(np.max(np.abs(data-ref))),'pass':bool(np.isfinite(data).all() and np.all(np.abs(data-ref)<=0.000002+0.0002*np.abs(ref)))}
    else:
        a=writer_arrays;score=a['q'].astype('f8')@a['k'].astype('f8').T/math.sqrt(64)+a['mask'].astype('f8')
        exp=np.exp(score-score.max(axis=1,keepdims=True));ref=(exp/exp.sum(axis=1,keepdims=True))@a['v'].astype('f8')
        data=np.empty((64,64),dtype='<f4');get(scratch,data.ctypes.data,0,data.nbytes)
        results['writer']={'finite':bool(np.isfinite(data).all()),'max_abs':float(np.max(np.abs(data-ref))),'pass':bool(np.isfinite(data).all() and np.all(np.abs(data-ref)<=0.0005+0.002*np.abs(ref)))}
    np.save(out/'writer.npy',data,allow_pickle=False)
    save(out/'numerical-result.json',{'writer':writer,'results':results,'numerical_pass':all(r['pass'] for r in results.values()),'path_coverage':'Must also verify conversion -> scratch writer -> same conversion reuse in actual Vulkan debug dispatch log. Numerical result alone is insufficient.'})
    api('ggml_gallocr_free',None,[P])(allocator);api('ggml_free',None,[P])(context);api('ggml_backend_free',None,[P])(backend);directory.close()
if __name__=='__main__':main()
