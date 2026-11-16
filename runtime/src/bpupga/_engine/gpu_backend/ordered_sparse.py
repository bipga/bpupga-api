"""Fixed-order CUDA SpMM; CSR construction stays on CUDA and is cached weakly."""
import ctypes,fcntl,hashlib,os,subprocess,tempfile,threading,weakref,shutil
from pathlib import Path
import torch

_LIB={};_CSR={};_LOCK=threading.Lock()

def library(device):
    compiler=os.environ.get('BPUPGA_NVCC') or shutil.which('nvcc')
    if not compiler:
        raise RuntimeError('CUDA Toolkit nvcc is required for ordered sparse multiplication')
    source=Path(__file__).with_name('ordered_sparse_cuda.cu')
    architecture=''.join(map(str,torch.cuda.get_device_capability(device)))
    digest=hashlib.sha256(source.read_bytes()+os.fsencode(compiler)+architecture.encode()).hexdigest()[:20]
    with _LOCK:
        if digest in _LIB:return _LIB[digest]
        cache=Path(tempfile.gettempdir())/f'bpupga_sgwl_spmm_{os.getuid()}'
        cache.mkdir(mode=0o700,exist_ok=True);target=cache/f'{digest}.so'
        with (cache/f'{digest}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not target.exists():
                temporary=target.with_suffix(f'.{os.getpid()}.so')
                result=subprocess.run([compiler,'-std=c++17','-O3','--fmad=false','--shared',
                    '-Xcompiler=-fPIC','-gencode',f'arch=compute_{architecture},code=sm_{architecture}',
                    str(source),'-o',str(temporary)],capture_output=True,text=True)
                if result.returncode:raise RuntimeError('Ordered SpMM build failed: '+result.stderr[-6000:])
                os.replace(temporary,target)
        lib=ctypes.CDLL(str(target));p=ctypes.c_void_p;i=ctypes.c_int64
        lib.bpupga_ordered_spmm.argtypes=[p,p,p,p,p,i,i,i,i,p]
        lib.bpupga_ordered_spmm.restype=ctypes.c_int;_LIB[digest]=lib
        return lib

def csr(matrix):
    key=id(matrix);old=_CSR.get(key)
    if old is not None and old[0]() is matrix:return old[1:]
    indices=matrix.indices();values=matrix.values()
    counts=torch.bincount(indices[0],minlength=matrix.shape[0])
    ptr=torch.cat((torch.zeros(1,device=matrix.device,dtype=torch.int64),counts.cumsum(0)))
    cols=indices[1].contiguous()
    def release(ref):
        if _CSR.get(key,(None,))[0] is ref:_CSR.pop(key,None)
    _CSR[key]=(weakref.ref(matrix,release),ptr,cols,values)
    return ptr,cols,values

def mm(matrix,rhs):
    if matrix.layout==torch.strided:return matrix@rhs
    if matrix.device.type!='cuda':raise ValueError('Deterministic S-GWL requires CUDA; no CPU fallback')
    if matrix.layout!=torch.sparse_coo or not matrix.is_coalesced():raise ValueError('Expected sorted coalesced COO')
    if matrix.dtype!=torch.float64 or rhs.dtype!=torch.float64 or matrix.device!=rhs.device:
        raise ValueError('Expected same-device float64 operands')
    if rhs.ndim!=2 or matrix.shape[1]!=rhs.shape[0]:raise ValueError('Invalid SpMM shapes')
    if matrix.shape[0]>65535:raise ValueError('Ordered SpMM row launch limit exceeded')
    ptr,cols,values=csr(matrix);out=torch.empty((matrix.shape[0],rhs.shape[1]),device=rhs.device,dtype=rhs.dtype)
    lib=library(matrix.device)
    code=lib.bpupga_ordered_spmm(ptr.data_ptr(),cols.data_ptr(),values.data_ptr(),rhs.data_ptr(),out.data_ptr(),
        matrix.shape[0],rhs.shape[1],rhs.stride(0),rhs.stride(1),torch.cuda.current_stream(matrix.device).cuda_stream)
    if code:raise RuntimeError(f'Ordered SpMM CUDA launch failed: {code}')
    return out
