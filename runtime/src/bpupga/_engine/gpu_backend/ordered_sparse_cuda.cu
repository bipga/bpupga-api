#include <cuda_runtime.h>
#include <stdint.h>

// Exactly one thread owns an output element. Sorted CSR neighbours are
// accumulated sequentially in float64, with no atomic floating-point writes.
__global__ void ordered_spmm(const int64_t* ptr, const int64_t* col,
                            const double* val, const double* rhs, double* out,
                            int64_t rows, int64_t width, int64_t s0, int64_t s1) {
    const int64_t row=blockIdx.y;
    const int64_t c=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=rows || c>=width) return;
    double sum=0.;
    for (int64_t k=ptr[row]; k<ptr[row+1]; ++k)
        sum=__dadd_rn(sum,__dmul_rn(val[k],rhs[col[k]*s0+c*s1]));
    out[row*width+c]=sum;
}

extern "C" int bpupga_ordered_spmm(const int64_t* ptr, const int64_t* col,
        const double* val, const double* rhs, double* out, int64_t rows,
        int64_t width, int64_t s0, int64_t s1, cudaStream_t stream) {
    if (rows && width)
        ordered_spmm<<<dim3((width+127)/128,rows),128,0,stream>>>(
            ptr,col,val,rhs,out,rows,width,s0,s1);
    return (int)cudaGetLastError();
}
