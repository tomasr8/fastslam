#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <algorithm>

#include "utils.h"

#define MAX_THREADS_PER_BLOCK 512

// NOTE "scan" and "prefix-sum" (psum) are used interchangably in this context

__global__
void block_psum(const unsigned int * const g_in,
                      unsigned int * const g_out,
                      unsigned int * const g_sums,
                const size_t n)
{
  extern __shared__ unsigned int smem[];
  const size_t bx = blockIdx.x * blockDim.x;
  const size_t tx = threadIdx.x;
  const size_t px = bx + tx;
  int offset = 1;

  // init
  smem[2*tx]   = g_in[2*px];
  smem[2*tx+1] = g_in[2*px+1];

  ////
  // up sweep
  ////
  for (int d = n >> 1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (tx < d)
    {
      int ai = offset * (2*tx+1) - 1;
      int bi = offset * (2*tx+2) - 1;

      smem[bi] += smem[ai];
    }
    offset <<= 1;
  }

  // save block sum and clear last element
  if (tx == 0) {
    if (g_sums != NULL)
      g_sums[blockIdx.x] = smem[n-1];
    smem[n-1] = 0;
  }

  ////
  // down sweep
  ////
  for (int d = 1; d < n; d <<= 1)
  {
    offset >>= 1;
    __syncthreads();

    if (tx < d)
    {
      int ai = offset * (2*tx+1) - 1;
      int bi = offset * (2*tx+2) - 1;

      // swap
      unsigned int t = smem[ai];
      smem[ai]  = smem[bi];
      smem[bi] += t;
    }
  }
  __syncthreads();

  // save scan result
  if(g_sums == NULL) {
    printf("%d <- %d\n", (int)(2*px), smem[2*tx]);
    printf("%d <- %d\n", (int)(2*px+1), smem[2*tx+1]);
  }

  g_out[2*px]   = smem[2*tx];
  g_out[2*px+1] = smem[2*tx+1];
}

__global__
void scatter_incr(      unsigned int * const d_array,
                  const unsigned int * const d_incr)
{
  const size_t bx = 2 * blockDim.x * blockIdx.x;
  const size_t tx = threadIdx.x;
  const unsigned int u = d_incr[blockIdx.x];
  d_array[bx + 2*tx]   += u;
  d_array[bx + 2*tx+1] += u;
}

// TODO 1) current version only works for len <= MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK
// TODO 2) current version doesnt handle bank conflicts
void psum(const unsigned int * const h_in,
                unsigned int * const h_out,
          const size_t len)
{
  unsigned int *d_in;
  checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int)*len));
  checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int)*len, cudaMemcpyHostToDevice));


  const unsigned int nthreads = MAX_THREADS_PER_BLOCK;
  const unsigned int block_size = 2 * nthreads;
  const unsigned int smem = block_size * sizeof(unsigned int);
  // n = smallest multiple of block_size such that larger than or equal to len
  const size_t n = len % block_size == 0 ? len : (1+len/block_size)*block_size;
  // number of blocks
  int nblocks = n/block_size;

  // allocate memories on gpu
  unsigned int *d_scan, *d_sums, *d_incr;
  checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*n));
  checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int)*nblocks));
  checkCudaErrors(cudaMalloc(&d_incr, sizeof(unsigned int)*nblocks));

  // scan array by blocks (block_size = 2 * num threads)
  block_psum<<<nblocks, nthreads, smem>>>(d_in, d_scan, d_sums, block_size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // scan block sums
  // TODO case when nblocks is bigger than block_size (see TODO 1)
  block_psum<<<1, nthreads, smem>>>(d_sums, d_incr, NULL, block_size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // scatter block sums back to scanned blocks
  scatter_incr<<<nblocks, nthreads>>>(d_scan, d_incr);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // copy scan result back to d_out (cutoff at length len)
  checkCudaErrors(cudaMemcpy(h_out, d_scan, sizeof(unsigned int)*len, cudaMemcpyDeviceToHost));

  // free allocated memories
  checkCudaErrors(cudaFree(d_incr));
  checkCudaErrors(cudaFree(d_sums));
  checkCudaErrors(cudaFree(d_scan));
}


int main(int argc, char *argv[])
{
  const size_t len = 8192 * 16;

  thrust::host_vector<unsigned int> h_in(len);
  thrust::host_vector<unsigned int> h_out(len);

  //thrust::generate(h_in.begin(), h_in.end(), rand);
  for (size_t i = 0; i < h_in.size(); i++)
    h_in[i] = 3*i;

  psum(&h_in[0], &h_out[0], len);

  for (size_t i = 0; i < h_in.size(); i++)
    std::cout << h_in[i] << "  " << h_out[i] << std::endl;

  return 0;
}
