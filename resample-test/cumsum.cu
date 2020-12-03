// NOTE "scan" and "prefix-sum" (psum) are used interchangably in this context
__global__
void block_psum(float *g_in,
    float *g_out,
    float *g_sums,
    int n, int use_gsums)
{
  extern __shared__ float smem[];
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
    if (use_gsums == 1)
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
      float t = smem[ai];
      smem[ai]  = smem[bi];
      smem[bi] += t;
    }
  }
  __syncthreads();

  // save scan result
  g_out[2*px]   = smem[2*tx];
  g_out[2*px+1] = smem[2*tx+1];
}


__global__ void scatter_incr(float *d_array, float *d_incr)
{
    const size_t bx = 2 * blockDim.x * blockIdx.x;
    const size_t tx = threadIdx.x;
    float u = d_incr[blockIdx.x];
    d_array[bx + 2*tx]   += u;
    d_array[bx + 2*tx+1] += u;
}