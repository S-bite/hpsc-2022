#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace cooperative_groups;

__global__ void bucket_sort(int *key, int *bucket, int *offset1, int *offset2,
                            int n, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  grid_group grid = this_grid();
  if (i < range) {
    bucket[i] = 0;
    offset1[i] = 0;
  }
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
    if (key[i] != range - 1)
      atomicAdd(&offset1[key[i] + 1], 1);
  }
  grid.sync();
  if (i >= range) {
    return;
  }
  for (int j = 1; j < range; j <<= 1) {
    offset2[i] = offset1[i];
    grid.sync();
    if (i >= j)
      offset1[i] += offset2[i - j];
    grid.sync();
  }
  int j = offset1[i];
  for (; bucket[i] > 0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  const int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n * sizeof(int));
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }

  printf("\n");
  int *bucket, *offset1, *offset2;
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&offset1, range * sizeof(int));
  cudaMallocManaged(&offset2, range * sizeof(int));
  void *args[] = {(void *)&key,     (void *)&bucket, (void *)&offset1,
                  (void *)&offset2, (void *)&n,      (void *)&range};
  cudaLaunchCooperativeKernel((void *)bucket_sort, 1, n, args);
  cudaDeviceSynchronize();
  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset1);
  cudaFree(offset2);
  return 0;
}
