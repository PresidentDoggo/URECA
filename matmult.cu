#include <iostream>
#define N 1000
#define TILE 16
#define CUDA_CHECK(call) do { \
		cudaError_t err = call; \
		if (err != cudaSuccess){ \
			fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
			cudaGetErrorString(err), __FILE__, __LINE__); \
			exit(err); \
		} \
	} while(0)

__global__ void multMatrices(const float (*A)[N], const float(*B)[N], float (*C)[N]) {

	// Shared memory optimization
	__shared__ float rowA[TILE][TILE];
	__shared__ float colB[TILE][TILE];

	// Calculate global column and row separately
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	// Sum for totalling the products of row/column pairs from A and B to calculate each element of C
	float sum = 0.0f;

	// Since we load only TILE elements from a row (from A) or column (from B) at a time, we must loop ceil(N/TILE) times
	for(int n = 0 ; n < (N + TILE-1)/TILE ; n++) {

		// Calculating this iteration's tile coordinates for each thread
		int tileCol = threadIdx.x + n*blockDim.x;
		int tileRow = threadIdx.y + n*blockDim.y;

		// Loading it the tile temporarily (if the coordinate is not out of bounds of A and B)
		rowA[threadIdx.y][threadIdx.x] = (row < N && tileCol < N) ? A[row][tileCol] : 0.0f;
		colB[threadIdx.y][threadIdx.x] = (tileRow < N && col < N) ? B[tileRow][col] : 0.0f;

		// Making sure all temporary tile elements are loaded
		__syncthreads();

		// Calculating the product of row/column pairs for this tile and adding it to the total sum for all tiles
		#pragma unroll
		for(int i = 0 ; i < TILE ; i++) {

			sum += rowA[threadIdx.y][i] * colB[i][threadIdx.x];

		}

		// Syncing again otherwise other warps may overwrite the temporary tile in the next iteration
		__syncthreads();

	}

	if(row < N && col < N)
		C[row][col] = sum;	

}


void printMatrix(const float (*M)[N]){

	for (int i = 0 ; i < N ; i++) {
		for (int j = 0 ; j < N ; j++) {
			printf("%.2f ", M[i][j]);
		}
		printf("\n");
	}

}

int check(const float (*A)[N], const float (*B)[N], const float (*C)[N]) {

	float allowed = 0.01;

	for (int i = 0 ; i < N ; i++) {

		for (int j = 0 ; j < N ; j++) {

			float result = 0.0;
			for (int k = 0 ; k < N ; k++) {
				result += A[i][k] * B[k][j];
			}

			float error = fabs(result - C[i][j]) / result;
			if (error > allowed) {
				printf("Error at position (%i,%i): Expected: %.2f, Actual: %.2f, Error: %.2f", i, j, result, C[i][j], error);
				return 1;
			}
		}
	}
	return 0;

}


int main() {

	// Allocate host memory
	float (*h_A)[N] = new float[N][N];
	float (*h_B)[N] = new float[N][N];
	float (*h_C)[N] = new float[N][N]();


	// Initialize input vectors
	for(int i = 0 ; i < N ; i++) {
		for (int j = 0 ; j < N ; j++) {

			h_A[i][j] = 100.0f * ((float) rand() / (float) RAND_MAX);
			h_B[i][j] = 100.0f * ((float) rand() / (float) RAND_MAX);

		}


	}

	
	// Allocate device memory
	float (*d_A)[N], (*d_B)[N], (*d_C)[N];

	size_t size = N * N * sizeof(float);

	CUDA_CHECK(cudaMalloc((void**)&d_A, size));
	CUDA_CHECK(cudaMalloc((void**)&d_B, size));
	CUDA_CHECK(cudaMalloc((void**)&d_C, size));

	// Copy data from host to device
	CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));


	// Declare cudaEvents (reuse for each run, destroy later)
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	dim3 threadsPerBlock(TILE, TILE);

        dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        cudaEventRecord(start);

	multMatrices<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

	CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        double FLOPS = ((double) 2*N*N*N / ms) / 1000000.0f;


	if (N < 101) {
	
		printf("Matrix A:\n");
		printMatrix(h_A);
		printf("Matrix B:\n");
		printMatrix(h_B);
		printf("Output matrix C:\n");
		printMatrix(h_C);


		if (check(h_A, h_B, h_C) == 1) {
			printf("\nCalculation erroneous");
		}
		else {
			printf("\nCalculation correct");
		}	
	}

        printf("\n%d x %d block: Elapsed time = %fs, GFLOPS/s = %f\n", threadsPerBlock.x, threadsPerBlock.y, ms/1000.0f, FLOPS);

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;



}
