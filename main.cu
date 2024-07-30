#include "include/errorCheckUtils.cuh"
#include "include/kernels.cuh"

#include <cuda_runtime.h>

#include <random>
#include <type_traits> // std::is_same
#include <iostream>

#define ceil(a, b) ( a + b - 1 )/b // utility function to calc ceil(a/b)

float randomFloat() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float randomInt() {
	return rand();
}


template<typename TP>
inline void initArray(TP* A, int size) {
	for (int i = 0; i < size; ++i)
		if (std::is_same<TP, float>::value) {
			A[i] = randomFloat();
		}
		else if(std::is_same<TP, int>::value) {
			A[i] = randomInt();
		}
}

template<typename TP>
inline TP calcSumHost(TP* A, int size){
	TP sum = 0;
	for (int i = 0; i < size; ++i)
		sum += A[i];

	return sum;
}

template<typename TP>
inline void checkResult(TP sum_h, TP sum_d) {
	std::cout <<"\nCPU SUM: "<<sum_h<<" GPU SUM: "<< sum_d << " ERROR: " << abs(sum_d - sum_h)*100.0/(sum_h)<<"%";
}


template<typename TP>
TP callReduce1(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, block.x));

		reduce1<TP> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		// store result pointer before swapping
		_res = d_res;

		// swap pointers
		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}


	// copy result back to CPU
	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}

template<typename TP>
TP callReduce2(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, block.x));

		reduce2<TP> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		// store result pointer before swapping
		_res = d_res;

		// swap pointers
		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}

	// copy result back to CPU
	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}


template<typename TP>
TP callReduce3(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, block.x));

		reduce3<TP> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		// store result pointer before swapping
		_res = d_res;

		// swap pointers
		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}

	// copy result back to CPU
	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}

template<typename TP>
TP callReduce4(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, block.x));

		reduce4<TP> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		// store result pointer before swapping
		_res = d_res;

		// swap pointers
		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}

	// copy result back to CPU
	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}


template<typename TP>
TP callReduce5(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, (2 * block.x) ));

		reduce5<TP> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		_res = d_res;

		// swap pointers
		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}

	// copy result back to CPU
	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}


template<typename TP>
TP callReduce6(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, (2 * block.x) ));

		reduce6<TP> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		_res = d_res;

		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}

	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}


template<typename TP>
TP callReduce7(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);


	TP* d_res;
	CUDA_CALL(cudaMalloc((void**)&d_res, size * sizeof(TP)));

	TP* _res;

	// recursion. reduce by /2 till only 1 element is left
	int N = size; // org size at first 
	while (N > 1) {
		dim3 grid(ceil(N, (2 * block.x) ));

		reduce7<TP, 256> <<<grid, block >>> (A_d , d_res, N);
		cudaDeviceSynchronize();

		N = grid.x;

		_res = d_res;

		TP* tmp = A_d;
		A_d = d_res;
		d_res = tmp;
	}

	TP res;
	CUDA_CALL(cudaMemcpy(&res, _res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}


template<typename TP>
TP callReduce8(TP* A_h, int size) {
	TP* A_d;
	CUDA_CALL(cudaMalloc((void**)&A_d, size * sizeof(TP)));
	CUDA_CALL(cudaMemcpy(A_d, A_h, size * sizeof(TP), cudaMemcpyHostToDevice));

	const int BLOCK_SIZE = 256;
	dim3 block(BLOCK_SIZE);

	int N = size; // org size at first 

	const int GRID_SIZE = std::min<int>(ceil(N, block.x), 2 * 46); // 2 x NUM SMs grids at most.
	TP* d_tmp;
	CUDA_CALL(cudaMalloc((void**)&d_tmp, GRID_SIZE * sizeof(TP)));

	// only 2 calls required, since we account multiple grid in once
	// launching 8 * NUM_SMs first
	// and then 1 to reduce whatever's left
	reduce8<TP, 256> <<< GRID_SIZE, block >>> (A_d , d_tmp, N);
	cudaDeviceSynchronize();

	N = GRID_SIZE; // will now reduce GRID_SIZE elements
	TP *d_res;
	cudaMalloc((void**)&d_res, sizeof(TP));
	// Will only launch 1 grid.
	reduce8<TP, 256> <<< 1, block >>> (d_tmp , d_res, N);
	cudaDeviceSynchronize();

	TP res;
	CUDA_CALL(cudaMemcpy(&res, d_res, sizeof(TP), cudaMemcpyDeviceToHost));
	return res;
}


int main() {

	int size = 1 << 22;

	#define DTYPE int
	DTYPE* h_A = (DTYPE*)malloc(size * sizeof(DTYPE));
	initArray<DTYPE>(h_A, size);

	DTYPE h_sum = calcSumHost<DTYPE>(h_A, size);

	DTYPE d_sum = callReduce1<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce2<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce3<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce4<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce5<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce6<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce7<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	d_sum = callReduce8<DTYPE>(h_A, size);
	checkResult<DTYPE>(h_sum, d_sum);

	return 0;
}