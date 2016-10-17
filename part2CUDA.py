import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

# Here defines the kernel code
kernel = """
    __global__ void MatrixMult(int Mdim, int Ndim, int Pdim,  float *A, float *AT, float *result)
    {
        // 2D Thread ID
        int idx = threadIdx.x;
        int idy = threadIdx.y;
        
        float tmp = 0.0f;
        
        int k;
        
        for(k = 0; k < Ndim; k++)
        {
            tmp += A[ idx*Ndim + k]* AT[ k*Pdim + idy];
        }
    
        result[idx*Pdim + idy] = tmp;
        //AT[idy*Dim + idx] = A[idx*Dim + idy];
    
    }
    
    __global__ void MatrixTranspose(unsigned int Mdim, unsigned int Ndim, float *A, float *AT)
    // Here we do the transpose, A is MxN, AT is NxM
    {
    // 2D Thread ID
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    
    // Transpose the square matrix A element by element
    AT[idy*Mdim + idx] = A[idx*Ndim + idy];
    
    }
    """

# Compile the kernel code, we only need to compile the kernel code once
mod = compiler.SourceModule(kernel)

# get the kernel function from the compiled module
# then we could call the function by using func = ( parameter1, parameter2, ...)
MatrixMul = mod.get_function("MatrixMult")
MatrixTran = mod.get_function("MatrixTranspose")


# Define the size of the loop
loop_size = 10

# Define the maximum value of the matrix
Max = 20

for n in range(1,loop_size):
    A = np.random.randint(Max, size = (3*n,2*n)).astype(np.float32)
    AT = np.matrix.transpose(A)

    A_gpu = gpuarray.to_gpu(A)
    AT_gpu = gpuarray.empty((2*n,3*n),A.dtype)
    result_gpu = gpuarray.empty((3*n,3*n),A.dtype)

    MatrixTran(np.uint32(3*n), np.uint32(2*n), A_gpu, AT_gpu, block = (3*n, 2*n, 1))
    MatrixMul(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu, block = (3*n, 3*n, 1) )

    AT = AT_gpu.get()
    result = result_gpu.get()
    result_np = np.dot(A,AT)
    equivalence = np.array_equal(result, result_np)

    print '\n Matrix A is \n', A
    print '\n Matrix AT is \n', AT
    print '\n Matrix result_gpu is \n', result
    print '\n Matrix result_np is \n', result_np
    if equivalence:
        print '\n CPU and GPU\'s results match\n'
    else:
        print '\n The results don\'t match\n'