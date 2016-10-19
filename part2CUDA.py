import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

# Here defines the kernel code
kernel = """
    __global__ void MatrixMultop2( int Mdim, int Ndim, int Pdim, float* A, float* AT, float* result)
    // Here, A is a matrix of size Mdim x Ndim, AT is a matrix of size Ndim x Pdim
    // The result is of size Mdim x Pdim
    // In this second optimization, one row will be calculated in each work item
    // A row of matrix A will be loaded into private memory
    {
    
        int idx = threadIdx.x;
        int idy;
        int k;
    
        float Arow[1024];
    
        for(k = 0; k < Ndim; k++)
        {
            Arow[k] = A[ idx * Ndim + k];
        }
    
    
    
    
        for(idy = 0; idy < Pdim; idy++)
        {
            float tmp = 0.0f;
    
            for(k = 0; k < Ndim; k++)
            {
                tmp += Arow[ idx * Ndim + k] * AT[ k * Pdim + idy ];
            }
    
            result[ idx * Pdim + idy ] = tmp;
        }
    }
    
    __global__ void MatrixMultop1( int Mdim, int Ndim, int Pdim, float* A, float* AT, float* result)
    // Here, A is a matrix of size Mdim x Ndim, AT is a matrix of size Ndim x Pdim
    // The result is of size Mdim x Pdim
    // In this first optimization, one row will be calculated instead of one element in each work item
    {
        //2D Thread ID
        int idx = threadIdx.x;
        int idy;
    
        int k;
    
    
        for(idy = 0; idy < Pdim; idy++)
        {
            float tmp = 0.0f;
    
            for(k = 0; k < Ndim; k++)
            {
                tmp += A[ idx * Ndim + k] * AT[ k * Pdim + idy ];
            }
    
            result[ idx * Pdim + idy ] = tmp;
        }
    }
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
MatrixMult_op1 = mod.get_function("MatrixMultop1")
MatrixMult_op2 = mod.get_function("MatrixMultop2")

# Define the size of the loop
loop_size = 10

# Define the maximum value of the matrix
Max = 20

cpu_times = np.zeros(loop_size).astype(np.float64)
gpu_times = np.zeros(loop_size).astype(np.float64)
gpu_op1_times = np.zeros(loop_size).astype(np.float64)
gpu_op2_times = np.zeros(loop_size).astype(np.float64)

# Execution size for the time average loop
ex_size = 5;

for n in range(1,loop_size):
    A = np.random.randint(Max, size = (3*n,2*n)).astype(np.float32)
    #AT = np.matrix.transpose(A)

    A_gpu = gpuarray.to_gpu(A)
    AT_gpu = gpuarray.empty((2*n,3*n),A.dtype)
    result_gpu = gpuarray.empty((3*n,3*n),A.dtype)
    result_gpu_op1 = gpuarray.empty((3*n,3*n),A.dtype)
    result_gpu_op2 = gpuarray.empty((3*n,3*n),A.dtype)
    
    MatrixTran(np.uint32(3*n), np.uint32(2*n), A_gpu, AT_gpu, block = (3*n, 2*n, 1))
    AT = AT_gpu.get()
    
    times_cpu = []
    result_np = []
    for M in xrange(ex_size):
        start_c = time.time()
        result_np = np.dot(A,AT)
        finish_c = time.time()
        times_cpu.append(finish_c - start_c)
    cpu_times[n] = np.average(times_cpu)

    times_gpu = []
    for M in xrange(ex_size):
        start = time.time()
        MatrixMul(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu, block = (3*n, 3*n, 1) )
        finish = time.time()
        times_gpu.append(finish - start)
    gpu_times[n] = np.average(times_gpu)

    times_gpu_op1 = []
    for M in xrange(ex_size):
        start1 = time.time()
        MatrixMult_op1(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu, block = (3*n, 1, 1))
        finish1 = time.time()
        times_gpu_op1.append(finish1 - start1)
    gpu_op1_times[n] = np.average(times_gpu_op1)
    
    times_gpu_op2 = []
    for M in xrange(ex_size):
        start2 = time.time()
        MatrixMult_op2(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu, block = (3*n, 1, 1))
        finish2 = time.time()
        times_gpu_op2.append(finish2 - start2)
    gpu_op2_times[n] = np.average(times_gpu_op2)


    result = result_gpu.get()
    result_op1 = result_gpu_op1.get()
    result_op2 = result_gpu_op2.get()

    equivalence = np.array_equal(result, result_np)
    equivalence_op1 = np.array_equal(result_op1, result_np)
    equivalence_op2 = np.array_equal(result_op2, result_np)

    print '\n Matrix A is \n', A
    print '\n Matrix AT is \n', AT
    print '\n Matrix result_gpu is \n', result
    print '\n Matrix result_gpu_op1 is \n', result_op1
    print '\n Matrix result_gpu_op2 is \n', result_op2
    print '\n Matrix result_np is \n', result_np
    if equivalence_op2:
        print '\n CPU and GPU_op2\'s results match\n'
    else:
        print '\n The results don\'t match\n'
    if equivalence_op1:
        print '\n CPU and GPU_op1\'s results match\n'
    else:
        print '\n The results don\'t match\n'
    if equivalence:
        print '\n CPU and GPU\'s results match\n'
    else:
        print '\n The results don\'t match\n'

# Here we draw the figure comparing the CPU and GPU execution time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

plt.gcf()

# Here is define the data and format of our two curve
# CPU execution time will be drawn in red dashes, GPU's in blue dashes
plt.plot(range(0,loop_size), cpu_times, 'y')
plt.plot(range(0,loop_size), gpu_times, 'r')
plt.plot(range(0,loop_size), gpu_op1_times, 'b')
plt.plot(range(0,loop_size), gpu_op2_times, 'g')

plt.legend(['CPU Algorithm','Naive GPU Algorithm', 'GPU Optimization 1', 'GPU Optimization 2'], loc='upper left')

# X axis's unit is the number of strings, each string's length is 13
plt.xlabel('Scale')

# Y axis is the time of execution in seconds
plt.ylabel('Time')

# Here sets the span of X axis
plt.gca().set_xlim(0, loop_size)

# Save the figure drawn for further analysis

plt.savefig('MatrixMulinCUDA.png')