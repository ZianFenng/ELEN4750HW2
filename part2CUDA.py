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
    // In this second optimization, one row will be calculated in each thread
    // A row of matrix A will be loaded into private memory to speed up the algorithm
    {
        // 1-D Thread ID
        int idx = threadIdx.x;
        
        int idy;
        int k;
    
        // Preallocate memory space for the row of A
        float Arow[1024];
        
        // Load the row of A from global mem
        for(k = 0; k < Ndim; k++)
        {
            Arow[k] = A[ idx * Ndim + k];
        }
    
    
    
        // Calculate a row of the result matrix
    
        for(idy = 0; idy < Pdim; idy++)
        {
        
            // Use this tmp in private memory to speed up
            float tmp = 0.0f;
    
            // Calculate an element of the result matrix
            for(k = 0; k < Ndim; k++)
            {
                tmp += Arow[k] * AT[ k * Pdim + idy ];
            }
            
            // Store the result element in global memory
            result[ idx * Pdim + idy ] = tmp;
        }
    }
    
    __global__ void MatrixMultop1( int Mdim, int Ndim, int Pdim, float* A, float* AT, float* result)
    // Here, A is a matrix of size Mdim x Ndim, AT is a matrix of size Ndim x Pdim
    // The result is of size Mdim x Pdim
    // In this first optimization, one row will be calculated instead of one element in each thread
    {
        // 1-D Thread ID
        int idx = threadIdx.x;
        
        int idy;
        int k;
    
        // Calculate a row of the result matrix
        for(idy = 0; idy < Pdim; idy++)
        {
            // Use local memory to speed up
            float tmp = 0.0f;
    
            // Calculate elements of the row
            for(k = 0; k < Ndim; k++)
            {
                tmp += A[ idx * Ndim + k] * AT[ k * Pdim + idy ];
            }
    
            // Store the result element in global memory
            result[ idx * Pdim + idy ] = tmp;
        }
    }
    __global__ void MatrixMult(int Mdim, int Ndim, int Pdim,  float *A, float *AT, float *result)
    {
        // 2D Thread ID
        int idx = threadIdx.x;
        int idy = threadIdx.y;
        
        // Use private memory to speed up
        float tmp = 0.0f;
        
        int k;
        
        // Calculate the element of the result matrix
        for(k = 0; k < Ndim; k++)
        {
            tmp += A[ idx*Ndim + k]* AT[ k*Pdim + idy];
        }
    
        // Store the result element in global memory
        result[idx*Pdim + idy] = tmp;
        
    
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

# Preallocate memory space to store the time for different algorithm
# Set the type to be np.float64 so that we can get best accuracy of the timing
cpu_times = np.zeros(loop_size).astype(np.float64)
gpu_times = np.zeros(loop_size).astype(np.float64)
gpu_op1_times = np.zeros(loop_size).astype(np.float64)
gpu_op2_times = np.zeros(loop_size).astype(np.float64)
transpose_cpu_times = np.zeros(loop_size).astype(np.float64)
transpose_gpu_times = np.zeros(loop_size).astype(np.float64)

# Execution size for the time average loop
ex_size = 10;

for n in range(1,loop_size):
    
    # Generate random origin MxN matrix
    # Set the ratio M:N = 3:2
    A = np.random.randint(Max, size = (3*n,2*n)).astype(np.float32)
    
    # Initialize the input buffer A_gpu and other four output buffers
    # AT_gpu is the output of the transpose func, also the input of all matrix mul func
    A_gpu = gpuarray.to_gpu(A)
    AT_gpu = gpuarray.empty((2*n,3*n),A.dtype)
    result_gpu = gpuarray.empty((3*n,3*n),A.dtype)
    result_gpu_op1 = gpuarray.empty((3*n,3*n),A.dtype)
    result_gpu_op2 = gpuarray.empty((3*n,3*n),A.dtype)
    
    
    # In this for loop, we will get the result of cpu transpose and the average timing of the cpu execution
    # The averaging here means to eliminate the influence of uncontrollable glitches, the same for the rest for loops below
    
    time_cpu_transpose = []
    for M in xrange(ex_size):
        start_tc = time.time()
        
        # CPU execution of transpose
        AT_cpu = np.matrix.transpose(A)
        
        finish_tc = time.time()
        time_cpu_transpose.append(finish_tc - start_tc)
    transpose_cpu_times[n] = np.average(time_cpu_transpose)

    # GPU logic of transpose
    time_gpu_transpose = []
    for M in xrange(ex_size):
        start_tg = time.time()
        MatrixTran(np.uint32(3*n), np.uint32(2*n), A_gpu, AT_gpu, block = (3*n, 2*n, 1))
        finish_tg = time.time()
        time_gpu_transpose.append(finish_tg - start_tg)
    transpose_gpu_times[n] = np.average(time_gpu_transpose)

    # Get the transpose result
    AT = AT_gpu.get()

    # Python function for matrix multiplication
    time_cpu = []
    result_np = []
    for M in xrange(ex_size):
        start_c = time.time()
        result_np = np.dot(A,AT)
        finish_c = time.time()
        time_cpu.append(finish_c - start_c)
    cpu_times[n] = np.average(time_cpu)

    # Naive GPU algorithm for matrix multiplication
    time_gpu = []
    for M in xrange(ex_size):
        start_g = time.time()
        MatrixMul(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu, block = (3*n, 3*n, 1) )
        finish_g = time.time()
        time_gpu.append(finish_g - start_g)
    gpu_times[n] = np.average(time_gpu)

    # First optimization of GPU algorithm
    time_gpu_op1 = []
    for M in xrange(ex_size):
        start_g1 = time.time()
        MatrixMult_op1(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu_op1, block = (3*n, 1, 1))
        finish_g1 = time.time()
        time_gpu_op1.append(finish_g1 - start_g1)
    gpu_op1_times[n] = np.average(time_gpu_op1)

    # Second optimization of GPU algorithm
    time_gpu_op2 = []
    for M in xrange(ex_size):
        start_g2 = time.time()
        MatrixMult_op2(np.int32(3*n), np.int32(2*n), np.int32(3*n), A_gpu, AT_gpu, result_gpu_op2, block = (3*n, 1, 1))
        finish_g2 = time.time()
        time_gpu_op2.append(finish_g2 - start_g2)
    gpu_op2_times[n] = np.average(time_gpu_op2)

    # Get the result of three GPU logic to check the correctness of the result
    result = result_gpu.get()
    result_op1 = result_gpu_op1.get()
    result_op2 = result_gpu_op2.get()

    # Check if the results of gpu algorithms are correct
    equivalence_tran = np.array_equal(AT, AT_cpu)
    equivalence = np.array_equal(result, result_np)
    equivalence_op1 = np.array_equal(result_op1, result_np)
    equivalence_op2 = np.array_equal(result_op2, result_np)

    # Print all the result we get and the correctness of these results
    print '\n Matrix A is \n', A
    print '\n Matrix AT is \n', AT
    print '\n Matrix AT_cpu is \n', AT_cpu
    if equivalence_tran:
        print '\n CPU and GPU transpose\'s results match\n'
    else:
        print '\n The transpose results don\'t match\n'
    print '\n Matrix result_gpu is \n', result
    print '\n Matrix result_gpu_op1 is \n', result_op1
    print '\n Matrix result_gpu_op2 is \n', result_op2
    print '\n Matrix result_cpu is \n', result_np
    if equivalence_op2:
        print '\n CPU and GPU_op2\'s results match\n'
    else:
        print '\n The second optimization\'s result is wrong\n'
    if equivalence_op1:
        print '\n CPU and GPU_op1\'s results match\n'
    else:
        print '\n The first optimization\'s result is wrong\n'
    if equivalence:
        print '\n CPU and GPU\'s results match\n'
    else:
        print '\n The naive logic\'s result is wrong\n'


# Here we draw the figure comparing the CPU and GPU execution time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

plt.gcf()

# These four curves are timings of matrix multiplication, they will be in the same subplot
plt.subplot(211)
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


# In this subplot are the timings of transpose logics
plt.subplot(212)
plt.plot(range(0,loop_size), transpose_gpu_times, 'r')
plt.plot(range(0,loop_size), transpose_cpu_times, 'y')

plt.legend(['GPU Transpose','CPU Transpose'], loc='upper left')

# Save the figure drawn for further analysis

plt.savefig('HM2CUDA.png')
