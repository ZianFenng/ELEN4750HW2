import time

import pyopencl as cl
import pyopencl.array
import numpy as np

# Select the desired OpenCL platform; this comes from the demo code
NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()

# Set up a command queue; we need to enable profiling to time GPU operations:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# In this naive implementation of matrix, each work item will only get one element of the result matrix
# A is a MxN matrix, AT, which is the transpose of A, is a NxM matrix
# To make the matrix multilication more general, I use three variable to represent the dimension size of the matrix
# The kernel code mainly reference the KITE-opencl slide
kernel = """
    __kernel void mat_mul_op2( const unsigned int Mdim, const unsigned int Ndim, const unsigned int Pdim, __global float* A, __global float* AT, __global float* result)
    // Here, A is a matrix of size Mdim x Ndim, AT is a matrix of size Ndim x Pdim
    // The result is of size Mdim x Pdim
    // In this second optimization, one row will be calculated in each work item
    // A row of matrix A will be loaded into private memory
    {

        unsigned int i = get_global_id(0);
        unsigned int j;
        unsigned int k;

        float Arow[1024];

        for(k = 0; k < Ndim; k++)
        {
            Arow[k] = A[ i * Ndim + k];
        }




        for(j = 0; j < Pdim; j++)
        {
            float tmp = 0.0f;

            for(k = 0; k < Ndim; k++)
            {
                tmp += A[ i * Ndim + k] * AT[ k * Pdim + j ];
            }

            result[ i * Pdim + j ] = tmp;
        }
    }
    __kernel void mat_mul_op1( const unsigned int Mdim, const unsigned int Ndim, const unsigned int Pdim, __global float* A, __global float* AT, __global float* result)
    // Here, A is a matrix of size Mdim x Ndim, AT is a matrix of size Ndim x Pdim
    // The result is of size Mdim x Pdim
    // In this first optimization, one row will be calculated instead of one element in each work item
    {

        unsigned int i = get_global_id(0);
        unsigned int j;

        unsigned int k;


        for(j = 0; j < Pdim; j++)
        {
            float tmp = 0.0f;

            for(k = 0; k < Ndim; k++)
            {
                tmp += A[ i * Ndim + k] * AT[ k * Pdim + j ];
            }

            result[ i * Pdim + j ] = tmp;
        }
    }
    __kernel void mat_mul( const unsigned int Mdim, const unsigned int Ndim, const unsigned int Pdim, __global float* A, __global float* AT, __global float* result)
    // Here, A is a matrix of size Mdim x Ndim, AT is a matrix of size Ndim x Pdim
    // The result is of size Mdim x Pdim
    {

        unsigned int i = get_global_id(0);
        unsigned int j = get_global_id(1);

        unsigned int k;

        float tmp = 0.0f;

        for(k = 0; k < Ndim; k++)
        {
            tmp += A[ i * Ndim + k] * AT[ k * Pdim + j ];
        }

        result[ i * Pdim + j ] = tmp;
    }

    __kernel void transpose(unsigned int Mdim, unsigned int Ndim,__global float* A, __global float* AT)
    {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    AT[j*Mdim + i] = A[i*Ndim + j];

    }
    """

# Compile the kernel code
prg = cl.Program(ctx, kernel).build()

# Define the size of the loop
loop_size = 50

# Maximum value of the matrix
max_value = 20

cpu_times = np.zeros(loop_size).astype(np.float64)
gpu_times = np.zeros(loop_size).astype(np.float64)
gpu_op1_times = np.zeros(loop_size).astype(np.float64)
gpu_op2_times = np.zeros(loop_size).astype(np.float64)

transpose_cpu_times = np.zeros(loop_size).astype(np.float64)
transpose_gpu_times = np.zeros(loop_size).astype(np.float64)

# Execution size for the time average loop
ex_size = 10;


for n in range(1, loop_size):
    # Generate random origin MxN matrix
    # Set the ratio M:N = 3:2
    A = np.random.randint(max_value, size = ( 3*n, 2*n) ).astype(np.float32)

    A_gpu = cl.array.to_device(queue, A)
    AT_gpu = cl.array.empty(queue, (2*n, 3*n), A.dtype)

    # Get the transpose of A
    time_cpu_transpose = []
    for M in xrange(ex_size):
        start_tc = time.time()
        AT = np.matrix.transpose(A)
        finish_tc = time.time()
        time_cpu_transpose.append(finish_tc - start_tc)
    transpose_cpu_times[n] = np.average(time_cpu_transpose)

    time_gpu_transpose = []
    for M in xrange(ex_size):
        start_tg = time.time()
        prg.transpose(queue, A.shape, None, np.uint32(3*n), np.uint32(2*n), A_gpu.data, AT_gpu.data)
        finish_tg = time.time()
        time_gpu_transpose.append(finish_tg - start_tg)
    transpose_gpu_times[n] = np.average(time_gpu_transpose)
    

    AT = AT_gpu.get()

    result_gpu = cl.array.empty(queue, (3 * n, 3 * n), A.dtype)
    result_gpu_op1 = cl.array.empty(queue, (3 * n, 3 * n), A.dtype)
    result_gpu_op2 = cl.array.empty(queue, (3 * n, 3 * n), A.dtype)

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
        prg.mat_mul(queue, result_gpu.shape, None, np.uint32(3*n), np.uint32(2*n), np.uint32(3*n), A_gpu.data, AT_gpu.data, result_gpu.data )
        finish = time.time()
        times_gpu.append(finish - start)
    gpu_times[n] = np.average(times_gpu)

    times_gpu_op1 = []
    for M in xrange(ex_size):
        start1 = time.time()
        prg.mat_mul_op1(queue, (3*n,), None, np.uint32(3*n), np.uint32(2*n), np.uint32(3*n), A_gpu.data, AT_gpu.data, result_gpu_op1.data )
        finish1 = time.time()
        times_gpu_op1.append(finish1 - start1)
    gpu_op1_times[n] = np.average(times_gpu_op1)

    times_gpu_op2 = []
    for M in xrange(ex_size):
        start2 = time.time()
        prg.mat_mul_op2(queue, (3*n,), None, np.uint32(3*n), np.uint32(2*n), np.uint32(3*n), A_gpu.data, AT_gpu.data, result_gpu_op2.data )
        finish2 = time.time()
        times_gpu_op2.append(finish2 - start2)
    gpu_op2_times[n] = np.average(times_gpu_op2)

    result = result_gpu.get()
    result_op1 = result_gpu_op1.get()
    result_op2 = result_gpu_op2.get()
#result_np = np.dot(A,AT)

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

plt.subplot(212)
plt.plot(range(0,loop_size), transpose_gpu_times, 'y')
plt.plot(range(0,loop_size), transpose_cpu_times, 'r')

plt.legend(['GPU Transpose','CPU Transpose'], loc='upper left')

# Save the figure drawn for further analysis

plt.savefig('MatrixMulinOpenCL.png')
