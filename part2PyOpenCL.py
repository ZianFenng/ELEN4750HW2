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
loop_size = 12

# Maximum value of the matrix
max_value = 20

for n in range(1, loop_size):
    # Generate random origin MxN matrix
    # Set the ratio M:N = 3:2
    A = np.random.randint(max_value, size = ( 3*n, 2*n) ).astype(np.float32)

    A_gpu = cl.array.to_device(queue, A)
    AT_gpu = cl.array.empty(queue, (2*n, 3*n), A.dtype)
    
    # Get the transpose of A
    
    prg.transpose(queue, A.shape, None, np.uint32(3*n), np.uint32(2*n), A_gpu.data, AT_gpu.data)
    
    AT = AT_gpu.get()
    
    result_gpu = cl.array.empty(queue, (3 * n, 3 * n), A.dtype)

    prg.mat_mul(queue, result_gpu.shape, None, np.uint32(3*n), np.uint32(2*n), np.uint32(3*n), A_gpu.data, AT_gpu.data, result_gpu.data )

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



