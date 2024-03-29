# In part 1, we should transpose a square matrix of dimension NxN using pyOpenCL
# Then, compare the transposed matrix with the original one to see if they are the same
# If so, the matrix is symmetrix matrix
# Doing so repeatly for different value of N

# I am going to put all data in global memory
# Each work item only deals with one element of the matrix AT(i,j) = A(j,i)




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

kernel = """
    __kernel void func(const unsigned int Dim,__global float* A, __global float* AT)
    {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    
    AT[i*Dim + j] = A[j*Dim + i];
    }
    """

# Compile the kernel code
prg = cl.Program(ctx, kernel).build()

# Define the size of the loop
loop_size = 100

ex_size = 5

cpu_times = np.zeros(loop_size).astype(np.float64)
gpu_times = np.zeros(loop_size).astype(np.float64)

for n in range(1, loop_size):
    # Generate random origin nxn matrix
    A = np.random.rand(n,n).astype(np.float32)
    # Transfer origin matrix and empty transposed matrix from host memory to device memory
    A_gpu = cl.array.to_device(queue, A)
    AT_gpu = cl.array.empty(queue, A.shape, A.dtype)
    
    # Call the kernel here, all data in global memory
    gpu_t = []
    for M in xrange(ex_size):
        start_g = time.time()
        prg.func(queue, A.shape, None, np.uint32(n), A_gpu.data, AT_gpu.data)
        finish_g = time.time()
        gpu_t.append(finish_g - start_g)
    gpu_times[n] = np.average(gpu_t)

    cpu_t = []
    for M in xrange(ex_size):
        start_c = time.time()
        AT_cpu = np.matrix.transpose(A)
        finish_c = time.time()
        cpu_t.append(finish_c - start_c)
    cpu_times[n] = np.average(cpu_t)
    # Get the transposed matrix from GPU
    AT = AT_gpu.get()
    
    # Judge the equivalence
    equivalence = np.array_equal(A, AT)
    
    # Print the result
    print '\n The matrix size is n * n and n equals to ', n
    print '\n Original Matrix A: \n', A
    print '\n Transposed Matrix AT: \n', AT
    if equivalence:
        print '\n AT equals to A, A is symmetric \n'
    else:
        print '\n AT doesn\'t equal to A, A is not symmetric \n'

# Here we draw the figure comparing the CPU and GPU execution time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

plt.gcf()

plt.plot(range(0,loop_size), cpu_times, 'y')
plt.plot(range(0,loop_size), gpu_times, 'r')

plt.legend(['CPU Algorithm','GPU Algorithm'], loc='upper left')

plt.xlabel('Matrix size')

plt.ylabel('Time')

plt.gca().set_xlim(0, loop_size)

plt.savefig('TransposeOpenCL.png')
