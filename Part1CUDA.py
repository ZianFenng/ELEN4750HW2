import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

# Here defines the kernel code
kernel = """
    __global__ void MatrixTranspose(unsigned int Dim, float *A, float *AT)
    {
    // 2D Thread ID
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    
    // Transpose the square matrix A element by element
    AT[idy*Dim + idx] = A[idx*Dim + idy];
    
    }
    """

# Here the keernel code is compiled, only once is ok
mod = compiler.SourceModule(kernel)

# Get the kernel function from the compiled module
# then we could call the function by using func = ( parameter1, parameter2, ...)
func = mod.get_function("MatrixTranspose")

# Define the size of the loop, this size here should be less then 32,
# since the local memory size of our device is only 1024
loop_size = 30

ex_size = 5

cpu_times = np.zeros(loop_size).astype(np.float64)
gpu_times = np.zeros(loop_size).astype(np.float64)

for n in range(1, loop_size):
    # Generate random origin nxn matrix
    A = np.random.rand(n,n).astype(np.float32)
    # Transfer origin matrix and empty transposed matrix from host memory to device memory
    A_gpu = gpuarray.to_gpu(A)
    AT_gpu = gpuarray.empty(A.shape, A.dtype)

    # Call the function, the block tells the device how to allocate local memory
    gpu_t = []
    for M in xrange(ex_size):
        start_g = time.time()
        func(np.uint32(n), A_gpu, AT_gpu, block = (n, n, 1))
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

    AT = AT_gpu.get()
    # Judge the equivalence of the result and the original matrix to see if it's symmetric
    equivalence = np.array_equal(A, AT)
    
    # Try the transpose func of np
    # ATnp = np.matrix.transpose(A)

    # Print the result
    print '\n The matrix size is n * n and n equals to ', n
    print '\n Original Matrix A: \n', A
    print '\n Transposed Matrix AT: \n', AT
    #print '\n Transposed Matrix ATnp: \n', ATnp
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

plt.savefig('TransposeCUDA.png')

