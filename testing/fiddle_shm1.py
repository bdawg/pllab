import numpy as np
from multiprocessing import shared_memory

shmname = 'myshm1'
data = np.array([1, 1, 2, 3, 5, 8]).astype('int64')

# shm1 = shared_memory.SharedMemory(name=shmname, create=True, size=data.nbytes)
shm1 = shared_memory.SharedMemory(name=shmname, create=True, size=4096)

# Shared numpy aray:
shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm1.buf)
shared_data[:] = np.copy(data)


#shm1.unlink()
