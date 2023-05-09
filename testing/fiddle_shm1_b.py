import numpy as np
from multiprocessing import shared_memory

import numpy as np
from multiprocessing import shared_memory

shmname = 'myshm1'

shm1 = shared_memory.SharedMemory(name=shmname)
# c = np.ndarray((6,), dtype=np.int64, buffer=shm1.buf)