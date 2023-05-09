import numpy as np
from multiprocessing import shared_memory

shmname = 'mysl1'

# data = np.array([1, 1, 2, 3, 5, 8]).astype('int64')

sl_nitems = 10
sl_itemlength = 100
# sl1 = shared_memory.ShareableList(range(4), name=shmname)
sl1 = shared_memory.ShareableList([' '*sl_itemlength]*sl_nitems, name=shmname)

sl1[0] = 'hello'
sl1[2] = 3.141



# sl1.shm.unlink()
