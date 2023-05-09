import numpy as np
from multiprocessing import shared_memory

shmname = 'mysl1'

sl1 = shared_memory.ShareableList(name=shmname)



