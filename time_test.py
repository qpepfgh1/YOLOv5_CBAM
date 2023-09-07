import time
import cv2
import io
from PIL import Image
import numpy as np
from numba import jit, cuda

path = "./datasets/15__00_01_54_02_01_NG.Jpg"
with open(path, 'rb') as f:
    data = f.read()
data_io = io.BytesIO(data)



start_time = time.time()
print(f"per before time: {time.time() - start_time:.5f} sec")
for i in [1,2,3]:
    print(f"per start time: {time.time() - start_time:.5f} sec")
    img = Image.open(data_io)
    # img = np.array(img)
    print(f"per end time: {time.time() - start_time:.5f} sec")