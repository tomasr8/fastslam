import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

with open("cuda/update3.cu", "r") as f:
    source = f.read()

cuda_update = SourceModule(source)