from gstools import SRF, Gaussian
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_gaussian_process(grid_size, time_len):
    x = y = range(100)
    model = Gaussian(dim=3, var=0.1, len_scale=10)
    srf = SRF(model, seed=20170519)
    num_steps = time_len
    frames = [srf.structured([x,y,i]) for i in range(num_steps)]