import matplotlib.pyplot as plt
import numpy as np 

n_epochs = 3

match n_epochs:
    case 1:
        smooth_losses = np.loadtxt('1_epoch')
    case 3:
        smooth_losses = np.loadtxt('3_epoch')
    case _:
        smooth_losses = None

plt.plot(np.arange(len(smooth_losses)), smooth_losses)
plt.xlabel('updates')
plt.ylabel('smooth loss')
plt.title(f'smooth loss according to the updates, for n epochs = {n_epochs}')
plt.show()
