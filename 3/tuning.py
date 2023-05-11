import matplotlib.pyplot as plt
import numpy as np

from datasetLoader import DatasetLoader
from optimizer import Optimizer

import multiprocessing as mp
# MODE CONFIG - debug, training, or tuning ? 
mode = "tuning"
dataset = DatasetLoader(mode) 


print("beginning of coarse search")

lambdas = np.logspace(-5,-1,9)
coarse_search = np.zeros((lambdas.shape[0], 2))

def train(l):
    optimizer = Optimizer(weight_decay_parameter=l, lr_cycle_magnitude=2, n_epochs=8, batch_size=100, dataset=dataset, hidden_layers_structure=[50, 50], batch_normalization=True, verbose=False)
    _, c, a = optimizer.resolve_with_SDG(plot=False, mode=mode)
    return a


if __name__ == "__main__"   :
    print(f'start with {mp.cpu_count()} CPUs')
    for l_id, l in enumerate(lambdas):

        print(f'\nweight_decay_parameter = {l:.5g}')
        pool = mp.Pool(mp.cpu_count())

        accuracies = pool.map(train, [l for i in range(10)])
        coarse_search[l_id] = np.array([l, np.average(accuracies)])
        pool.close()
    coarse_search[:,1] = coarse_search[:,1]*100
    np.savetxt("coarse_search.txt",coarse_search,delimiter=',', fmt='%f')

# if __name__ == "__main__":   
    

    

#     activites_list = pool.map(process_a_vessel, [locations[1] for locations in locations_group_by_vessel_id])
#     activites = pd.concat(activites_list).reset_index(drop=True)
#     activites.sort_values(by="datetime_start", ascending=True, inplace=True)
#     activites.to_csv("vessel_activites_with_parallelization.csv", index=False)


    

#     print(f"{time.time() - start} seconds")





