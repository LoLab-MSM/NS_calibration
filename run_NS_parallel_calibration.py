
from earm_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('earm_data.csv', index_col=None)
tspan = data['Time'].values
exp_bid = data['norm_IC-RP'].values
exp_smac = data['IMS-RP'].values
exp_parp = data['norm_EC-RP'].values

model_solver = ScipyOdeSimulator(model, tspan, integrator='vode',
                                 integrator_options={'atol': 1e-6,
                                                     'rtol': 1e-6}
                                 )

def obj_fun(parameters=None, plot=False):
    traj = model_solver.run(param_values=parameters)
    x = traj.dataframe
    bid = x['tBid_total'].values
    cparp = x['CPARP_total'].values
    csmac = x['cSmac_total'].values
    bid = (bid - bid.min()) / (bid.max() - bid.min())
    cparp = (cparp - cparp.min()) / (cparp.max() - cparp.min())
    csmac = (csmac - csmac.min()) / (csmac.max() - csmac.min())
    e1 = np.sqrt(((exp_bid - bid) ** 2).sum(axis=0))
    e2 = np.sqrt(((exp_parp - cparp) ** 2).sum(axis=0))
    e3 = np.sqrt(((exp_smac - csmac) ** 2).sum(axis=0))
    error = e1+e2+e3
    if plot:
        plt.subplot(311)
        plt.title('tBID')
        plt.plot(tspan, bid, label='sim')
        plt.plot(tspan, exp_bid, label='exp')
        plt.legend()
        plt.subplot(312)
        plt.title('cSMAC')
        plt.plot(tspan, csmac, label='sim')
        plt.plot(tspan, exp_smac, label='exp')
        plt.legend()
        plt.subplot(313)
        plt.title('cPARP')
        plt.plot(tspan, cparp, label='sim')
        plt.plot(tspan, exp_parp, label='exp')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return [error, parameters]


if __name__ == '__main__':
    from NS_parallel_calibration import NS
    import multiprocessing as mp
    import time

    starting_score = obj_fun(plot=True)[0]
    print("Starting score = {}".format(starting_score))

    p = mp.Pool(4)
    ns = NS(model, obj_fun, population=10, max_iterations=20,
            target_score=0.01, set_number=10, process_pool=p)
    st = time.time()
    ns.run(verbose=True)
    et = time.time()
    print("Time for MP = {}".format(et-st))
    ns.save()

    ns = NS(model, obj_fun, population=10, max_iterations=20,
            target_score=0.01, set_number=10, process_pool=None)
    st = time.time()
    ns.run(verbose=True)
    et = time.time()
    print("Time for single process = {}".format(et - st))

    # for i in ns.working_set:
    #     obj_fun(i[1], plot=True)
