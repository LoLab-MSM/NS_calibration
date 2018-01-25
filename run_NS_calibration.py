
from NS_calibration import NS
from earm_model import model
from earm_objective import objective_function
from process_earm_data import process_data


NS(model,
    objective_function,
    process_data,
    'earm_data.csv',
    1000,                         # population size
    10000,                       # max number of iterations
    0.01,                          # target score (lower is better)
    10)                           # number of parameter sets guaranteed to meet target score

