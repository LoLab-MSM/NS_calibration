
from NS_calibration import NS
from model_804 import model
from momp_objective import objective_function
from process_momp_data import process_data


NS(model,
    objective_function,
    process_data,
    'momp_data.csv',
    10000,                         # population size
    100000,                       # max number of iterations
    0.0001,                          # target score (lower is better)
    100)                           # number of parameter sets guaranteed to meet target score

