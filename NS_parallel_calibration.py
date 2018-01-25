
# GNU General Public License software (C) Michael Kochen 2017.
# Based on the algorithm published by Sivia and Skilling 2006,
# and translated to Python by Issac Trotts in 2007.
# http://www.inference.org.uk/bayesys/

# A nested sampling algorithm to calibrate PySB models.

# Kernel density estimation is used as a method
# for new point selection.

# Prior (parameter) ranges depend on the type of reaction
# following types are considered here:

#   * 1st order forward (1kf)
#   * 2nd order forward (2kf)
#   * 1st order reverse (1kr)
#   * catalysis reactions (1kc)

# Parameter names ending with'_0' (initial values) are exempted.

# The termination criteria is based on a maximum score
# that is obtained for a given number of models. For
# example, 5 models with a score of 0.01 or less (lower
# is better)

# The output is a family of parameter sets that is equal
# to the size of the nested sampling population and with
# a given number of them less than the provided score criteria.
# Additional statistics (number of iterations, etc.) are also given.

from pysb.integrate import Solver
import numpy as np
from math import *
from os.path import isfile
from copy import deepcopy
import pathos.pools as pp
import itertools
from pathos.helpers import mp as multiprocess

class NS:

    def __init__(self, model, objective_function, process_data, data, population,
                 max_iterations, target_score, set_number, proc_number):

        self.model = model
        self.objective_function = objective_function
        self.data = data
        self.model_solver = None
        self.prior_1kf = [-4, 0]
        self.prior_2kf = [-10, -2]  # [-8, -4]
        self.prior_1kr = [-5, 1]    # [-4, 0]
        self.prior_1kc = [-4, 3]    # [-1, 3]
        self.iterations = max_iterations
        self.scalar = 10.0
        self.reduction = 0.9
        self.scalar_reductions = 0
        self.scalar_limit = .0001
        self.iteration = 0
        self.simulations = 0
        self.useless = 10
        self.N = population
        self.target_score = target_score
        self.set_number = set_number - 1
        self.time = []
        self.working_set = None
        self.params = []
        self.stop = 'None'
        self.process_num = proc_number
        self.dummy = [x for x in range(self.process_num)]
        self.processed_data = process_data(self.data)
        self._initiate_log()
        self._nested_sampling_KDE()
        self._output()

    def _output(self):

        summary_index = 0
        while isfile(self.model.name + '_results_' + str(summary_index) + '.txt'):
            summary_index += 1
        summary_object = self.model.name + '_results_' + str(summary_index) + '.txt'

        summary = open(summary_object, 'w')
        summary.write(self.model.name + '_results' + '\n')
        summary.write('parameters: ' + str(len(self.params)) + '\n')
        summary.write('iterations: ' + str(self.iteration) + '\n')
        summary.write('simulations: ' + str(self.simulations) + '\n')
        summary.write('scalar: ' + str(self.scalar) + '\n')
        summary.write('scalar reduction: ' + str(self.scalar_reductions) + '\n')
        summary.write('stop criteria: ' + self.stop + '\n\n')

        summary.write('parameter sets\n')
        self.working_set.sort()
        for i, each in enumerate(self.working_set):
            summary.write(str(i+1) + ' score: ' + str(each[0]) + '\n')
        summary.write('\n')
        for i, each in enumerate(self.working_set):
            summary.write(''.join(str(each[1])[1:-1]) + '\n')
        summary.write('\n')

        summary.close()


        # # randomly choose points from parameter space
        # coords = []
        # np.random.seed()
        # for each in self.params:
        #     if isinstance(each, list):
        #         coords.append(10 ** (np.random.uniform(each[0], each[1])))
        #     else:
        #         coords.append(each)
        #
        # objective = self._compute_objective(deepcopy(coords))
        # # if not isnan(objective):
        #
        # print len(self.working_set), objective
        #
        # return [objective, coords]

    def _initiate_log(self):

        def parallel_initialize(num_tasks):

            count = 0
            pop_list = []
            while count < num_tasks:
                coords = []
                np.random.seed()
                for item in self.params:
                    if isinstance(item, list):
                        coords.append(10 ** (np.random.uniform(item[0], item[1])))
                    else:
                        coords.append(item)
                objective = self._compute_objective(deepcopy(coords))
                pop_list.append([objective, coords])
                count += 1

            return pop_list

        # retrieve time points from data
        for each in self.processed_data:
            self.time.append(float(each[0]))

        # set parameter values or prior ranges (log-space)
        for each in self.model.parameters:

            if each.name[-2:] == '_0':
                self.params.append(each.value)
            if each.name[-3:] == '1kf':
                self.params.append(self.prior_1kf)
            if each.name[-3:] == '2kf':
                self.params.append(self.prior_2kf)
            if each.name[-3:] == '1kr':
                self.params.append(self.prior_1kr)
            if each.name[-3:] == '1kc':
                self.params.append(self.prior_1kc)

        # create solver object
        self.model_solver = Solver(self.model, self.time, integrator='lsoda', integrator_options={'atol': 1e-12, 'rtol': 1e-12, 'mxstep': 20000})

        # distribute the working set population among processes
        tasks_per_process = [int(ceil((float(self.N))/float(self.process_num))) for _ in range(self.process_num)]
        sum_tpp = sum(tasks_per_process)
        while sum_tpp < self.N:
            sum_tpp += 1
            tasks_per_process[-1] += 1

        # construct, in parallel, the working population of N parameter sets
        p = pp.ProcessPool(self.process_num)
        self.working_set = list(itertools.chain.from_iterable(p.map(parallel_initialize, tasks_per_process)))
        self.working_set.sort()

    def _compute_objective(self, point):

        # simulate a point
        self.model_solver.run(point)

        # construct simulation trajectories
        sim_trajectories = [['time']]
        for each in self.model.observables:
            sim_trajectories[0].append(each.name)

        for i, each in enumerate(self.model_solver.yobs):
            sim_trajectories.append([self.time[i]] + list(each))

        # calculate the cost
        cost = self.objective_function(self.processed_data, sim_trajectories)
        if isinstance(cost, float):
            return cost
        else:
            return False

    # def _parallel_nested_sampling(self, dummy):
    #
    #     # sample from the prior
    #     test_point = self._KDE_sample_log(dummy)
    #
    #     # calculate objective
    #     test_point_objective = self._compute_objective(test_point)
    #
    #     return[test_point_objective, test_point]

    def _nested_sampling_KDE(self):

        useless_samples = 0
        iteration = 1
        score_criteria = self.working_set[self.set_number][0]

        def parallel_nested_sampling(dummy):

            # sample from the prior
            test_point = self._KDE_sample_log(dummy)

            # calculate objective
            test_point_objective = self._compute_objective(test_point)

            return [test_point_objective, test_point]

        # test new points until termination criteria are met

        p = pp.ProcessPool(self.process_num)
        while iteration <= self.iterations and self.scalar > self.scalar_limit and score_criteria > self.target_score:

            print iteration, self.scalar, score_criteria
            self.simulations += self.process_num
            self.iteration = iteration

            if useless_samples >= self.useless:
                self.scalar *= self.reduction
                useless_samples = 0
                self.scalar_reductions += 1

            provisional_points = p.map(parallel_nested_sampling, self.dummy)
            new_points = []
            for each in provisional_points:
                if not isnan(each[0]):
                    if each[0] < self.working_set[-1][0]:
                        new_points.append(each)
                        iteration += 1
                    else:
                        useless_samples += 1
                else:
                    useless_samples += 1

            score_criteria = self.working_set[self.set_number][0]

            self.working_set.extend(new_points)
            self.working_set.sort()
            self.working_set = self.working_set[:self.N]

        if iteration > self.iterations:
            self.stop = 'iterations'
        if self.scalar <= self.scalar_limit:
            self.stop = 'scalar limit'
        if score_criteria <= self.target_score:
            self.stop = 'target score reached'

        self.working_set.sort()

    def _KDE_sample_log(self, dummy):

        # select data point
        np.random.seed()
        data_point = np.random.randint(0, len(self.working_set)-1)
        coordinates = self.working_set[data_point][1]

        # select parameter values individually from normal with respect to prior boundary
        new_point = []
        for i, each in enumerate(coordinates):
            if isinstance(self.params[i], float):
                new_point.append(self.params[i])
            else:
                accept = False
                log_coord = None

                while not accept:
                    log_coord = np.random.normal(log10(each), self.scalar)
                    if self.params[i][0] <= log_coord <= self.params[i][1]:
                        accept = True
                new_point.append(10**log_coord)

        return new_point
