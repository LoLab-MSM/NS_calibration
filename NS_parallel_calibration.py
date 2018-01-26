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

from math import *
from os.path import isfile

import numpy as np


class NS(object):

    def __init__(self, model, objective_function, population, max_iterations,
                 target_score, set_number, process_pool=None):

        self.model = model
        self.objective_function = objective_function
        self.prior_1kf = [-4, 0]
        self.prior_2kf = [-10, -2]  # [-8, -4]
        self.prior_1kr = [-5, 1]  # [-4, 0]
        self.prior_1kc = [-4, 3]  # [-1, 3]
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

        self.working_set = None
        self.params = []
        self.stop = 'None'
        if process_pool is not None:
            self.map = process_pool.map
        else:
            self.map = map
        self._initiate_log()

    def save(self):

        summary_index = 0
        while isfile(
                self.model.name + '_results_' + str(summary_index) + '.txt'):
            summary_index += 1
        summary_object = self.model.name + '_results_' + str(
            summary_index) + '.txt'

        summary = open(summary_object, 'w')
        summary.write(self.model.name + '_results' + '\n')
        summary.write('parameters: ' + str(len(self.params)) + '\n')
        summary.write('iterations: ' + str(self.iteration) + '\n')
        summary.write('simulations: ' + str(self.simulations) + '\n')
        summary.write('scalar: ' + str(self.scalar) + '\n')
        summary.write(
            'scalar reduction: ' + str(self.scalar_reductions) + '\n')
        summary.write('stop criteria: ' + self.stop + '\n\n')

        summary.write('parameter sets\n')
        self.working_set.sort()
        for i, each in enumerate(self.working_set):
            summary.write(str(i + 1) + ' score: ' + str(each[0]) + '\n')
        summary.write('\n')
        for i, each in enumerate(self.working_set):
            summary.write(''.join(str(each[1])[1:-1]) + '\n')
        summary.write('\n')

        summary.close()

    def _initiate_log(self):
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

        self.working_set = []
        first_points = []
        for i in range(self.N):
            coords = []
            np.random.seed()
            for item in self.params:
                if isinstance(item, list):
                    coords.append(
                        10 ** (np.random.uniform(item[0], item[1])))
                else:
                    coords.append(item)
            first_points.append(coords)
        self.working_set = self.map(self.objective_function, first_points)
        self.working_set.sort()

    def run(self, verbose=False):

        useless_samples = 0
        iteration = 1
        score_criteria = self.working_set[self.set_number][0]

        # test new points until termination criteria are met
        while iteration <= self.iterations and self.scalar > self.scalar_limit and score_criteria > self.target_score:
            if verbose:
                print(iteration)
            self.iteration = iteration

            if useless_samples >= self.useless:
                self.scalar *= self.reduction
                useless_samples = 0
                self.scalar_reductions += 1

            # generating random samples here prevents needing to change np seed
            new_points = [self._KDE_sample_log() for _ in range(self.N)]
            provisional_points = self.map(self.objective_function, new_points)
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

    def _KDE_sample_log(self):

        # select data point
        np.random.seed()
        data_point = np.random.randint(0, len(self.working_set) - 1)
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
                new_point.append(10 ** log_coord)

        return new_point
