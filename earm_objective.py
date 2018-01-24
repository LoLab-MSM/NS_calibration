
# Likelihood functions will be problem specific,
# depending on the available data and the given model.

import math


def objective_function(ex_data, simulation):

    for i, each in enumerate(ex_data):
        ex_data[i][0] = ex_data[i][0]/180.0
    for i, each in enumerate(simulation):
        if i != 0:
            simulation[i][0] = simulation[i][0]/180.0

    # setup: find indices for [bidM, bidU, bidT, smacA, smacC, smacM, parpC, parpU]
    names = ['BidM_obs', 'BidU_obs', 'BidT_obs', 'SmacA_obs', 'SmacC_obs', 'SmacM_obs', 'ParpC_obs', 'ParpU_obs']
    indices = [None for _ in range(len(names))]
    for i, each in enumerate(simulation[0]):
        for j, item in enumerate(names):
            if each == item:
                indices[j] = i

    # --------------------------------------------------------------------------------

    # # original objective function
    # # calculate likelihood in the form of negative mean squared error
    # mse = 0.0
    # n = 0.0
    # for i, each in enumerate(ex_data):
    #
    #     if i > 0:
    #         mse += (ex_data[i][1] - simulation[i][indices[0]]/(simulation[i][indices[0]] + simulation[i][indices[1]]
    #  + simulation[i][indices[2]]))**2
    #         mse += (ex_data[i][2] - simulation[i][indices[3]]/(simulation[i][indices[3]] + simulation[i][indices[4]]
    # + simulation[i][indices[5]]))**2
    #         mse += (ex_data[i][3] - simulation[i][indices[6]]/(simulation[i][indices[6]]
    # + simulation[i][indices[7]]))**2
    #         n += 3.0
    #
    # return mse/n

    # --------------------------------------------------------------------------------

    # mean squared error of the minimum distance from the experimental
    # data points to those line segments on either side of the experimental
    # point and created by the simulated points.

    points = []
    values = []
    for i, each in enumerate(simulation):
        if i >= 1:

            b_point = simulation[i][indices[0]] / (simulation[i][indices[0]] + simulation[i][indices[1]] + simulation[i][indices[2]])
            s_point = simulation[i][indices[3]] / (simulation[i][indices[3]] + simulation[i][indices[4]] + simulation[i][indices[5]])
            p_point = simulation[i][indices[6]] / (simulation[i][indices[6]] + simulation[i][indices[7]])
            points.append([simulation[i][0], b_point, s_point, p_point])

            values.append([simulation[i][indices[0]], simulation[i][indices[1]], simulation[i][indices[2]],
                           simulation[i][indices[3]], simulation[i][indices[4]], simulation[i][indices[5]],
                           simulation[i][indices[6]], simulation[i][indices[7]]])

    scores = []

    for i, each in enumerate(ex_data):

        best = [100000000, 100000000, 100000000]

        if i == 0:

            # bid scores
            min_perp_dist_b = - ((points[i+1][0] - points[i][0]) * (points[i][0] - ex_data[i][0])
                                 + (points[i+1][1] - points[i][1]) * (points[i][1] - ex_data[i][1])) \
                              / ((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)

            if 0 <= min_perp_dist_b <= 1:
                score_b = abs((points[i+1][0] - points[i][0]) * (points[i][1] - ex_data[i][1])
                              - (points[i][0] - ex_data[i][0]) * (points[i+1][1] - points[i][1])) \
                          / math.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)

            else:
                dist_1 = math.sqrt((points[i][0] - ex_data[i][0])**2 + (points[i][1] - ex_data[i][1])**2)
                dist_2 = math.sqrt((points[i+1][0] - ex_data[i][0])**2 + (points[i+1][1] - ex_data[i][1])**2)
                score_b = min(dist_1, dist_2)

            if score_b < best[0]:
                best[0] = score_b

            # smac score
            min_perp_dist_s = - ((points[i + 1][0] - points[i][0]) * (points[i][0] - ex_data[i][0])
                                 + (points[i + 1][2] - points[i][2]) * (points[i][2] - ex_data[i][2])) \
                              / ((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][2] - points[i][2]) ** 2)

            if 0 <= min_perp_dist_s <= 1:
                score_s = abs((points[i + 1][0] - points[i][0]) * (points[i][2] - ex_data[i][2])
                              - (points[i][0] - ex_data[i][0]) * (points[i + 1][2] - points[i][2])) \
                          / math.sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][2] - points[i][2]) ** 2)

            else:
                dist_1 = math.sqrt((points[i][0] - ex_data[i][0]) ** 2 + (points[i][2] - ex_data[i][2]) ** 2)
                dist_2 = math.sqrt((points[i + 1][0] - ex_data[i][0]) ** 2 + (points[i + 1][2] - ex_data[i][2]) ** 2)
                score_s = min(dist_1, dist_2)

            if score_s < best[1]:
                best[1] = score_s

            # parp score
            min_perp_dist_p = - ((points[i + 1][0] - points[i][0]) * (points[i][0] - ex_data[i][0])
                                 + (points[i + 1][3] - points[i][3]) * (points[i][3] - ex_data[i][3])) \
                              / ((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][3] - points[i][3]) ** 2)

            if 0 <= min_perp_dist_p <= 1:
                score_p = abs((points[i + 1][0] - points[i][0]) * (points[i][3] - ex_data[i][3])
                              - (points[i][0] - ex_data[i][0]) * (points[i + 1][3] - points[i][3])) \
                          / math.sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][3] - points[i][3]) ** 2)

            else:
                dist_1 = math.sqrt((points[i][0] - ex_data[i][0]) ** 2 + (points[i][3] - ex_data[i][3]) ** 2)
                dist_2 = math.sqrt((points[i + 1][0] - ex_data[i][0]) ** 2 + (points[i + 1][3] - ex_data[i][3]) ** 2)
                score_p = min(dist_1, dist_2)

            if score_p < best[2]:
                best[2] = score_p

        if 0 < i < len(ex_data)-1:

            for j in [i-1, i]:

                # bid scores
                min_perp_dist_b = - ((points[j+1][0] - points[j][0]) * (points[j][0] - ex_data[i][0])
                                     + (points[j+1][1] - points[j][1]) * (points[j][1] - ex_data[i][1])) \
                                  / ((points[j+1][0] - points[j][0])**2 + (points[j+1][1] - points[j][1])**2)

                if 0 <= min_perp_dist_b <= 1:
                    score_b = abs((points[j+1][0] - points[j][0]) * (points[j][1] - ex_data[i][1])
                                  - (points[j][0] - ex_data[i][0]) * (points[j+1][1] - points[j][1])) \
                              / math.sqrt((points[j+1][0] - points[j][0])**2 + (points[j+1][1] - points[j][1])**2)

                else:
                    dist_1 = math.sqrt((points[j][0] - ex_data[i][0])**2 + (points[j][1] - ex_data[i][1])**2)
                    dist_2 = math.sqrt((points[j+1][0] - ex_data[i][0])**2 + (points[j+1][1] - ex_data[i][1])**2)
                    score_b = min(dist_1, dist_2)

                if score_b < best[0]:
                    best[0] = score_b

                # smac score
                min_perp_dist_s = - ((points[j + 1][0] - points[j][0]) * (points[j][0] - ex_data[i][0])
                                     + (points[j + 1][2] - points[j][2]) * (points[j][2] - ex_data[i][2])) \
                                  / ((points[j + 1][0] - points[j][0]) ** 2 + (points[j + 1][2] - points[j][2]) ** 2)

                if 0 <= min_perp_dist_s <= 1:
                    score_s = abs((points[j + 1][0] - points[j][0]) * (points[j][2] - ex_data[i][2])
                                  - (points[j][0] - ex_data[i][0]) * (points[j + 1][2] - points[j][2])) \
                              / math.sqrt((points[j + 1][0] - points[j][0]) ** 2 + (points[j + 1][2] - points[j][2]) ** 2)

                else:
                    dist_1 = math.sqrt((points[j][0] - ex_data[i][0]) ** 2 + (points[j][2] - ex_data[i][2]) ** 2)
                    dist_2 = math.sqrt((points[j + 1][0] - ex_data[i][0]) ** 2 + (points[j + 1][2] - ex_data[i][2]) ** 2)
                    score_s = min(dist_1, dist_2)

                if score_s < best[1]:
                    best[1] = score_s

                # parp score
                min_perp_dist_p = - ((points[j + 1][0] - points[j][0]) * (points[j][0] - ex_data[i][0])
                                     + (points[j + 1][3] - points[j][3]) * (points[j][3] - ex_data[i][3])) \
                                  / ((points[j + 1][0] - points[j][0]) ** 2 + (points[j + 1][3] - points[j][3]) ** 2)

                if 0 <= min_perp_dist_p <= 1:
                    score_p = abs((points[j + 1][0] - points[j][0]) * (points[j][3] - ex_data[i][3])
                                  - (points[j][0] - ex_data[i][0]) * (points[j + 1][3] - points[j][3])) \
                              / math.sqrt((points[j + 1][0] - points[j][0]) ** 2 + (points[j + 1][3] - points[j][3]) ** 2)

                else:
                    dist_1 = math.sqrt((points[j][0] - ex_data[i][0]) ** 2 + (points[j][3] - ex_data[i][3]) ** 2)
                    dist_2 = math.sqrt((points[j + 1][0] - ex_data[i][0]) ** 2 + (points[j + 1][3] - ex_data[i][3]) ** 2)
                    score_p = min(dist_1, dist_2)

                if score_p < best[2]:
                    best[2] = score_p

        if i == len(ex_data)-1:

            # bid scores
            min_perp_dist_b = - ((points[i-1+1][0] - points[i-1][0]) * (points[i-1][0] - ex_data[i][0])
                                 + (points[i-1+1][1] - points[i-1][1]) * (points[i-1][1] - ex_data[i][1])) \
                              / ((points[i-1+1][0] - points[i-1][0])**2 + (points[i-1+1][1] - points[i-1][1])**2)

            if 0 <= min_perp_dist_b <= 1:
                score_b = abs((points[i-1+1][0] - points[i-1][0]) * (points[i-1][1] - ex_data[i][1])
                              - (points[i-1][0] - ex_data[i][0]) * (points[i-1+1][1] - points[i-1][1])) \
                          / math.sqrt((points[i-1+1][0] - points[i-1][0])**2 + (points[i-1+1][1] - points[i-1][1])**2)

            else:
                dist_1 = math.sqrt((points[i-1][0] - ex_data[i][0])**2 + (points[i-1][1] - ex_data[i][1])**2)
                dist_2 = math.sqrt((points[i-1+1][0] - ex_data[i][0])**2 + (points[i-1+1][1] - ex_data[i][1])**2)
                score_b = min(dist_1, dist_2)

            if score_b < best[0]:
                best[0] = score_b

            # smac score
            min_perp_dist_s = - ((points[i-1 + 1][0] - points[i-1][0]) * (points[i-1][0] - ex_data[i][0])
                                 + (points[i-1 + 1][2] - points[i-1][2]) * (points[i-1][2] - ex_data[i][2])) \
                              / ((points[i-1 + 1][0] - points[i-1][0]) ** 2 + (points[i-1 + 1][2] - points[i-1][2]) ** 2)

            if 0 <= min_perp_dist_s <= 1:
                score_s = abs((points[i-1 + 1][0] - points[i-1][0]) * (points[i-1][2] - ex_data[i][2])
                              - (points[i-1][0] - ex_data[i][0]) * (points[i-1 + 1][2] - points[i-1][2])) \
                          / math.sqrt((points[i-1 + 1][0] - points[i-1][0]) ** 2 + (points[i-1 + 1][2] - points[i-1][2]) ** 2)

            else:
                dist_1 = math.sqrt((points[i-1][0] - ex_data[i][0]) ** 2 + (points[i-1][2] - ex_data[i][2]) ** 2)
                dist_2 = math.sqrt((points[i-1 + 1][0] - ex_data[i][0]) ** 2 + (points[i-1 + 1][2] - ex_data[i][2]) ** 2)
                score_s = min(dist_1, dist_2)

            if score_s < best[1]:
                best[1] = score_s

            # parp score
            min_perp_dist_p = - ((points[i-1 + 1][0] - points[i-1][0]) * (points[i-1][0] - ex_data[i][0])
                                 + (points[i-1 + 1][3] - points[i-1][3]) * (points[i-1][3] - ex_data[i][3])) \
                    / ((points[i-1 + 1][0] - points[i-1][0]) ** 2 + (points[i-1 + 1][3] - points[i-1][3]) ** 2)

            if 0 <= min_perp_dist_p <= 1:
                score_p = abs((points[i-1 + 1][0] - points[i-1][0]) * (points[i-1][3] - ex_data[i][3])
                              - (points[i-1][0] - ex_data[i][0]) * (points[i-1 + 1][3] - points[i-1][3])) \
                          / math.sqrt((points[i-1 + 1][0] - points[i-1][0]) ** 2 + (points[i-1 + 1][3] - points[i-1][3]) ** 2)

            else:
                dist_1 = math.sqrt((points[i-1][0] - ex_data[i][0]) ** 2 + (points[i-1][3] - ex_data[i][3]) ** 2)
                dist_2 = math.sqrt((points[i-1 + 1][0] - ex_data[i][0]) ** 2 + (points[i-1 + 1][3] - ex_data[i][3]) ** 2)
                score_p = min(dist_1, dist_2)

            if score_p < best[2]:
                best[2] = score_p

        scores.append(best)

    se = 0
    for each in scores:
        for item in each:
            se += item**2
    mse = se / (len(scores) * len(scores[0]))

    return mse
