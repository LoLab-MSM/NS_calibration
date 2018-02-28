
import csv


def process_data(data):

    # Here, the first column is time and
    # the first row are observable labels.
    # This function will change depending on the
    # data and the objective function.

    # todo: combine data processing and objective function

    data_object = []
    with open(data) as data_file:
        reader = csv.reader(data_file)
        line = list(reader)
        for each in line[1:]:
            data_object.append(each)

    for i, each in enumerate(data_object):
        for j, item in enumerate(each):
            data_object[i][j] = float(data_object[i][j])
    for i, each in enumerate(data_object):
        data_object[i][0] = data_object[i][0]

    return data_object
