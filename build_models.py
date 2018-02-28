
# This is a utility to to take a list of parameter sets
# and fill out the original model with them.
# Arguments are the model name (str) and the top number of models desired (int).


# def build_models(model_name, num_models):

model_name = 'model_804.py'
num_models = 10

parameter_count = 0
mod = open(model_name, 'r')
for line in mod:
    if 'Parameter' in line and 'pysb' not in line:
        parameter_count += 1

params = open('model_804_results_0.txt', 'r')

ps = False
i = 0
for p_line in params:
    if ps:
        param_list = p_line[:-1].split()

        if len(param_list) == parameter_count:
            mod = open(model_name, 'r')
            new_model_name = model_name[:-3] + '_' + str(i) + '.py'
            new_mod = open(new_model_name, 'w')

            for m_line in mod:
                if 'Parameter' in m_line and 'pysb' not in m_line:
                    new_param_line = m_line.split(',')[0] + ', ' + param_list.pop(0)[:-1] + ')'
                    new_mod.write(new_param_line + '\n')
                else:
                    new_mod.write(m_line)
            mod.close()
            i += 1

    if i >= int(num_models):
        break

    if p_line.strip() == 'parameter sets':
        ps = True

