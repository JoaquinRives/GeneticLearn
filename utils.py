import numpy as np




"""
#######################################################################################################################

# Selection methods

#######################################################################################################################
"""

population = [{'fitness': 2023}, {'fitness': 100}, {'fitness': 70230}, {'fitness': 561}, {'fitness': 100},
              {'fitness': 2560}, {'fitness': 55}, {'fitness': 1000}, {'fitness': 1}, {'fitness': 100}, {'fitness': 1},
              {'fitness': 100}, {'fitness': 200}, {'fitness': 5500000}, {'fitness': 100}, {'fitness': 100},
              {'fitness': 220}, {'fitness': 553}]


# Roulette wheel
def roulette_wheel(population, n=1):
    max = sum(individual['fitness'] for individual in population)
    # TODO max
    # TODO max
    # TODO max
    # TODO max
    points = sorted([np.random.uniform(0, max) for i in range(n)])

    selected = []
    for P in points:
        current_fitness = 0
        for individual in population:
            current_fitness += individual['fitness']
            if current_fitness >= P:
                selected.append(individual)
                break
    return selected


# Rank selection
def rank_selection(population, n=1):
    population = sorted(population, key=lambda x: x['fitness'])

    probabilities = []
    for i in range(len(population) + 1):
        if i == 0:
            probabilities.append(0)
        else:
            next_proba = (i + 1) / len(population)
            probabilities.append(probabilities[i - 1] + (next_proba / len(population)))

    selected_test = []
    selected = []

    while len(selected) < n:
        random_number = np.random.rand()
        for j in range(len(population)):
            if random_number < probabilities[j + 1]:
                selected.append(population[j])
                break

    return selected[0] if len(selected) == 1 else selected


def rank(eval_df, n=1):
    n = int(n)
    eval_df = eval_df.copy()
    eval_df.sort_values(by='fitness', axis=0, inplace=True)

    pop_length = eval_df.shape[0]

    probabilities = []
    for i in range(pop_length + 1):
        if i == 0:
            probabilities.append(0)
        else:
            next_proba = (i + 1) / pop_length
            probabilities.append(probabilities[i - 1] + (next_proba / pop_length))

    selected = []
    while len(selected) < n:
        random_number = np.random.rand()
        for j in range(pop_length):
            if random_number < probabilities[j + 1]:
                selected.append(eval_df.loc[j, 'pop_index'])
                break

    return selected[0] if len(selected) == 1 else selected


# Stochastic universal sampling
# TODO
# def stochastic_universal_sampling(population, n=1):
#
#     population = sorted(population, key=lambda x: x['fitness'])
#
#     F = sum(individual['fitness'] for individual in population)  # total fitness of Population
#     N = n+1  # number of individuals to select
#     P = F / N  # distance between the pointers
#     start = np.random.uniform(0, P)  # random number between 0 and P
#     points = [(start + (i*P)) for i in range(0, N-1)]
#
#     selected = []
#     for P in points:
#         current_fitness = 0
#         for individual in population:
#             current_fitness += individual['fitness']
#             if current_fitness >= P:
#                 selected.append(individual)
#                 break
#
#     return selected[0] if len(selected) == 1 else selected


# roulette wheel
pop_RWS = roulette_wheel(population, n=1)
pop_RWS = [i['fitness'] for i in pop_RWS]
freq_RWS = np.unique(np.array(pop_RWS), return_counts=True)
print(len(freq_RWS[0]))


pop_RWS = []
for j in range(10000):
    pop_RWS.append(roulette_wheel(population, n=1))

pop_RWS = [i[0]['fitness'] for i in pop_RWS]
freq_RWS = np.unique(np.array(pop_RWS), return_counts=True)
print(len(freq_RWS[0]))


#
# # rank
# pop_rank = []
# pop_SUS = rank_selection(population, n=10000)
# pop_SUS = [i['fitness'] for i in pop_SUS]
# freq_rank = np.unique(np.array(pop_SUS), return_counts=True)
# print(len(freq_rank[0]))

# # stocastic
# pop_SUS = []
# for i in range(1000):
#     pop_SUS += stochastic_universal_sampling(population, n=10) # TODO
#
# pop_SUS = [i['fitness'] for i in pop_SUS]
# freq_SUS = np.unique(np.array(pop_SUS), return_counts=True)
# print(len(freq_SUS[0]))


"""
#######################################################################################################################

# Gray encoding

#######################################################################################################################
"""


def binarytoGray(binary):
    gray = ""
    gray += binary[0]
    for i in range(1, len(binary)):
        gray += '0' if (binary[i - 1] == binary[i]) else '1'
    return gray


def graytoBinary(gray):
    binary = []
    binary += gray[0]
    for i in range(1, len(gray)):
        if gray[i] == '0':
            binary += binary[i - 1]
        else:
            binary += '1' if (binary[i - 1] == '0') else '0'
    return np.array(binary)


# Python 3 Program to convert given decimal number into decimal equivalent of its gray code form
def decimal2grayCode(n):
    return n ^ (n >> 1)


# Python3 Program to convert given decimal number of gray code into its inverse  in decimal form
def grayCode2decimal(n):
    inv = 0
    while n:
        inv = inv ^ n
        n = n >> 1
    return inv

































