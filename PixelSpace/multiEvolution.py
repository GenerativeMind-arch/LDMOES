
#from collections import defaultdict
from collections import defaultdict
import numpy as np
import random
import matplotlib.pyplot as plt
import math



class Individual(object):
    def __init__(self):

        self.solution = None
        self.objective = defaultdict()
        self.n = 0
        self.rank = 0
        self.S = []
        self.distance = 0


    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min


    def calculate_objective(self, objective_fun):

        self.objective = objective_fun(self.solution)


    def __lt__(self, other):

        v1 = list(self.objective.values())
        v2 = list(other.objective.values())

        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return 0
        return 1

def main():

    generations = 50
    popnum = 50
    eta = 1



    poplength = 2
    bound_min = 0
    bound_max = 1
    objective_fun = LOSSCOST


    P = []
    for i in range(popnum):

        P.append(Individual())
        P[i].solution = np.random.rand(poplength) * (bound_max - bound_min) + bound_min
        P[i].bound_process(bound_min, bound_max)
        P[i].calculate_objective(objective_fun)

    fast_non_dominated_sort(P)


    Q = make_new_pop(P, eta, bound_min, bound_max, objective_fun)

    P_t = P
    Q_t = Q


    for gen_cur in range(generations):
        R_t = P_t + Q_t
        F = fast_non_dominated_sort(R_t)

        P_n = []
        i = 1
        while len(P_n) + len(F[i]) < popnum:
            crowding_distance_assignment(F[i])
            P_n = P_n + F[i]
            i = i + 1
        F[i].sort(key=lambda x: x.distance)
        P_n = P_n + F[i][:popnum - len(P_n)]
        Q_n = make_new_pop(P_n, eta, bound_min, bound_max, objective_fun)

        P_t = P_n
        Q_t = Q_n

        plt.clf()
        plt.title('current generation_P_t:' + str(gen_cur + 1))
        plot_P(P_t)
        plt.pause(0.1)



    plt.show()

    return 0


def fast_non_dominated_sort(P):

    F = defaultdict(list)

    for p in P:
        p.S = []
        p.n = 0

        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p

        if p.n == 0:
            p.rank = 1
            F[1].append(p)

    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q

    return F



def make_new_pop(P, eta, bound_min, bound_max, objective_fun):

    popnum = len(P)
    Q = []
    # binary tournament selection
    for i in range(int(popnum / 2)):

        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])

        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.solution == parent2.solution).all():

            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])

        Two_offspring = crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun)


        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q



def crowding_distance_assignment(L):

    l = len(L)

    for i in range(l):
        L[i].distance = 0

    for m in L[0].objective.keys():
        L.sort(key=lambda x: x.objective[m])
        L[0].distance = float('inf')
        L[l - 1].distance = float('inf')  # so that boundary points are always selected


        f_max = L[l - 1].objective[m]
        f_min = L[0].objective[m]


        try:
            for i in range(1, l - 1):  # for all other points
                L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
        except Exception:
            print(str(m) + "max" + str(f_max) + "min" + str(f_min))


def binary_tournament(ind1, ind2):

    if ind1.rank != ind2.rank:
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:

        return ind1 if ind1.distance > ind2.distance else ind2
    else:
        return ind1



def crossover_mutation(parent1, parent2, eta, bound_min, bound_max, objective_fun):

    poplength = len(parent1.solution)

    offspring1 = Individual()
    offspring2 = Individual()

    offspring1.solution = np.empty(poplength)
    offspring2.solution = np.empty(poplength)


    for i in range(poplength):
        rand = random.random()

        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])
        offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])


    for i in range(poplength):
        mu = random.random()

        delta = 2 * mu ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
        offspring1.solution[i] = offspring1.solution[i] + delta


    offspring1.bound_process(bound_min, bound_max)
    offspring2.bound_process(bound_min, bound_max)

    offspring1.calculate_objective(objective_fun)
    offspring2.calculate_objective(objective_fun)

    return [offspring1, offspring2]


def LOSSCOST(x):
    f = defaultdict(float)
    poplength = len(x)

    f[1] = 0
    f[2] = 0

    for i in range(poplength - 1):
        f[1] = f[1] + (-10) * math.exp((-0.2) * (x[i] ** 2 + x[i + 1] ** 2) ** 0.5)

    for i in range(poplength):
        f[2] = f[2] + abs(x[i]) ** 0.8 + 5 * math.sin(x[i] ** 3)

    return f


def plot_P(P):

    X = []
    Y = []
    for ind in P:
        X.append(ind.objective[1])
        Y.append(ind.objective[2])

    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.scatter(X, Y)



if __name__ == '__main__':
    main()
