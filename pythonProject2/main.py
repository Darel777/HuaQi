import os
from flask import Flask, jsonify, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import random
import werkzeug
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools

IS_SERVERLESS = bool(os.environ.get('SERVERLESS'))

app = Flask(__name__)
CORS(app, supports_credentias=True)


@app.route("/")
def index():
    return render_template('index.html')

def compute_A(m, e_a: np.matrix, e_b: np.matrix) -> np.matrix:
    A = np.block([
        [np.eye(m + 1), np.zeros(shape=(m + 1, m)), np.zeros(shape=(m + 1, m))],
        [np.zeros(shape=(m, m + 1)), e_a, np.zeros(shape=(m, m))],
        [np.zeros(shape=(m, m + 1)), np.zeros(shape=(m, m)), e_b]
    ])
    return A


def compute_B(m, e_a: np.matrix, e_b: np.matrix, Q_a: np.matrix, Q_b: np.matrix, Lambda: np.matrix) -> np.matrix:
    delta = np.block([
        [np.eye(m)],
        [-np.eye(m)]
    ])
    delta_a = np.block([
        [np.eye(m)],
        [np.zeros(shape=(m, m))]
    ])
    delta_b = np.block([
        [np.zeros(shape=(m, m))],
        [-np.eye(m)]
    ])
    kappa_a = 2 * np.dot(Q_a, np.transpose(delta_a)) - np.dot(Lambda, np.transpose(delta))
    kappa_b = 2 * np.dot(Q_b, np.transpose(delta_b)) + np.dot(Lambda, np.transpose(delta))
    B = np.block([
        [np.zeros(shape=(1, 2 * m))],
        [-np.transpose(delta)],
        [np.dot(e_a, kappa_a)],
        [np.dot(e_b, kappa_b)]
    ])
    return B


def compute_Bbar(A: np.matrix, B: np.matrix, N: np.matrix, m) -> np.matrix:
    B_bar = np.random.random(size=((N + 1) * (3 * m + 1), (N + 1) * (2 * m)))
    Row = B_bar.shape[0]
    Col = B_bar.shape[1]
    row_A = A.shape[0]
    col_A = A.shape[1]
    row_B = B.shape[0]
    col_B = B.shape[1]

    for i in range(0, N + 1):  # i row block
        if i == 0:
            B_bar[0:row_B, :] = np.zeros(shape=(row_B, Col))
        else:
            posy = i * row_B
            for j in range(0, N + 1):
                posx = j * col_B
                if i - 1 - j >= 0:
                    tmp = np.eye(row_A, col_A)
                    for k in range(0, i - 1 - j):
                        tmp = np.dot(tmp, A)
                    B_bar[posy:posy + row_B, posx:posx + col_B] = np.dot(tmp, B)
                else:
                    B_bar[posy:posy + row_B, posx:posx + col_B] = np.zeros(shape=(row_B, col_B))
    return B_bar


def compute_D(B_bar: np.matrix, N_bar: np.matrix, Q_bar: np.matrix) -> np.matrix:
    D = 0.5 * (np.dot(np.transpose(B_bar), N_bar) + np.dot(np.transpose(N_bar), B_bar)) + Q_bar
    return D


def getQBar(qList, N):
    def getQk(qList):
        #     def f1(x, k):
        #         return 1 / (2 * np.power(x, k))
        #     qList = f1(qList, k)
        return np.diag(qList)

    def getQ(Qa, Qb):
        return np.diag(np.append(np.diag(Qa), np.diag(Qb)))

    Q = getQ(getQk(qList), getQk(qList))

    diag = np.diag(Q)
    for i in range(N):
        diag = np.append(diag, np.diag(Q))
    return np.diag(diag)


def getNBar(s, Lambda, m, N):
    def getN(s, Lambda, m):
        def getL(m):
            return np.block([
                [np.matlib.identity(m)],
                [np.matlib.identity(m)]
            ])

        def getDelta(m):
            return np.block([
                [np.matlib.identity(m)],
                [- np.matlib.identity(m)]
            ])

        return np.block([
            [np.dot(0.5, np.dot(np.matrix(s), getL(m).T))],
            [- np.dot(Lambda, getDelta(m).T)],
            [np.matlib.identity(2 * m)]
        ])

    N_M = getN(s, Lambda, m)
    ret = N_M
    for i in range(N):
        ret = np.block([
            [ret, np.zeros((ret.shape[0], N_M.shape[1]))],
            [np.zeros((N_M.shape[0], ret.shape[1])), N_M]
        ])
    return ret


# simple test to check row and column
# getQBar([1,2], 1, 2, 2)
# simple test to check row and column
# getNBar([1,2], np.matrix('2,3;1,2'), 2, 1)

## -------------------------------------------------


def calculate_cov_u(prices):
    """
    param:
    prices: the prices of k stocks(n days)
    """
    k = prices.shape[0]
    n = prices.shape[1]
    u = []
    for i in range(k):
        for j in range(n):
            if j == 0:
                u.append(0)
            else:
                u.append(prices[i, j] - prices[i, j - 1])
    return np.cov(u)


def calculate_delta_bar(m, N):
    """
    param:
    m : length of Identity Matrix
    """
    m_i = np.eye(m)
    delta = np.vstack((m_i, -m_i))
    delta_bar = delta
    for _ in range(N):
        delta_bar = np.block([
            [delta_bar, np.zeros((delta_bar.shape[0], m))],
            [np.zeros((2 * m, delta_bar.shape[1])), delta]
        ])
    return delta_bar


## user input

N = 60  # 60s / 1s
m = 5  # 5 stocks
alpha = 0.01
z_0 = np.array([12345, 23456, 34567, 45678, 56789]).reshape(-1, 1)
y_0 = np.array(
    [1, 12345, 23456, 34567, 45678, 56789, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]).reshape(-1, 1)

@app.route('/getdata3', methods=['GET', 'POST'])
def test3():
    return getdata()
@app.route('/getdata', methods=['GET', 'POST'])
def test():
        data = request.values.to_dict()
        global N
        N = int(int(data['T'])/float(data['dt']))
        Q1 = int(data['Q1'])
        Q2 = int(data['Q2'])
        Q3 = int(data['Q3'])
        Q4 = int(data['Q4'])
        Q5 = int(data['Q5'])
        global z_0
        z_0 = np.array([Q1, Q2, Q3, Q4, Q5]).reshape(-1, 1)
        global y_0
        y_0 = np.array(
            [1, Q1, Q2, Q3, Q4, Q5, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]).reshape(-1, 1)
        return "ok"

@app.route('/getdata2', methods=['GET', 'POST'])
def test2():
    print(N)
    return "ok2"


## parameters

tmp = [0.791563991, 0.344248271, 0.228542842, 0.340894616, 0.294072451]
e_a = np.diag(tmp)
e_b = np.diag(tmp)
q = [2.018395e-7, 0.80259e-7, 1.569895e-7, 0.91911e-7, 2.277465e-7]  # Qk的对角元素
Q_a = np.diag(q)
Q_b = np.diag(q)
Lambda = \
    [[3.28215684e-07, 6.63839012e-09, -1.00268907e-08, 4.48132892e-09, 8.61585966e-09],
     [5.07531629e-09, 1.93484228e-07, -1.19245828e-08, 7.13086914e-09, 4.95035541e-09],
     [-4.91166675e-09, 3.77172646e-09, 2.74169796e-07, 7.14052929e-09, -4.38602488e-09],
     [-2.92290470e-11, 1.02314373e-09, 7.92652124e-09, 3.11588892e-07, 2.62929654e-09],
     [1.30935855e-08, 2.80971602e-09, 1.80795822e-09, 5.21900473e-09, 3.62881350e-07]]

s = [0.016160851157205888, 0.012192241706520914, 0.014377036385239097, 0.013019132451480479, 0.016170088040406753]

A = compute_A(m, e_a, e_b)
B = compute_B(m, e_a, e_b, Q_a, Q_b, Lambda)


# 超参数声明
##change this value to verify size


POPULATION_SIZE = 1000  # number of individuals in population
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1  # probability for mutating an individual
MAX_GENERATION = 2  # max number of generations for stopping condition


def rand():
    return random.uniform(100, 1000)


# def compute_Dbar(D, alpha, delta_bar, Sigma_u):
#   2 * D + alpha * delta_bar @ Sigma_u @ delta_bar.T
def compute_Dbar(D):
    D_bar = 2 * D
    return D_bar


def compute_Abar(A, N):
    tmp = np.eye(len(A))
    A_bar = tmp
    for i in range(1, N + 1):
        tmp = tmp @ A
        A_bar = np.hstack((A_bar, tmp))

    return A_bar.T





def Fitness(individual):
    x = np.array(individual).reshape(ONE_MAX_LENGTH, -1)
    if ((x > 0).all()):
        ret = 1 / 2 * x.T @ D_bar @ x + (y_0.T @ A_bar.T @ N_bar) @ x + z_0.T @ Lambda @ z_0
        return ret,  # deap中的适用度表示为元组，因此，当返回单个值时，需要用逗号将其声明为元组。
    else:
        return 1145141919810,  # magic number 惩罚拉满


def getdata():
    global A
    A = compute_A(m, e_a, e_b)

    global B_bar
    B_bar = compute_Bbar(A, B, N, m)

    global ONE_MAX_LENGTH
    ONE_MAX_LENGTH = 2 * (N + 1) * m  # length of tuple to be optimized

    global toolbox
    toolbox = base.Toolbox()  # 定义随机变量
    toolbox.register("Rand", rand)  # 注册随机数运算

    global creator
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 多变量最小化 Yes！
    creator.create("Individual", list, fitness=creator.FitnessMin)  # 个体类
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Rand, ONE_MAX_LENGTH)  # 实例化对象个体

    # 实例化群体
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    # 注意：由于未定义最后一个实例化参数（个体数量），在随后的调用中要自行定义
    global N_bar
    N_bar = getNBar(s, Lambda, m, N)
    global Q_bar
    Q_bar = getQBar(q, N)

    global D
    D = compute_D(B_bar, N_bar, Q_bar)
    # Sigma_u = calculate_cov_u(prices)
    global delta_bar
    delta_bar = calculate_delta_bar(m, N)
    global D_bar
    D_bar = compute_Dbar(D)
    global A_bar
    A_bar = compute_Abar(A, N)
    # 注册评价函数的别名
    toolbox.register("evaluate", Fitness)
    # deap默认使用“evaluate”作为评价函数名

    toolbox.register("select", tools.selTournament, tournsize=30)

    toolbox.register("mate", tools.cxOnePoint)
    # mutGaussian函数遍历个体的所有特征，并且对于每个特征值，
    # 都将使用indpb参数值作为概率。
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)


    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0
    fitnessValues = list(map(toolbox.evaluate, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    fitnessValues = [individual.fitness.values[0][0][0] for individual in population]

    maxFitnessValues = []
    meanFitnessValues = []
    #    populationSize = []    # for test： 确认population没有偷偷减少

    cnt = 0
    while max(fitnessValues) < 100 * ONE_MAX_LENGTH and generationCounter < MAX_GENERATION:
        generationCounter = generationCounter + 1
        #        populationSize.append(len(population))

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):  # 随机交配， 概率0.9（90%）
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:  # 随机变异， 概率0.1
                toolbox.mutate(mutant)
                del mutant.fitness.values

        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring

        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitnessValue = max(fitnessValues)
        meanFitnessValue = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitnessValue)
        meanFitnessValues.append(meanFitnessValue)

        best_index = fitnessValues.index(max(fitnessValues))


        if generationCounter == 2:
            return str((population[best_index]))


# 启动服务，监听 9000 端口，监听地址为 0.0.0.0
if __name__ == '__main__':
    app.run(debug=IS_SERVERLESS != True, port=9000, host='0.0.0.0')
