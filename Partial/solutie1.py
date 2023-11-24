from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import random
#functie care determina aruncarea unei monede ce probabilitate de a primi stema: probability

def trowCoin(probability):
    if (random.random() < probability):
        return 1
    return 0

def game():
    f = trowCoin(1/2)
    # F este stema deci P0 joaca primul
    if (f):
        #Player1 arunca moneda cu pStema = 1/3
        player0 = trowCoin(1/3)
        player1 = 0
        #Player2 arunca player1+1 monede cu probabilitate 1/2
        for i in range(player0+1):
            player1 += trowCoin(1/2)
        if player0 >= player1:
            return 0
        return 1
    else:
        #f nu este stema deci p1 joaca
        player1 = trowCoin(1/2)
        player0 = 0
        for i in range(player1+1):
            player0 += trowCoin(1/3)
        if player1 >= player0:
            return 1
        return 0

totalResults = 0
numberGamesPlayed = 20000
for _ in range(numberGamesPlayed):
    totalResults += game()

print(f"Player1 are sanse de castig de: ",totalResults/numberGamesPlayed)
print(f"Player2 are sanse de castig de: ",(numberGamesPlayed-totalResults)/numberGamesPlayed)

# definirea modelului bayesian
bayesModel = BayesianNetwork([('Start', 'P0'), ('P0', 'P1'), ('Start', 'P1')])

#definirea nodului de start cu sansa de 50/50
cpd_PlayerStart = TabularCPD('Start', 2, [[0.5], [0.5]])

#definirea nodului de player1 care in functie de cine castiga prima runda are valori diferite
cpd_P0 = TabularCPD('P0', 2, [[2/3, 0.5], 
                              [1/3, 0.5]], 
                             evidence=['Start'], evidence_card=[2])
#definirea celei dea doua runde in functie de cine o castiga pe prima
cpd_P1 = TabularCPD('P1', 2, [[2/3, 1/3, 1/2, 1/2], 
                              [1/3, 2/3, 1/2, 1/2]], 
                              evidence=['P0', 'Start'], evidence_card=[2, 2])

bayesModel.add_cpds(cpd_PlayerStart, cpd_P0, cpd_P1)

assert bayesModel.check_model()

# Inferenta pe modelul de mai sus
infer = VariableElimination(bayesModel)
# Ce valoare are start stiind ca P1 = 0
prob = infer.query(variables=['P0'], evidence={'P1': 0}) 
print(prob)
        
        




    