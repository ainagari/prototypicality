import os
import sys
import re
import sklearn
from sklearn.metrics import cohen_kappa_score
from agreement import returnValues

fileA1 = open(sys.argv[1], 'r')
fileA2 = open(sys.argv[2], 'r')

scoresA1 = returnValues(fileA1)
scoresA2 = returnValues(fileA2)

# @10
flatscoresA1 = []
flatscoresA2 = []

# @5
flatscoresA1_5 = []
flatscoresA2_5 = []

# @1
flatscoresA1_1 = []
flatscoresA2_1 = []


for nounvalues in scoresA1:
        for ANvalue in nounvalues:
                flatscoresA1.append(ANvalue)
        for ANvalue in nounvalues[:5]:
                flatscoresA1_5.append(ANvalue)
        for ANvalue in nounvalues[:1]:
                flatscoresA1_1.append(ANvalue)


for nounvalues in scoresA2:
        for ANvalue in nounvalues:
                flatscoresA2.append(ANvalue)
        for ANvalue in nounvalues[:5]:
                flatscoresA2_5.append(ANvalue)
        for ANvalue in nounvalues[:1]:
                flatscoresA2_1.append(ANvalue)


flatscoresBoth = []
for m, a in zip(flatscoresA1, flatscoresA2):
	if m == 1 and a == 1:
		flatscoresBoth.append(1)
	else:
		flatscoresBoth.append(0)


flatscoresBoth5 = []
for m, a in zip(flatscoresA1_5, flatscoresA2_5):
        if m == 1 and a == 1:
                flatscoresBoth5.append(1)
        else:
                flatscoresBoth5.append(0)

flatscoresBoth1 = []
for m, a in zip(flatscoresA1_1, flatscoresA2_1):
        if m == 1 and a == 1:
                flatscoresBoth1.append(1)
        else:
                flatscoresBoth1.append(0)



print("NUMBER of correct properties according to:")
print("------@ 1: -------")
print("A1:", sum(flatscoresA1_1))
print("A2:", sum(flatscoresA2_1))
print("both:", sum(flatscoresBoth1))

print("------@ 5: -------")
print("A1:", sum(flatscoresA1_5))
print("A2:", sum(flatscoresA2_5))
print("both:", sum(flatscoresBoth5))


print("------@ 10: -------")
print("A1:", sum(flatscoresA1))
print("A2:", sum(flatscoresA2))
print("both:", sum(flatscoresBoth))


