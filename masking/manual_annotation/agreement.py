import os
import sys
import re
import sklearn
from sklearn.metrics import cohen_kappa_score



def has_numbers(inputString):
	return bool(re.search(r'\d', inputString))

def returnValues(inputFileName):
	content = inputFileName.readlines()	
	scores = []

	for line in content:
		line = line.rstrip('\n')
		line = line.rstrip('\r')
		if has_numbers(line):
			line.rstrip()
			line = line.lstrip(',,')

			values = line.split(',')
			valuesList = []
			for v in values:
				valuesList.append(int(v))
		
			scores.append(valuesList) 
	return(scores)

def calculateKappa(labeler1,labeler2):
	kappa = cohen_kappa_score(labeler1, labeler2)
	return(kappa)


if __name__ == "__main__":

	fileA1 = open(sys.argv[1], 'r')
	fileA2 = open(sys.argv[2], 'r')

	scoresA1 = returnValues(fileA1)
	scoresA2 = returnValues(fileA2)


	flatscoresA1 = []
	flatscoresA2 = []

	for nounvalues in scoresA1:
		for ANvalue in nounvalues:
			flatscoresA1.append(ANvalue)

	for nounvalues in scoresA2:
		for ANvalue in nounvalues:
			flatscoresA2.append(ANvalue)




	for i in range(len(scoresA1)):
		valuesA1 = scoresA1[i]
		print('valM : ',valuesA1) 
		if len(valuesA1) < 10:
			print('less than 10 in valuesA1 : ', i)
			raise ValueError('less than 10 in valuesA1 : ', i)
			sys.exit()
		valuesA2 = scoresA2[i]
		if len(valuesA2) < 10:
			print('less than 10 in valuesA2 : ', i)
			raise ValueError('less than 10 in valuesA2 : ', i)
			sys.exit()
		print('valA : ',valuesA2)
		kappa = calculateKappa(valuesA1,valuesA2)
		print('kappa : ', kappa)
		i += 1
		

	microkappa = calculateKappa(flatscoresA1, flatscoresA2)
	print("micro-'averaged' kappa is:", microkappa)
	print ('lenScoresA1 : ', len(scoresA1), 'lenScoresA2 : ', len(scoresA2))
