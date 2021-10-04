from transformers import BertTokenizer
import numpy as np
from scipy.stats import chi2_contingency
from bert_as_mlm import getTopTen
from wordnet_evaluation import read_gold_file
import os
import argparse


def convertnoun(noun, plural_nouns, pf):
        if "plural" in pf:
                return plural_nouns[noun]
        else:
             	return noun



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
	parser.add_argument("--predictions_dir", default="cloze_queries/properties/predictions/bert-large", type=str, help="path to directory containing files with predictions")
        parser.add_argument("--gold", default="McRae_properties.gold" type=str, help="Provide the file with ground-truth property labels for evaluation.")
        parser.add_argument("--model_name", default="bert-base-uncased", type=str, help="the name of the model (for the tokenizer)")
	predictions_dir = args.predictions_dir
        goldfile = args.gold     

	tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")   	

	features = read_gold_file(goldfile)

	morethanonepiece_global = []
	for noun in features:
        	tokenized = tokenizer.tokenize(noun)
	        if len(tokenized) > 1:
        	        morethanonepiece_global.append(noun)

	print("proportion of nouns with more than one piece in the whole dataset:", len(morethanonepiece_global)/len(features))

	# plural forms
	plural_nouns = dict()
	with open("../cloze_queries/properties/predictions/bert-large/plural", 'r') as p:
		lines = p.readlines()
		for row in lines:
			word, sentence, predictions = row.split(' :: ')
			nounform = sentence.strip().split()[0]
			plural_nouns[word.strip()] = nounform


	for pf in os.listdir(predictions_dir):
        	predictionFile = predictions_dir + "/" + pf 

	        predict_dict = {}
        	total_words = 0
	        noun_forms = {}

        	with open(predictionFile, 'r') as p:
                	lines = p.readlines()
	                for row in lines:
        	                word, sentence, predictions = row.split(' :: ')
                	        nounform = sentence.strip().split()[0]
                        	noun_forms[word.strip()] = nounform

	                        topten = getTopTen(predictions)
        	                predict_dict[word.strip()] = topten
                	        total_words += 1

	        nothingfound_nouns_by_k = dict()
        	somethingfound_nouns_by_k = dict()

	        for k in [1,5,10]:
        	        nothingfound_nouns_by_k[k] = []
                	somethingfound_nouns_by_k[k] = []

	        for noun in predict_dict:
        	        predicted = predict_dict[noun]
                	predicted = np.array(predicted)

	                if noun in features:
        	                gold_features = features[noun]
                	        gold_features = np.array(gold_features)

                        	top_one = predicted[0:1]
	                        top_five = predicted[0:5]
        	                top_ten = predicted[0:10]

                	        inters_top_one = np.intersect1d(gold_features, top_one)
                        	inters_top_five = np.intersect1d(gold_features, top_five)
	                        inters_top_ten = np.intersect1d(gold_features, top_ten)

        	                if len(inters_top_one) == 0:
                	                nothingfound_nouns_by_k[1].append(noun)
                        	else:
                                	somethingfound_nouns_by_k[1].append(noun)
	                        if len(inters_top_five) == 0:
        	                        nothingfound_nouns_by_k[5].append(noun)
                	        else:
                        	        somethingfound_nouns_by_k[5].append(noun)
	                        if len(inters_top_ten) == 0:
        	                        nothingfound_nouns_by_k[10].append(noun)
                	        else:
                        	        somethingfound_nouns_by_k[10].append(noun)


        	for k in [10]: 
                	print("total wrong nouns @"+str(k)+":", len(nothingfound_nouns_by_k[k]))
	                incorrect_morethanonepiece = []
        	        correct_morethanonepiece = []

                	for noun in nothingfound_nouns_by_k[k]:
                        	if len(tokenizer.tokenize(convertnoun(noun, plural_nouns, pf))) > 1:
                                	incorrect_morethanonepiece.append(noun)
	                for noun in somethingfound_nouns_by_k[k]:
        	                if len(tokenizer.tokenize(convertnoun(noun, plural_nouns, pf))) > 1:
                	                correct_morethanonepiece.append(noun)

	        print("*******",pf,"********")
        	print("proportion of nouns with more than one piece:", len(incorrect_morethanonepiece)/len(nothingfound_nouns_by_k[k]))
	        print("number of correct singleword nouns:", len(somethingfound_nouns_by_k[k])-len(correct_morethanonepiece))
        	print("number of correct multiword nouns:", len(correct_morethanonepiece))
	        print("number of incorrect singleword nouns:", len(nothingfound_nouns_by_k[k])-len(incorrect_morethanonepiece)) # total incorrect nouns - the multipiece ones
        	print("number of incorrect multiword nouns:", len(incorrect_morethanonepiece))


	        result= chi2_contingency([[len(somethingfound_nouns_by_k[k])-len(correct_morethanonepiece),len(correct_morethanonepiece)], [len(nothingfound_nouns_by_k[k])-len(incorrect_morethanonepiece), len(incorrect_morethanonepiece)]])
        	print("chisquare p value:", result[1])
	        print("chisquare X value:", result[0])

