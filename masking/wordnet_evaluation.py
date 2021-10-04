import csv
import re
import numpy as np
from nltk.corpus import wordnet as wn
from bert_as_mlm import getTopTen
import os
import argparse

#predictionfiles = ["singular","singular_can_be","plural_can_be","singular_usually","plural_usually","singular_generally","plural_generally","plural","plural_some","plural_all","plural_most"]

#goldfile = "McRae_properties.gold"

def read_gold_file(goldfile):
        features = dict()
        with open(goldfile, 'r') as gold:
                reader = csv.reader(gold, delimiter='\t')
                for row in reader:
                        feat = re.sub('is_(.*)', r'\1', row[1])
                        ats = []
                        # some properties in MRD involve two adjectives separated by '/'
                        ats = ([str(x.strip()) for x in feat.split("/") if "/" in feat])
                        if '_' not in feat:
                                if row[0] in features:
                                        if len(ats) > 0:
                                                for a in ats: features[row[0]].append(str(a.strip()))
                                        else:
                                                features[row[0]].append(str(feat.strip()))
                                else:
                                        if len(ats) > 0:
                                                features[row[0]] = []
                                                for a in ats: features[row[0]].append(str(a.strip()))
                                        else:
                                                features[row[0]] = [str(feat.strip())]
        return features

def expand_gold_features(gold_features):
	expanded_gold_features = []
	for word in gold_features:
		expanded_gold_features.append(word)
		synsets = wn.synsets(word,pos='a')
		for ss in synsets:
			lemmas = ss.lemmas()
			for lemma in lemmas:
				expanded_gold_features.append(lemma.name())
	return list(set(expanded_gold_features))


if __name__ == "__main__":

        parser = argparse.ArgumentParser()
	parser.add_argument("--predictions_dir", default="cloze_queries/properties/predictions/bert-large", type=str, help="path to directory containing files with predictions")
        parser.add_argument("--gold", default="McRae_properties.gold" type=str, help="Provide the file with ground-truth property labels for evaluation.")
	predictions_dir = args.predictions_dir
        goldfile = args.gold        

        features = read_gold_file(goldfile)

        for pf in os.listdir(predictions_dir):
                predictionFile = predictions_dir + "/" + pf


                predict_dict = {}
                total_words = 0
                noun_forms = {}
                totalws_one = 0
               	totalws_five = 0
       	        totalws_ten = 0
                correctws_one = 0
                correctws_five = 0
                correctws_ten = 0
                maxpossible_one = 0
                maxpossible_five = 0
                maxpossible_ten = 0        

                with open(predictionFile, 'r') as p:
                        lines = p.readlines()
                        for row in lines:
                                word, sentence, predictions = row.split(' :: ')
                                nounform = sentence.strip().split()[0]
                                noun_forms[word.strip()] = nounform

                                topten = getTopTen(predictions)
                                predict_dict[word.strip()] = topten
                                total_words += 1

                number_words_with_rec_at_1 = number_words_with_rec_at_5 = number_words_with_rec_at_10 = sum_rec_at_1 = sum_rec_at_5 = sum_rec_at_10 = 0
                number_gold_features = inters_top_ten_number_items  = {}

                for noun in predict_dict:
                        predicted = predict_dict[noun]
                        predicted = np.array(predicted)

                        if noun in features:
                                gold_features = features[noun]
                                gold_features = np.array(gold_features)

                                top_one = predicted[0:1]
                                top_five = predicted[0:5]
                                top_ten = predicted[0:10]
                                totalws_one +=1
                                totalws_five +=5
                                totalws_ten +=10

			        # expand gold features with wordnet synonyms
                                exp_gold_features = expand_gold_features(gold_features)

                                expinters_top_one = np.intersect1d(exp_gold_features, top_one)
                                expinters_top_five = np.intersect1d(exp_gold_features, top_five)
                                expinters_top_ten = np.intersect1d(exp_gold_features, top_ten)

                                maxpossible_one +=1
                                maxpossible_five += 5 if len(gold_features) >= 5 else len(gold_features)
                                maxpossible_ten += 10 if len(gold_features) >= 10	else len(gold_features)

                                rec_at1 = rec_at5 = rec_at10 = 0.0

                                # proportion of gold features that were predicted
                                rec_at1 = len(expinters_top_one) / len(gold_features)
                                rec_at5 = len(expinters_top_five) / len(gold_features)
                                rec_at10 = len(expinters_top_ten) / len(gold_features)
                                correctws_one += len(expinters_top_one)
                                correctws_five += len(expinters_top_five)
                                correctws_ten += len(expinters_top_ten)
                                                                                         
                                sum_rec_at_1 += rec_at1
                                sum_rec_at_5 += rec_at5
                                sum_rec_at_10 += rec_at10

                                if (rec_at1 > 0):
                                        number_words_with_rec_at_1 += 1

                                if (rec_at5 > 0):
                                        number_words_with_rec_at_5 += 1

                                if (rec_at10 > 0):
                                        number_words_with_rec_at_10 += 1


                print("*********", pf, "**********")

                print('Proportion of properties (over predicted words) that are correct: ')
                print('@1 :', correctws_one/totalws_one, "max is:", maxpossible_one/totalws_one)
                print('@5 :', correctws_five/totalws_five,  "max is:", maxpossible_five/totalws_five)
                print('@10 :', correctws_ten/totalws_ten,  "max is:", maxpossible_ten/totalws_ten)

                print('NUMBER of properties that are correct: ')
                print('@1 :', correctws_one, "max is:", maxpossible_one)
                print('@5 :', correctws_five,  "max is:", maxpossible_five)
                print('@10 :', correctws_ten,  "max is:", maxpossible_ten)

                print('Number of words with at least one correct property predicted at a specific rank: ')
                print('@1 :', number_words_with_rec_at_1)
                print('@5 :', number_words_with_rec_at_5)
                print('@10 :', number_words_with_rec_at_10)

                print('total_words :', total_words)

                ## Average over all words (even the ones for which no correct predictions were made: total_words)
                macro_avg_recat1 = sum_rec_at_1/total_words
                macro_avg_recat5 = sum_rec_at_5/total_words
                macro_avg_recat10 = sum_rec_at_10/total_words
                ## Average over words for which a correct prediction was made @1, 5 or 10)
                macro_avg_recat1_bis = sum_rec_at_1/number_words_with_rec_at_1
                macro_avg_recat5_bis = sum_rec_at_5/number_words_with_rec_at_5
                macro_avg_recat10_bis = sum_rec_at_10/number_words_with_rec_at_10

                print ('Average recall over words for which at least one correct prediction was made @1, 5 or 10: ')

                print('macro_avg_recat1_bis :', '%.2f' % macro_avg_recat1_bis)
                print('macro_avg_recat5_bis :', '%.2f' % macro_avg_recat5_bis)
                print('macro_avg_recat10_bis :', '%.2f' % macro_avg_recat10_bis)

                print("\n")
