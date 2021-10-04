import os
import numpy as np
from nltk.corpus import wordnet as wn
from wordnet_evaluation import read_gold_file
from bert_as_mlm import getTopTen
import argparse


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

                all_adjs_top_one = []
                all_adjs_top_five = []
                all_adjs_top_ten = []

                all_preds_top_one = []
       	        all_preds_top_five = []
               	all_preds_top_ten = []


                for noun in predict_dict:
                        predicted = predict_dict[noun]
                        predicted = np.array(predicted)

                        if noun in features:
                                gold_features = features[noun]
                                gold_features = np.array(gold_features)

                                top_one = predicted[0:1]
                                top_five = predicted[0:5]
                                top_ten = predicted[0:10]

                                #### now calculate the proportion of predictions that are adjectives
                                for w in top_one:
                                        all_preds_top_one.append(w)
                                        if len(wn.synsets(w, "a")) > 0:
                                                all_adjs_top_one.append(w)
               	       	       	for w in top_five:
       	               	       	       	all_preds_top_five.append(w)
       	       	               	       	if len(wn.synsets(w, "a")) > 0:
       	       	       	               	       	all_adjs_top_five.append(w)
               	       	       	for w in top_ten:
       	               	       	       	all_preds_top_ten.append(w)
       	       	               	       	if len(wn.synsets(w, "a")) > 0:
       	       	       	               	       	all_adjs_top_ten.append(w)


                print("*********", pf, "**********")


                print('Proportion of predicted words that are adjectives: ')
                print('@1 :', len(all_adjs_top_one)/len(all_preds_top_one))
                print('@5 :', len(all_adjs_top_five)/len(all_preds_top_five))
                print('@10 :', len(all_adjs_top_ten)/len(all_preds_top_ten))

                print("\n")
