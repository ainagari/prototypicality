
'''
This script loads a pretrained bert model and uses it as a language model to predict words (or wordpieces) for [MASK] tokens in a sentence

'''

import os
import sys
import re
import csv
import argparse
import numpy as np
import torch
from torch.nn import Softmax
from transformers import (BertConfig, BertTokenizer, BertForMaskedLM)



out_files_list = []

def main():
	parser = argparse.ArgumentParser()
	## Required parameters
	parser.add_argument("--model_name", default="bert-base-uncased", type=str,
						help="The bert model we want to query.")
						# Possible values: bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased
	
	parser.add_argument("--input_fileOrDir", type=str, required=True, 
						help="The file with the cloze-task queries. Or the current directory containing the query files.")

	parser.add_argument("--predict_properties", action='store_true',  
						help="Use this argument to predict the masked properties.")	
	parser.add_argument("--eval_properties", action='store_true', 
						help="Use this argument to evaluate noun property masking predictions.")
						

	parser.add_argument("--predict_and_eval_quantifiers", action='store_true', help="Use this argument to predict masked quantifiers and evaluate the predictions.")
	
	parser.add_argument("--gold", type=str, 
						help="Provide the file with ground-truth property labels for evaluation.")
						# The gold file is needed only for property prediction evaluation. For quantifiers prediction, we check the presence of all, most and some in predictions for Set A and B.


	args = parser.parse_args()

	if args.input_fileOrDir is None:
		raise ValueError("Cannot make predictions without an input file containing cloze-task queries. Please make sure to always provide an input file.")

	if args.eval_properties:
		if args.gold is None:
			raise ValueError("Please provide the file with ground-truth/gold property labels needed for evaluation.")

	full_rankings = dict()
	if os.path.isdir(args.input_fileOrDir):
		for file in os.listdir(args.input_fileOrDir):
			if args.predict_and_eval_quantifiers:
				if file.endswith (".queries"):
					full_rankings[file] = mask_predict(args.input_fileOrDir + "/" + file,args)
			elif args.predict_properties:
				if file.endswith (".prop"):   
					mask_predict(args.input_fileOrDir + "/" + file,args)
	else: 
		mask_predict(args.input_fileOrDir,args)


	if args.eval_properties:
		predictionfiles = [args.input_fileOrDir + "/" + x for x in os.listdir(args.input_fileOrDir)]
		eval_predict_properties(predictionfiles,args.gold,args.model_name)
	
	elif args.predict_and_eval_quantifiers:
		#predictionfiles = [args.input_fileOrDir + "/" + x for x in os.listdir(args.input_fileOrDir)]
		eval_predict_quantifiers(full_rankings,args.model_name)


def eval_predict_properties(modelPredictions,goldfile,modelname):

	## the gold file
	features = {}  							 # dictionary of gold noun attributes/properties

	prop_eval_file = 'property-eval-results_' + modelname + '.txt'
	outProp = open(prop_eval_file, 'w')

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

	
	for predictionFile in modelPredictions:
		
		print(predictionFile)

		predict_dict = {}
		total_words = 0	
		
		pattern = predictionFile.split("/")[-1]
		
		## the predictions file
		with open(predictionFile, 'r') as p:
			lines = p.readlines()
			for row in lines:
				word, sentence, predictions = row.split(' :: ')
				
				topten = getTopTen(predictions)	
				predict_dict[word.strip()] = topten 
				total_words += 1

		# words for which a correct (gold) property is found in the @1, @5 or @10 positions

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

				inters_top_one = np.intersect1d(gold_features, top_one)
				inters_top_five = np.intersect1d(gold_features, top_five)
				inters_top_ten = np.intersect1d(gold_features, top_ten)

				## how many nouns have a specific number of correct properties predicted @10

				if (len(inters_top_ten) in inters_top_ten_number_items):
					inters_top_ten_number_items[len(inters_top_ten)] += 1
				else:
					inters_top_ten_number_items[len(inters_top_ten)] = 1

				rec_at1 = rec_at5 = rec_at10 = 0.0

				# proportion of gold features that were predicted

				rec_at1 = len(inters_top_one) / len(gold_features)
				rec_at5 = len(inters_top_five) / len(gold_features)
				rec_at10 = len(inters_top_ten) / len(gold_features)

				sum_rec_at_1 += rec_at1
				sum_rec_at_5 += rec_at5
				sum_rec_at_10 += rec_at10

				if (rec_at1 > 0):
					number_words_with_rec_at_1 += 1
				
				if (rec_at5 > 0):
					number_words_with_rec_at_5 += 1
				
				if (rec_at10 > 0):
					number_words_with_rec_at_10 += 1

		print('MASK PATTERN:', pattern, file=outProp)			

		print('Number of words with at least one correct property predicted at a specific rank: ',file=outProp)
		print('@1 :', number_words_with_rec_at_1, file=outProp)
		print('@5 :', number_words_with_rec_at_5, file=outProp)
		print('@10 :', number_words_with_rec_at_10, file=outProp)

		print('total_words :', total_words, file=outProp)

		## Average over all words (even the ones for which no correct predictions were made: total_words)
		macro_avg_recat1 = sum_rec_at_1/total_words
		macro_avg_recat5 = sum_rec_at_5/total_words
		macro_avg_recat10 = sum_rec_at_10/total_words


		## Average over words for which a correct prediction was made @1, 5 or 10)
		macro_avg_recat1_bis = sum_rec_at_1/number_words_with_rec_at_1
		macro_avg_recat5_bis = sum_rec_at_5/number_words_with_rec_at_5
		macro_avg_recat10_bis = sum_rec_at_10/number_words_with_rec_at_10

		print ('Average recall over words for which at least one correct prediction was made @1, 5 or 10: ',file=outProp)

		print('macro_avg_recat1_bis :', '%.2f' % macro_avg_recat1_bis, file=outProp)
		print('macro_avg_recat5_bis :', '%.2f' % macro_avg_recat5_bis, file=outProp)
		print('macro_avg_recat10_bis :', '%.2f' % macro_avg_recat10_bis, file=outProp)

		print('\n',file=outProp)



def eval_predict_quantifiers(full_rankings,modelname):
	for setname in full_rankings: 
		pdb.set_trace()
		if "SetA" in setname:
			evalSetA(full_rankings[setname],modelname)
		elif "SetB" in setname:
			evalSetB(full_rankings[setname],modelname)


def getTopTen(nounPredictions):
	predictions = re.sub(r"\[(.*?)\]", r"\1", nounPredictions.strip())	
	predictions = predictions.split(', ')

	predict_list = []
	
	for p in predictions:
		p = p.replace("'","")
		predict_list.append(p)

	toptenlist = predict_list[:10]
	return(toptenlist)





def MRR(modelPredictions):
	ranks = dict()
	MRRs = dict()
	for quantifier in ["all","most","some"]:
		ranks[quantifier] = []

	for predictions in modelPredictions:
		for quantifier in ["all","most","some"]:
			ranks[quantifier].append(predictions.index(quantifier))
	for quantifier in ranks:
		MRRs[quantifier] = np.average([1 / (r+1) for r in ranks[quantifier]])
						
	return MRRs
			


def evalSetA(fullrankings,modelname):

	setA_eval_file = 'SetA-eval-results_' + modelname + '.txt' 
	outSetA = open(setA_eval_file, 'w')

	some = most = aLL = all_first = most_first = all_and_some = most_and_some = 0
	for predictions in fullrankings:
		topten = predictions[:10]
		predictions_number = len(fullrankings)
	
		if "all" in topten:
			aLL += 1
			if "some" in topten:
				all_and_some += 1
				# get the position of "all" and "some" if they are both found in the predictions
				pos_all = topten.index("all")
				pos_some = topten.index("some")
				if pos_all < pos_some:
					all_first +=1

		if "most" in topten:
			most += 1
			if "some" in topten:
				most_and_some += 1
				# get the position of "most" and "some" if they are both found in the predictions
				pos_most = topten.index("most")
				pos_some = topten.index("some")
				if pos_most < pos_some:
					most_first +=1

		if "some" in topten:
			some += 1

	MRRs = MRR(fullrankings)
	print('Evaluation Results - Quantifier Prediction for SetA', file=outSetA)
	print('ALL @10:', aLL,'/',predictions_number, file=outSetA)
	print('MOST @10:', most,'/',predictions_number, file=outSetA)
	print('SOME @10:', some,'/',predictions_number, file=outSetA)
	print('ALL < SOME:', all_first,'/',all_and_some, file=outSetA)
	print('MOST < SOME:', most_first,'/',most_and_some, file=outSetA)
	print('MRR ALL:', MRRs["all"], file=outSetA)
	print('MRR MOST:', MRRs["most"], file=outSetA)
	print('MRR SOME:', MRRs["some"], file=outSetA)


def evalSetB(fullrankings,modelname):

	setB_eval_file = 'SetB-eval-results_' + modelname + '.txt' 
	outSetB = open(setB_eval_file, 'w')

	some = aLL = most = some_and_all = some_and_most = all_and_some = some_before_all = some_before_most = some_first = some_and_all_and_most = 0

	for predictions in fullrankings:
		topten = predictions[:10]
		predictions_number = len(fullrankings)
				
		if "some" in topten:
			some += 1
			pos_some = topten.index("some")

			if "all" in topten:
				pos_all = topten.index("all")
				some_and_all += 1
				if pos_some < pos_all:
					some_before_all += 1

				if "most" in topten:
					pos_most = topten.index("most")
					some_and_all_and_most += 1

					if pos_some < pos_all:
						if pos_some < pos_most:
							some_first +=1

				
			elif "most" in topten:
				pos_most = topten.index("most")
				some_and_most += 1
				if pos_some < pos_most:
					some_before_most += 1

		if "all" in topten:
			aLL += 1

		if "most" in topten:
			most += 1

	MRRs = MRR(fullrankings)
	print('Evaluation Results - Quantifier Prediction for SetB', file=outSetB)
	print('SOME @10:',some,'/',predictions_number, file=outSetB)
	print('ALL @10:', aLL,'/',predictions_number, file=outSetB)
	print('MOST @10:', most,'/',predictions_number, file=outSetB)
	print('SOME < ALL:', some_before_all,'/',some_and_all, file=outSetB)
	print('SOME < MOST:', some_before_most,'/',some_and_most, file=outSetB)
	print('MRR ALL:', MRRs["all"], file=outSetB)
	print('MRR MOST:', MRRs["most"], file=outSetB)
	print('MRR SOME:', MRRs["some"], file=outSetB)

def mask_predict(input_file,args):		
	sentence_file = open(input_file, 'r')
	sentences = sentence_file.readlines()

	words = []
	sentences_only = []
	for sentence in sentences:
		word, sent_only = sentence.split(' :: ')
		words.append(word)
		sentences_only.append(sent_only)

	#### Setting some parameters (do not change batch size)

	model_name = args.model_name

	# The predictions made by bert for the masked tokens will be printed to this file.  

	if args.predict_properties: 
		name = re.sub(r"^(.*).prop", r'\1', input_file.split("/")[-1]) 
		output_file = name + '.' + model_name + '.prop'

	elif args.predict_and_eval_quantifiers:
		name = re.sub('.mask.queries', '', input_file.split("/")[-1]) 
		output_file = name + '.' + model_name 

	out_files_list.append(output_file)
	out = open(output_file, 'w')

	torch.set_default_tensor_type(torch.cuda.FloatTensor)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	batch_size = 1

	if "uncased" in model_name:
		tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
	else:
		tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

	##### Preparing data and model

	tokenized_sentences = [ ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences_only]
	config = BertConfig.from_pretrained(model_name)
	model = BertForMaskedLM.from_pretrained(model_name, config=config)
	softmax = Softmax(dim=0)    
	global_predictions = []
	global_predictions_fullranking = []
	global_predictions_fullranking_probs = []
	model.eval()
	N=10
	with torch.no_grad():
		for tok_sent in tokenized_sentences:
			sentence_predictions = []
			input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_sent)]).to(device) 

			outputs = model(input_ids)  
			logits = outputs[0]
	        
			masked_positions = [idx for idx, token in enumerate(tok_sent) if token == "[MASK]"] # indices of [MASK] tokens in wp-tokenized sentence
			
			for masked_position in masked_positions:
				logits_at_position = logits[0][masked_position]  				# 0 because batch size is 1
	            # logits_at_position is a vector of shape (size of vocabulary).
	            # this is the vector of "scores" assigned to each wordpiece. We can apply softmax to turn these scores into actual (easily interpretable) probabilities
	            # (if we are not interested in probabilities but just in the ranking order, we don't need to apply softmax)

				softmaxed_logits_at_position = softmax(logits_at_position)

			
				all_bestindices = torch.argsort(softmaxed_logits_at_position, descending=True)
				all_bestpieces = [tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in all_bestindices]
				all_bestprob_pieces = [tokenizer.convert_ids_to_tokens([idx.item()]) for idx in all_bestindices[:N]]
				bestNpieces = [tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in all_bestindices[:N]]

				global_predictions.append(bestNpieces)
				global_predictions_fullranking.append(all_bestpieces)
				global_predictions_fullranking_probs.append(all_bestprob_pieces)
	           
				global_predictions.append(bestNpieces)

	for word, sentence, prediction in zip(words, sentences_only, global_predictions):
		print(word, ' :: ', sentence.strip(), ' :: ', prediction, file=out)
	if args.eval_and_predict_quantifiers:
	        probabilities_outfn = name + '.' + model_name + "_probabilities"
        	with open(probabilities_outfn,'w') as outfn:
                	outfn.write("word\tsentence\tw1\tp1\tw2\tp2\tw3\tp3\tw4\tp4\tw5\tp5\tw6\tp6\tw7\tp7\tw8\tp8\tw9\tp9\tw10\tp10\tprob_all\tprob_most\tprob_some\n")
	                for word, sentence, prediction in zip(words, sentences_only, global_predictions_fullranking_probs):
                	        firstten = [str(item) for sublist in prediction[:10] for item in sublist]
                        	for w, p in prediction:
                                	if w == "all":
                                        	proball = p
	                                elif w == "some":
        	                                probsome = p
                	                elif w == "most":
                        	                probmost = p
	                        outfn.write(word + "\t" + sentence.strip() + "\t" + "\t".join(firstten) + "\t" + str(proball) + "\t" +str(probmost)+ "\t" + str(probsome) + "\n")



	return global_predictions_fullranking

if __name__ == "__main__":
    main()





