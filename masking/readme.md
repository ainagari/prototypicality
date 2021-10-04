

## DATA and MODEL PREDICTIONS for Cloze task experiments


### QUANTIFIER MASKING

#### QUERIES

The folder `cloze_queries/quantifiers/queries` contains two files: `SetA.mask.queries` and `SetB.mask.queries`. 

The files contain the cloze statements used to query BERT for the quantifiers of the nouns in Set A and Set B. The composition of each set is described in detail in the paper.


#### PREDICTIONS

The folder `cloze_queries/quantifiers/predictions/bert-base` and `bert-large` contain four files each. 

`SetA.bert-[base|large]-uncased`, `SetB.bert-[base|large]-uncased`: Each of these files contains the top ten predictions made by the bert-base-uncased and bert-large-uncased models for filling the quantifier slot in Set A and B queries.

`SetA.bert-[base|large]-uncased_probabilities`, `SetB.bert-[base|large]-uncased_probabilities`: The files ended in "\_probabilities" contain, additionally, the probabilities assigned by the model to each of the top ten predictions, as well as to the quantifiers ALL, SOME and MOST.


### PROPERTY MASKING

#### QUERIES

The folder `cloze_queries/properties/queries` contains the cloze statements used to query BERT for noun properties. 

The format of the files is "NOUN :: cloze-task-query" where NOUN is the noun from the McRae dataset for which a query is proposed. For example:

```
pearl :: a pearl is [MASK].
banjo :: banjos are [MASK].
```

NOUN is in the form in which it is found in the McRae norms. 

The name of the files denotes the singular or plural cloze task template used to generate the queries. 
e.g., queries in `plural_usually.prop` are in plural form and contain the adverb "usually" (_veils are usually [MASK]._).


#### PREDICTIONS

(in `cloze_queries/properties/predictions/bert-base` and `bert-large`)

Each file in these folders contains the top ten predictions made by the bert-base-uncased and bert-large-uncased models for the masked tokens in the corresponding queries. The name of each file denotes the singular or plural query template that was used to generate the cloze-task statements therein. The format of the files is as follows: 

```
NOUN :: query :: [list of comma-separated predictions] 

e.g., pearl  ::  pearls can be [MASK].  ::  ['used', 'found', 'worn', 'made', 'sold', 'produced', 'bought', 'added', 'caught', 'kept']
```


### MAKING PREDICTIONS

If you wish to use BERT for property or quantifier prediction, you can run the `bert_as_mlm.py` script with the following arguments:


For the evaluation of property predictions:

+ `--model_name`: The bert model we want to query. Default model: bert-base-uncased. Possible values: bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased.
	
+ `--input_fileOrDir`: The file with the cloze-task queries. Or the current directory containing the query files.

+ `--predict_properties`: Use this argument to predict the masked properties.

+ `--eval_properties`: Use this argument to evaluate noun property masking predictions.
						
+ `--predict_and_eval_quantifiers`: Use this argument to predict the masked quantifiers and evaluate the predictions.
						
+ `--gold`: Provide the file with ground-truth property labels for evaluation.


Usage examples

##### To predict MASKED PROPERTIES

`python bert_as_mlm.py --input_fileOrDir cloze_queries/properties/queries --predict_properties`

- Model predictions for each type of query will be saved in the files named: `[PATTERN].[MODEL NAME].prop` 
There are 11 queries and 11 output files, one for each query.

##### To evaluate the predictions of masked properties --

`python bert_as_mlm.py --input_fileOrDir cloze_queries/properties/predictions --eval_properties --gold McRae_properties.gold`

- Evaluation results are in the file: `property-eval-results_[MODEL NAME].txt` 
(e.g., property-eval-results_bert-base-uncased.txt)


##### To predict MASKED QUANTIFIERS and evaluate them 

`python bert_as_mlm.py --input_fileOrDir cloze_queries/properties/predictions --predict_and_eval --gold McRae_properties.gold`



### WORDNET EVALUATION 

You can run the relaxed WordNet-based evaluation of masked property prediction (Section 4.1 of the paper) with the following command:

`python wordnet_evaluation.py --predictions_dir cloze_queries/properties/predictions/bert-large --gold McRae_properties.gold`


### OTHER ANALYSES 

The following command runs the analysis in Appendix B.5, which looks at the impact that word splitting has on performance:

`python wordpiece_analysis.py --predictions_dir cloze_queries/properties/predictions/bert-large --gold McRae_properties.gold --model_name bert-large-uncased`

To calculate the proportion of adjectives among predictions, you can use:

`python calculate_proportion_of_adjs.py --predictions_dir cloze_queries/properties/predictions/bert-large --gold McRae_properties.gold`












