


This directory contains the code to run the classification and fine-tuning experiments described in Sections 5 and 6 of the paper.


## Embedding-based classification

To run the embedding-based classification experiments (Section 5.2) with BERT embeddings, you first need to extract BERT representations with the following command:

`python extract_BERT_representations.py` 

By default, `bert-base-uncased` is used. You can specify the name or path of another model with the flag `--model_name_or_path`.
They will be saved as a .pkl file called "HVD_BERT_representations.pkl". You can also extract representations for BERT (ISO) by using the flag --bert_iso. They will be saved in a file called "HVD_BERT-ISO_representations.pkl".

Once representations have been extracted, run:

`python cv_classif_mcrae.py --vector_type bert --path_to_vectors HVD_BERT_representations.pkl`

or, for BERT (ISO):

`python cv_classif_mcrae.py --vector_type bert-iso --path_to_vectors HVD_BERT-ISO_representations.pkl`

Results of the best configuration will be printed on screen and the full results and predictions will be saved into `CVresults/` as .pkl files.

To run classification with other embedding types (word2vec or fasttext), you need to download the pymagnitude library and download the embeddings in .magnitude format. You can do so [here](https://github.com/plasticityai/magnitude). 

`python cv_classif_mcrae.py --vector_type [w2v|fasttext] --path_to_vectors [path to .magnitude vectors]`

To run the baselines:

`python cv_classif_mcrae.py --baseline [majority|allproto]`


## Fine-tuning on HVD

To run the fine-tuning experiments on HVD (Section 5.3), you can use the command:

`sh run_HVD_finetuning.sh`

This will run the BERT-CLS and BERT-TOK cross-validation experiments that gave the best results in the paper. Results, predictions and models will be saved in `FTresults_HVD/`.

## Fine-tuning on Addone

To fine-tune BERT on the Addone dataset, run:

`sh run_Addone_finetuning.sh`

Similarly to the command for HVD, this will run the BERT-CLS and BERT-TOK experiments that gave the best results in the paper.

You do not need to download any additional data. The Addone dataset was downloaded from [here](www.seas.upenn.edu/~nlp/resources/AN-composition.tgz) and converted into .pkl files for our experiments.



