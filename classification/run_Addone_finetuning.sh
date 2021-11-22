
########################### BERT (CLS)

MODEL_TYPE="bert"
MODEL_NAME="bert-base-uncased"

#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/cls_lr5e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 5e-5
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/cls_lr3e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 3e-5
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/cls_lr1e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 1e-4
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/cls_lr1e-6/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 1e-6
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/cls_lr5e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 5e-4
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/cls_lr3e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 3e-4


#######################################################################################
######################## BERT (TOK) 

## noun_adj

MODEL_TYPE="bert-bytoken"
COMPARISON="noun_adj"


#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounadj_lr5e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 5e-5 --comparison $COMPARISON
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounadj_lr3e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 3e-5 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounadj_lr1e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 1e-4 --comparison $COMPARISON
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounadj_lr1e-6/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 1e-6 --comparison $COMPARISON
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounadj_lr5e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 5e-4 --comparison $COMPARISON
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounadj_lr3e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_epochs --save_model --dataset addone --learning_rate 3e-4 --comparison $COMPARISON


## noun_noun
COMPARISON="noun_noun"
':
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounnoun_lr5e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 5e-5 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounnoun_lr3e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 3e-5 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounnoun_lr1e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 1e-4 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounnoun_lr1e-6/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 1e-6 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounnoun_lr5e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 5e-4 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_nounnoun_lr3e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 3e-4 --comparison $COMPARISON
'

## noun-adjnoun

COMPARISON="noun_adjnoun"
':
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_noun-adjnoun_lr5e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 5e-5 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_noun-adjnoun_lr3e-5/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 3e-5 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_noun-adjnoun_lr1e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 1e-4 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_noun-adjnoun_lr1e-6/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 1e-6 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_noun-adjnoun_lr5e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 5e-4 --comparison $COMPARISON
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_Addone/bytoken_noun-adjnoun_lr3e-4/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 5 --save_preds --save_model --dataset addone --learning_rate 3e-4 --comparison $COMPARISON
'


