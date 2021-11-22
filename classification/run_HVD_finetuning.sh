

####### BERT (TOK) NOUN-NOUN


MODEL_TYPE="bert-bytoken"
MODEL_NAME="bert-base-uncased"
COMPARISON="noun_noun"

: '

for TEST_FOLD in 1 2 3 4 5
do
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounnoun_lr5e-5/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounnoun_lr3e-5/fold$TESTFOLD/  --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounnoun_lr1e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounnoun_lr1e-6/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-6 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounnoun_lr5e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounnoun_lr3e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
done
'


####### BERT (TOK) NOUN-ADJ

COMPARISON="noun_adj"

for TEST_FOLD in 1 2 3 4 5
do

#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounadj_lr5e-5/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounadj_lr3e-5/fold$TESTFOLD/  --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounadj_lr1e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounadj_lr1e-6/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-6 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounadj_lr5e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_nounadj_lr3e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
done


####### BERT (TOK) NOUN - ADJ+NOUN

COMPARISON="noun_adjnoun"

: '
for fold in 1 2 3 4 5
do
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_noun-adjnoun_lr5e-5/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_noun-adjnoun_lr3e-5/fold$TESTFOLD/  --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_noun-adjnoun_lr1e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_noun-adjnoun_lr1e-6/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-6 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_noun-adjnoun_lr5e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/bytoken_noun-adjnoun_lr3e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
done
'


####### BERT (CLS) 

MODEL_TYPE="bert"

for fold in 1 2 3 4 5
do
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/cls_lr5e-5/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/cls_lr3e-5/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-5 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/cls_lr1e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/cls_lr1e-6/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 1e-6 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/cls_lr5e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 5e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLD
#python finetuning.py --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME --output_dir FTresults_HVD/cls_lr3e-4/fold$TESTFOLD/ --do_train --do_eval --evaluate_during_training --per_gpu_train_batch_size 8 --num_train_epochs 3 --save_preds --save_epochs --save_model --dataset HVD --learning_rate 3e-4 --eval_dataset HVD --comparison $COMPARISON --test_fold $TEST_FOLDdone
done





