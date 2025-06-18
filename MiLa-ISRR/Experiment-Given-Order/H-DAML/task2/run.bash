# training
python maml-task2.py --epoch 50000 --batch 4 --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 1 --multi-dt 3

# evaluation
python eval-maml-task2.py --epoch 50000 --batch EVAL_BATCH --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 1 --multi-dt 3

# testing
python convert-maml-task2.py --epoch 50000 --batch 1 --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 1 --multi-dt 3 --device cpu
