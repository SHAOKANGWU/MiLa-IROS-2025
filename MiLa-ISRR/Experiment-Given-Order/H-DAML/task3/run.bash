#python maml-task3.py --epoch 50000 --batch 4 --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 2 --multi-dt 3

#python eval-maml-task3.py --epoch 50000 --batch EVAL_BATCH --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 2 --multi-dt 3

#python convert-maml-task3.py --epoch 50000 --batch 1 --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 2 --multi-dt 3 --device cpu


#python maml-task3.py --epoch 50000 --batch 4 --save-name tmp --inner-lr 0.0005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 6 --subtask 2 --multi-dt 6

#--epoch 50000 --batch 4 --save-name tmp --inner-lr 0.00005 --outer-lr 0.00001 --loss-factor 10000 --adaptation-steps 1 --interval 6 --subtask 2 --multi-dt 6

python maml-task3.py --epoch 50000 --batch 6 --save-name tmp --inner-lr 0.0001 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 6 --subtask 2 --multi-dt 6
