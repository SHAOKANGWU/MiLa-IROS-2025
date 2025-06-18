#python maml-task1.py --epoch 50000 --batch 10 --save-name tmp --inner-lr 0.005 --outer-lr 0.001 --loss-factor 100 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

#python maml-task1.py --epoch 50000 --batch 10 --save-name tmp --inner-lr 0.0005 --outer-lr 0.0001 --loss-factor 1000 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

#python maml-task1.py --epoch 50000 --batch 8 --save-name maml-softmax2d-delta-task1 --inner-lr 0.0001 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

#python maml-task1.py --epoch 50000 --batch 8 --save-name maml-softmax2d-delta-task1 --inner-lr 0.0001 --outer-lr 0.00001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

# TODO: The first hyperparameter
python maml-task1.py --epoch 50000 --batch 4 --save-name maml-softmax2d-delta-task1 --inner-lr 0.0001 --outer-lr 0.00001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

# TODO: The second hyperparameter
python maml-task1.py --epoch 50000 --batch 4 --save-name maml-softmax2d-delta-task1 --inner-lr 0.00005 --outer-lr 0.00001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3