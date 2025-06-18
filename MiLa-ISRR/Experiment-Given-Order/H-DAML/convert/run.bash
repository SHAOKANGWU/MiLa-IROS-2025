python convert-task1.py --batch 10 --save-name tmp --inner-lr 0.005 --outer-lr 0.001 --loss-factor 100 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3 --device cpu

# TODO: old model, time step = 3
#python convert-task2.py --batch 1 --save-name tmp --inner-lr 0.005 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3 --subtask 1 --multi-dt 3 --device cpu

# TODO: new model. time step = 6
python convert-task2.py --batch 1 --save-name tmp --inner-lr 0.001 --outer-lr 0.0001 --loss-factor 1000 --adaptation-steps 1 --interval 6 --subtask 1 --multi-dt 6 --device cpu


python convert-task3.py --batch 6 --save-name tmp --inner-lr 0.0001 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 6 --subtask 2 --multi-dt 6 --device cpu
