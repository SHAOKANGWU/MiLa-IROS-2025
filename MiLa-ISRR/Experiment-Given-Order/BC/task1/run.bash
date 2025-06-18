# TODO: Hyperparameter 1 (i.e., learning rate)
python bc-task1.py --epoch 50000 --batch 10 --save-name tmp-bc-task1-hyper1 --lr 0.001 --loss-factor 100 --interval 3 --subtask 0 --multi-dt 3

# TODO: Hyperparameter 1 (i.e., learning rate), with changeable goal
python bc-task1-novel.py --epoch 50000 --batch 10 --save-name tmp-bc-task1-hyper1-novel --lr 0.001 --loss-factor 100 --interval 3 --subtask 0 --multi-dt 3




# TODO: Hyperparameter 2 (i.e., learning rate)
python bc-task1.py --epoch 50000 --batch 10 --save-name tmp-bc-task1-hyper2 --lr 0.0005 --loss-factor 100 --interval 3 --subtask 0 --multi-dt 3

# TODO: Hyperparameter 2 (i.e., learning rate), with changeable goal
python bc-task1-novel.py --epoch 50000 --batch 10 --save-name tmp-bc-task1-hyper2-novel --lr 0.0005 --loss-factor 100 --interval 3 --subtask 0 --multi-dt 3


