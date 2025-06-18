python maml-visual.py --epoch 50000 --batch 10 --save-name maml-softmax2d-nonNovel-task1 --inner-lr 0.005 --outer-lr 0.001 --loss-factor 100 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

python maml-visual-novel.py --epoch 50000 --batch 10 --save-name maml-softmax2d-task1-novel --inner-lr 0.005 --outer-lr 0.001 --loss-factor 100 --adaptation-steps 1 --interval 3 --subtask 0 --multi-dt 3

python maml-visual.py --epoch 50000 --batch 5 --save-name maml-softmax2d-task2 --inner-lr 0.005 --outer-lr 0.001 --loss-factor 100 --adaptation-steps 1 --interval 3 --subtask 1 --multi-dt 3