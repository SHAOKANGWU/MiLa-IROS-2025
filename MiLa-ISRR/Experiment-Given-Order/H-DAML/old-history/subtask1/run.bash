python MAML-bt-delta-1.py --epoch 50000 --batch 6 --save-name sub1 --inner-lr 0.00001 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3

python MAML-bt-delta-1-continue.py --epoch 50000 --batch 6 --save-name sub1-continue --inner-lr 0.00001 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3

python MAML-bt-delta-1-Novel.py --epoch 50000 --batch 6 --save-name sub1-Novel --inner-lr 0.00001 --outer-lr 0.0001 --loss-factor 10000 --adaptation-steps 1 --interval 3
