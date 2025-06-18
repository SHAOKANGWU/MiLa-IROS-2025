# TODO: Below command is to train GCSL-predict-next-state, and in validation we do add noise
#python GCSL.py --epoch 50000 --batch 10 --save-name GCSL --sample-size 40 --interval 3 --lr 0.0001

# TODO: Below command is to train GCSL-increment (delta state), and in validation we do add noise
python GCSL.py --epoch 50000 --batch 10 --save-name GCSL-increment --sample-size 40 --down-sample-factor 2 --interval 3 --lr 0.0005


# TODO: Below command is to train GCSL-increment-MAML-Model (delta state), and in validation we do not add noise
python GCSL-MAML-Model.py --epoch 50000 --batch 10 --save-name GCSL-increment-MAML-Model --sample-size 40 --down-sample-factor 2 --interval 3 --lr 0.0005