############################Non-Novel Distractor############################
# Run this command first to train a model from scratch (non-Novel distractor)
python MiLa-NoGMM.py --epoch 50000 --batch 10 --save-name nonNovel

# After training, run this command to evaluate the above model's performance
python eval-MiLa-NoGMM.py --epoch 50000 --batch 10 --save-name tmp
############################################################################



##############################Novel Distractor##############################
# Run this command to fine-tune a non-Novel model (Novel distractor)
python MiLa-NoGMM-Novel.py --epoch 50000 --batch 10 --save-name MiLa-Novel
############################################################################