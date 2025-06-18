############################Non-Novel Distractor############################
# Run this command first to train a model from scratch (non-Novel distractor)
python MiLa.py --epoch 50000 --batch 10 --save-name MiLa-nonNovel

# Run this command first to train a model from scratch (non-Novel distractor)
#   and we do not add noise on evaluation set, and we correctly set .train() and .val() method
python MiLa-correct-train-and-val.py --epoch 50000 --batch 10 --save-name MiLa-MAML-Model-nonNovel

# Run this command first to train a model from scratch (non-Novel distractor), with Model be as consistent as MAML
#   and we do not add noise on evaluation set, and we correctly set .train() and .val() method
python MiLa-MAML-Model.py --epoch 50000 --batch 10 --save-name MiLa-MAML-Model-nonNovel

# After training, run this command to evaluate the above model's performance
python eval-MiLa.py --epoch 50000 --batch 10 --save-name tmp
############################################################################


##############################Novel Distractor##############################
# Run this command to fine-tune a non-Novel model (Novel distractor)
python MiLa-Novel.py --epoch 50000 --batch 10 --save-name MiLa-Novel

# After training, run this command to evaluate the above model's performance
python eval-MiLa-Novel.py --epoch 50000 --batch 10 --save-name tmp
############################################################################