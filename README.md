# crowdsource_toxicity_classification

# Code Structure 

We provide the code for training a toxicity classifier and a worker quality estimator.

- `noisy_learning_question.py`: main script to train a toxicity classifier.
- `cus_trainer.py`: implement partial label learning methods (Jin et al. and PRODEN) and vanilla soft label method.
- `RL_trainer.py`: implement our method.
- `pm_*.py`: implement the participant-mine voting method.

# Key points in `RL_trainer.py`

- function `compute_loss`: implement soft label method with estimated worker weight.
- function `compute_valid_loss`: implement cross-entropy loss in the validation set.
- function `inner_training_loop` (line 1326 - line 1351): implement group DRO.

# Executing the program
Run `run_question.sh` to execute our program.


