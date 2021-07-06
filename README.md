#### A few example of run commands

`python experiments/experiment1.py --hyper_parameter_name 00 --experiment_name ./results/testing --run_title testing --learning_rate 100.0`

Run only a Double RBF-DQN experiment: 
python experiments/experiment.py --hyper_parameter_name 00 --experiment_name ./results/test --run_title test --double

Run a Double & Nstep returns experiment with step size 3:

python experiments/experiment.py --hyper_parameter_name 00 --experiment_name ./results/test --run_title test --double --nstep 3
