### Rainbow RBFDQN

Run flags: 

`--seed <seed_num>`

`--experiment_name <exp_name>` 
Usually use something like `results/experiment1`, to create a new folder in the results root directory. 

`--run_title <run_title>`
Creates a sub-folder under the results/<exp_name>/ directory and stores all the logs, hyperparameters, and meta logger information here. 

`--log` 
Use this flag if you want to store model checkpoint files for the network & target network. Files are stored under `results/<exp_name>/<run_title>/logs`

`--double` 
If you want to use double DQN

`--nstep <step_size>`
If you want to use multi step returns

`--dueling` 
If you want to use a dueling architecture.

`--mean` 
By default the dueling architecture uses the max combine operator to merge the base value and advantage values. You can use the mean as the combine operator with this flag (default is max)

`--layer_normalization`
If you want to apply layer normalization on hidden layers, before the activation function

`--noisy_layers`
If you want noisy linear layers to be used in place of linear layers, and use parameter noise for the exploration strategy


### Example Run Commands
`python experiments/experiment.py --hyper_parameter_name 00 --experiment_name ./results/test --run_title test --double`

`python experiments/experiment.py --hyper_parameter_name 00 --experiment_name ./results/test --run_title test --double --nstep 3`

`python experiments/experiment.py --hyper_parameter_name 00 --experiment_name ./results/test --run_title test --per --dueling`
