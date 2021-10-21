### Rainbow RBFDQN

Run flags: 

`--seed <seed_num>`

`--experiment_name <exp_name>` 
Usually use something like `results/experiment1`, to create a new folder in the results root directory. 

`--run_title <run_title>`
Creates a sub-folder under the results/<exp_name>/ directory and stores all the logs, hyperparameters, and meta logger information here. 

`--log` 
Use this flag if you want to store model checkpoint files for the network & target network. Files are stored under `results/<exp_name>/<run_title>/logs`

`--double <True / False>` 
If you want to use double DQN

`--nstep <step_size>`
If you want to use multi step returns, by default 1 is 1-step updates

`--dueling <True / False>` 
If you want to use a dueling architecture.

`--mean <True / False>` 
By default the dueling architecture uses the max combine operator to merge the base value and advantage values. You can use the mean as the combine operator with this flag (default is max)

`--layer_normalization <True / False>`
If you want to apply layer normalization on hidden layers, before the activation function

`--noisy_layers <True / False>`
If you want noisy linear layers to be used in place of linear layers, and use parameter noise for the exploration strategy

`--distributional <True / False> `
If you want to use distributional RBF-DQN 

`--reward_norm <clip / max> `
Select how to normalize your rewards

`--per <True / False>` 
Use priority experience replay

`--alpha <float>`
The value of alpha to use with per, controls what proportion of samples are selected greedily (based on priority) vs uniformly. Default value of 0.1. 

`--per_beta_start <float>` 
The starting value for beta in PER (controls the importance sampling compensation, 1 for full compensation, 0 for no compensation). Defaults to 0.4

`--should_schedule_beta <True / False>`
Whether or not to anneal the value of beta to go from `per_beta_start` to 1.0 over the course of training. Defaults to true. 

`--random_betas <True / False>`
Whether or not to use a randomly initialized beta for each centroid index.

### Example Run Commands
`python experiments/experiment.py --hyper_parameter_name 10 --seed 0 --experiment_name "./results/<TASK_NAME>" --run_title "<VARIATION>" --double True --per True --nstep 4 --dueling True --noisy_layers True`

### Onager to run sweeps 

`onager prelaunch +jobname rainbow_experiment1 +command "python experiments/experiment.py --experiment_name ./results/testing" +arg --seed 0 +arg --double True False +arg --nstep 1 2 3 4 5 +arg --dueling True False +arg --layer_normalization True False +arg --noisy_layers True False +arg --run_title +tag run_title`
