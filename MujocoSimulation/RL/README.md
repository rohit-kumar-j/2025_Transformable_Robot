# Running the RL simulations

## Prerequisites
```bash
python3 -m venv <env_name> # Create venv if necessary
source <env_name>/bin/activate # Activate the env
pip install -r requirements.txt # Install requirements
```

## Running the simulation
```bash
which python # check python executable path to make sure that it is the correct env

#train
cd MujocoSimulation/RL/code
python train.py
python test.py # to view the results
```

## Viewing results with tensorboard
In another terminal run:
```bash
source <env_name>/bin/activate # Activate the env
cd MujocoSimulation/RL/code 
tensorboard --logdir ./ppo_tensorboard
```
## Structure
The code creates 2 directories, `MujocoSimulation/RL/code/ppo_chceckpoints` to store the checkpoints every 100,000 iterations and a 
`MujocoSimulation/RL/code/ppo_tensorboard` directory to log the results.

## **NOTE**
Graphs in tensorboard will appear only after starting the train cycle

