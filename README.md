# Imitation Learning for Gamette Players

This repository uses the implementation of [Multi-Modal Imitation Learning in Partially Observable Environments](https://github.com/MarkFzp/infogail-pomdp) and adapts it for imitating the behavior of [Gamette](https://dl.acm.org/doi/abs/10.1145/3313831.3376571) players in supply chain decisions. 

The code uses gameplay trajectories and decisions of human players who interacted with the [CRISP](https://gitlab.com/syifan/crisp) simulation through a Gamette environment. For the imitation learning task, the repository relies on a partially observable environment called *gym-crisp* which I developed using [OpenAI's Gym](https://gym.openai.com/) toolkit. 

## Dependencies

Follow the dependencies in [Multi-Modal Imitation Learning in Partially Observable Environments](https://github.com/MarkFzp/infogail-pomdp).

## Run Imitation Learning

All hyperparameters can be configured in `config.py` file. After adjusting the hyperparameters and other variables, simply run:

```
python3 train.py
```

## Solving for Optimal Policy 

To solve for optimal policy using PPO2 solver run:

```
cd expert
python3 solve.py
```

the configuration for the solver is located in `expert/config.py`