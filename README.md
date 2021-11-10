# ME5406_Project2_Dynamic_Obstacle_Grid_FHL ![maven](https://img.shields.io/badge/NUS-ME5406-important)
This repository contains the implementation of reinforcement learning algorithms like PPO and A2C, to solve the problem: Dynamic Obstacle Avoidance in Generalized Environment. And test the generalization and migration of the trained model using these algorithms.

## Project Description
> The objective of this project is to Deep Reinforcement Learning techniques to implement the **Dynamic Obstacle Avoidance in Generalized Environment**. 
> The problem is essentially a grid-world scenario in which the agent’s target is to go from the start point, go through the room by exit which was randomly setalong the wall, and reach the goal which set in another room, while avoiding crashing into dynamic obstacles in the environment. Meanwhile, the adding of the field of views enables the agent to have the ability of partial or fully observation. It has to be mentioned that the generalization ability oftrained model is tested during the process.

## Project Preparation ![maven](https://img.shields.io/badge/Project-Preparation-important)
 ### Virtual Environment Creation
 First, create the virtual environment using Anoconda and activate the created environment in Ubuntu 18.04.
 
```python
$ conda create -n obstacle_grid python=3.6
$ source activate obstacle_grid
```

 ### Requirements Install ![maven](https://img.shields.io/badge/Python-3.6-important) ![maven](https://img.shields.io/badge/Python-Requirements-important)
The project is based on the python version `Python 3.6.8`. For the requirements, a new virtual environmrnt is recommended. You should install the required packages in `requirements.txt` using:
```python
pip install -r requirements.txt
```

## Project Execution ![maven](https://img.shields.io/badge/Project-Execution-important)
The main scripts of the project are: `train.py`, `evaluate.py`, and `visualize.py`. For the detailed usage please refer to the parser in the corresponding files. The example of training, evaluation and visualization can be illustrated as:
```python
python train.py --env 'ThreeRoom' --algo ppo  --frames-per-proc 128
```
