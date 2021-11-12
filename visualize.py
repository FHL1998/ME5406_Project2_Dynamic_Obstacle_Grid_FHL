import argparse

import gym
import numpy

import utils
# from custom_env.register import register
from utils import device
from gym.envs.registration import register

# Parse arguments
global frames
parser = argparse.ArgumentParser()
parser.add_argument("--env", default='FourRoom', required=True,
                    help="name of the env (default: 'FourRoom')")
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename(e.g. './storage')")
parser.add_argument("--episodes", type=int, default=5,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--recurrence", type=int, default=1,
                    help="Set this value only if --memory is set to True.")

args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
print(f"Device: {device}\n")
register(
    id='FourRooms-Dynamic-Obstacles-21x21-v0',
    entry_point='custom_env.env.env:FourRoomsDynamicObstaclesEnv21x21',
    reward_threshold=0.95
)

register(
    id='ThreeRooms-Dynamic-Obstacles-21x21-v0',
    entry_point='custom_env.env.env:ThreeRoomsDynamicObstaclesEnv21x21',
    reward_threshold=0.95
)

# Load environment
if args.env == 'FourRoom':
    env = gym.make('FourRooms-Dynamic-Obstacles-21x21-v0')
    env.seed(args.seed)
else:
    env = gym.make('ThreeRooms-Dynamic-Obstacles-21x21-v0')
    env.seed(args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
default_storage_name = f"{args.algo}_{args.recurrence}"
model_dir = './storage/ThreeRoom/{}'.format(default_storage_name)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory)
print("Agent loaded\n")

# Run the agent
if args.gif:
    from array2gif import write_gif
    frames = []

# Create a window to view the environment
env.render('human')
for episode in range(args.episodes):
    obs = env.reset()

    while True:
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            env.reset()
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
    print("Done.")
