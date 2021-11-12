import argparse
import time
import gym
import torch

from algorithms.utils.penv import ParallelEnv

import utils
# from utils import device
from gym.envs.registration import register

from utils.figure import mean_smooth

device = "cpu"
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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trained_env", default='ThreeRoom',
                    help="name of the env used to train (default: 'ThreeRoom')")
parser.add_argument("--eval_env", default='FourRoom', required=True,
                    help="name of the env (default: 'FourRoom')")
parser.add_argument("--algo", required=True,
                    help="name of algorithm (REQUIRED)")
parser.add_argument("--episodes", type=int, default=200,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=6,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=True,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--recurrence", type=int, default=1,
                    help="Set this value only if --memory is set to True.")


args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
print(f"Device: {device}\n")

if __name__ == '__main__':

    # Load environments
    if args.eval_env == 'FourRoom':
        env = gym.make('FourRooms-Dynamic-Obstacles-21x21-v0')
    else:
        env = gym.make('ThreeRooms-Dynamic-Obstacles-21x21-v0')
    envs = []
    for i in range(args.procs):
        env.seed(args.seed + 1000 * i)
        envs.append(env)
    print('envs', envs)
    env = ParallelEnv(envs)
    obs_space_shape, preprocess_observation = utils.get_obss_preprocessor(envs[0].observation_space)
    print("Environments loaded\n")

    # Load agent
    default_storage_name = f"{args.algo}_{args.recurrence}"
    print('default_storage_name', default_storage_name)
    model_dir = './storage/{}/{}'.format(args.trained_env, default_storage_name)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs, use_memory=args.memory)
    print("Agent loaded\n")

    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    observation = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)
    default_storage_name = f"{args.algo}_{args.recurrence}"

    while log_done_counter < args.episodes:
        actions = agent.get_actions(observation)
        observation, rewards, done, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, done)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(done):
            if done:
                log_done_counter += 1
                print('evaluation episode {}'.format(log_done_counter))
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

        end_time = time.time()

        # Print logs
        if log_done_counter >= 1:
            num_frames = sum(logs["num_frames_per_episode"])
            fps = num_frames / (end_time - start_time)
            duration = int(end_time - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    # Print worst episodes
    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print(
                "episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
    if args.argmax:
        csv_file, csv_logger = utils.get_csv_logger('results/evaluation_results/{}/{}/argmax'.
                                                    format(args.eval_env,default_storage_name))
    else:
        csv_file, csv_logger = utils.get_csv_logger('results/evaluation_results/{}/{}'.
                                                    format(args.eval_env, default_storage_name))
    header = ["episode", "return_per_episode", "num_frames_per_episode"]
    csv_logger.writerow(header)
    for i in range(0, args.episodes):
        data = [i+1, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]]
        csv_logger.writerow(data)
        csv_file.flush()
