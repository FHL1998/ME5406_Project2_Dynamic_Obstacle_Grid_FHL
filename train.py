# coding:utf-8
import argparse
import time
import datetime
import tensorboardX
import sys

import gym

import algorithms
import wandb

import utils
from gym.envs.registration import register
import torch
from model import ACModel

device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# General parameters
parser.add_argument("--env", default='ThreeRoom', required=True,
                    help="name of the env (default: 'ThreeRoom')")
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=8,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=1.5 * 10 ** 7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument('--wandb-project-name', type=str, default="me5406",
                    help="the wandb's project name")
parser.add_argument('--wandb-entity', type=str, default="fhl1998",
                    help="the entity of wandb's project")
parser.add_argument('--prod-mode', type=bool, default=False,
                    help='run the script in production mode and use wandb to log outputs')

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None, required=True,
                    help="number of frames per process before update (default: 8 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001/0.00085)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="gradient norm clipping coefficient (default: 0.5) ")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is back-propagated (default: 1). If > 1, a LSTM is added to "
                         "the model to have memory.")

args = parser.parse_args()

args.mem = args.recurrence > 1

if __name__ == '__main__':
    if args.env == 'ThreeRoom':
        env = gym.make('ThreeRooms-Dynamic-Obstacles-21x21-v0')
    else:
        env = gym.make('FourRooms-Dynamic-Obstacles-21x21-v0')

    if args.prod_mode:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args),
                   monitor_gym=True, save_code=True)

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_storage_name = f"{args.algo}_{args.recurrence}"
    model_dir = './storage/{}/{}'.format(args.env, default_storage_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        env.seed(args.seed + 100 * i)
        envs.append(env)
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space_shape, preprocess_observation = utils.get_obss_preprocessor(envs[0].observation_space)
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    actor_critic_model = ACModel(obs_space_shape, envs[0].action_space, args.mem)
    if "model_state" in status:
        actor_critic_model.load_state_dict(status["model_state"])
    actor_critic_model.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(actor_critic_model))

    # Load algo

    if args.algo == "a2c":
        algo = algorithms.A2CAlgo(envs, actor_critic_model, device, args.frames_per_proc, args.discount, args.lr,
                                  args.gae_lambda,
                                  args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                  args.optim_alpha, args.optim_eps, preprocess_observation)
    elif args.algo == "ppo":
        algo = algorithms.PPOAlgo(envs, actor_critic_model, device, args.frames_per_proc, args.discount, args.lr,
                                  args.gae_lambda,
                                  args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                  args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_observation)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    return_per_episode_dict = {}
    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    # args.frames: number of frames of training here args.frames 代表 total frames number
    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()

        # num_frames 返回的是 4*128=512 frames
        exps, logs1 = algo.collect_experiences()
        # print('LOGS1', logs1)
        # print('distribution', logs1["dist"])
        logs2 = algo.update_parameters(exps)
        # print('LOGS2', logs2)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            # print('logs["return_per_episode"]', logs["return_per_episode"])
            # calculation of mean, std, min, max
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
            for keys in return_per_episode.keys():
                return_per_episode_dict[keys] = return_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]

            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | Duration {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} "
                "| H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(*data))

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": actor_critic_model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
