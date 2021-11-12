import argparse

import pandas
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--algo",  default='ppo', help="name of algorithm (REQUIRED)")
parser.add_argument("--trained_env", default='ThreeRoom',
                    help="name of the env used to train (default: 'ThreeRoom')")
parser.add_argument("--eval_env", default='FourRoom',
                    help="name of the env (default: 'FourRoom')")
args = parser.parse_args()


def mean_smooth(input_data, smoothness):
    return pandas.Series(list(input_data)).rolling(smoothness, min_periods=5).mean()


def plot_evaluation(algorithm, logs, save):
    plt.rc('font', family='Times New Roman', size=12)
    plt.figure()
    plt.plot(logs["episode"], logs["return_per_episode"], 'b', alpha=0.2)
    plt.plot(logs["episode"], mean_smooth(logs["return_per_episode"], 20), label='PPO', color='blue')
    plt.title('Evaluation Algorithm: {} Metric: Return'.format(algorithm))
    plt.legend(loc='right')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.savefig('plots/{}/evaluation_return.jpg'.format(save), bbox_inches='tight', pad_inches=0.1)
    plt.show()


csv_ppo = pandas.read_csv('results/evaluation_results/ppo_1/log.csv')
csv_ppo_2 = pandas.read_csv('results/evaluation_results/ppo_2/log.csv')
csv_ppo_4 = pandas.read_csv('results/evaluation_results/ppo_4/log.csv')

csv_a2c = pandas.read_csv('results/evaluation_results/a2c_1/log.csv')
csv_a2c_2 = pandas.read_csv('results/evaluation_results/a2c_2/log.csv')
csv_a2c_4 = pandas.read_csv('results/evaluation_results/a2c_4/log.csv')

plt.rc('font', family='Times New Roman', size=12)
plt.figure()

if args.algo == 'ppo':
    csv_ppo_mean = csv_ppo["return_per_episode"].mean()
    csv_ppo_2_mean = csv_ppo_2["return_per_episode"].mean()
    csv_ppo_4_mean = csv_ppo_4["return_per_episode"].mean()
    plt.plot(csv_ppo["episode"], csv_ppo["return_per_episode"], 'b', alpha=0.1)
    plt.plot(csv_ppo["episode"], mean_smooth(csv_ppo["return_per_episode"], 20),
             label='PPO (return mean: %.1f)' % csv_ppo_mean, color='blue')
    plt.plot(csv_ppo_2["episode"], csv_ppo_2["return_per_episode"], 'r', alpha=0.1)
    plt.plot(csv_ppo_2["episode"], mean_smooth(csv_ppo_2["return_per_episode"], 20),
             label='PPO+LSTM 2 (return mean: %.1f)' % csv_ppo_2_mean, color='red')
    plt.plot(csv_ppo_4["episode"], csv_ppo_4["return_per_episode"], 'burlywood', alpha=0.1)
    plt.plot(csv_ppo_4["episode"], mean_smooth(csv_ppo_4["return_per_episode"], 20),
             label='PPO+LSTM 4 (return mean: %.1f)' % csv_ppo_4_mean, color='burlywood')
    plt.axhline(y=csv_ppo_mean, c="b", ls="-.", lw=1)
    plt.axhline(y=csv_ppo_2_mean, c="r", ls="--", lw=1)
    plt.axhline(y=csv_ppo_4_mean, c="burlywood", ls=":", lw=1)
    plt.title('PPO Env:4-room  Metric: Return')
    plt.legend(loc='best')
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Return')
    plt.savefig('plots/{}/{}_evaluation_return.jpg'.format('PPO', 'ppo'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
elif args.algo == 'a2c':
    csv_a2c_mean = csv_a2c["return_per_episode"].mean()
    csv_a2c_2_mean = csv_a2c_2["return_per_episode"].mean()
    csv_a2c_4_mean = csv_a2c_4["return_per_episode"].mean()
    plt.plot(csv_a2c["episode"], csv_a2c["return_per_episode"], 'green', alpha=0.1)
    plt.plot(csv_a2c["episode"], mean_smooth(csv_a2c["return_per_episode"], 20),
             label='A2C (return mean: %.1f)' % csv_a2c_mean, color='green')
    plt.plot(csv_a2c_2["episode"], csv_a2c_2["return_per_episode"], 'maroon', alpha=0.1)
    plt.plot(csv_a2c_2["episode"], mean_smooth(csv_a2c_2["return_per_episode"], 20),
             label='A2C+LSTM 2 (return mean: %.1f)' % csv_a2c_2_mean, color='maroon')
    plt.plot(csv_a2c_4["episode"], csv_a2c_4["return_per_episode"], 'c', alpha=0.1)
    plt.plot(csv_a2c_4["episode"], mean_smooth(csv_a2c_4["return_per_episode"], 20),
             label='A2C+LSTM 4 (return mean: %.1f)' % csv_a2c_4_mean, color='c')
    plt.axhline(y=csv_a2c_mean, c="green", ls="-.", lw=1)
    plt.axhline(y=csv_a2c_2_mean, c="maroon", ls="--", lw=1)
    plt.axhline(y=csv_a2c_4_mean, c="c", ls=":", lw=1)

    plt.title('A2C Env:4-room  Metric: Return')
    plt.legend(loc='best')
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Return')
    plt.savefig('plots/{}/{}_evaluation_return.jpg'.format('A2C', 'a2c'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

