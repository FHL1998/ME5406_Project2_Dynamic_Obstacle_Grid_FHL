import argparse

import pandas
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--algo", default='a2c', help="name of algorithm (REQUIRED)")
args = parser.parse_args()


def mean_smooth(input_data):
    return pandas.Series(list(input_data)).rolling(30, min_periods=5).mean()


# csv_ppo = pandas.read_csv('results/evaluation_results/FourRoom/ppo_1/log.csv')
# csv_ppo_log_argmax = pandas.read_csv('results/evaluation_results/FourRoom/ppo_1/argmax/log.csv')
# csv_ppo_lstm2 = pandas.read_csv('results/evaluation_results/FourRoom/ppo_2/log.csv')
# csv_ppo_lstm2_log_argmax = pandas.read_csv('results/evaluation_results/FourRoom/ppo_2/argmax/log.csv')
# csv_ppo_lstm4 = pandas.read_csv('results/evaluation_results/FourRoom/ppo_4/log.csv')
# csv_ppo_lstm4_log_argmax = pandas.read_csv('results/evaluation_results/FourRoom/ppo_4/argmax/log.csv')
csv_ppo = pandas.read_csv('results/evaluation_results/ThreeRoom/ppo_1/log.csv')
csv_ppo_log_argmax = pandas.read_csv('results/evaluation_results/ThreeRoom/ppo_1/argmax/log.csv')
csv_ppo_lstm2 = pandas.read_csv('results/evaluation_results/ThreeRoom/ppo_2/log.csv')
csv_ppo_lstm2_log_argmax = pandas.read_csv('results/evaluation_results/ThreeRoom/ppo_2/argmax/log.csv')
csv_ppo_lstm4 = pandas.read_csv('results/evaluation_results/ThreeRoom/ppo_4/log.csv')
csv_ppo_lstm4_log_argmax = pandas.read_csv('results/evaluation_results/ThreeRoom/ppo_4/argmax/log.csv')
csv_ppo_mean = csv_ppo["return_per_episode"].mean()
csv_ppo_log_argmax_mean = csv_ppo_log_argmax["return_per_episode"].mean()
csv_ppo_lstm2_mean = csv_ppo_lstm2["return_per_episode"].mean()
csv_ppo_lstm2_log_argmax_mean = csv_ppo_lstm2_log_argmax["return_per_episode"].mean()
csv_ppo_lstm4_mean = csv_ppo_lstm4["return_per_episode"].mean()
csv_ppo_lstm4_log_argmax_mean = csv_ppo_lstm4_log_argmax["return_per_episode"].mean()

csv_a2c = pandas.read_csv('results/evaluation_results/ThreeRoom/a2c_1/log.csv')
csv_a2c_log_argmax = pandas.read_csv('results/evaluation_results/ThreeRoom/a2c_1/argmax/log.csv')
csv_a2c_lstm2 = pandas.read_csv('results/evaluation_results/ThreeRoom/a2c_2/log.csv')
csv_a2c_lstm2_log_argmax = pandas.read_csv('results/evaluation_results/ThreeRoom/a2c_2/argmax/log.csv')
csv_a2c_lstm4 = pandas.read_csv('results/evaluation_results/ThreeRoom/a2c_4/log.csv')
csv_a2c_lstm4_log_argmax = pandas.read_csv('results/evaluation_results/ThreeRoom/a2c_4/argmax/log.csv')
csv_a2c_mean = csv_a2c["return_per_episode"].mean()
csv_a2c_log_argmax_mean = csv_a2c_log_argmax["return_per_episode"].mean()
csv_a2c_lstm2_mean = csv_a2c_lstm2["return_per_episode"].mean()
csv_a2c_lstm2_log_argmax_mean = csv_a2c_lstm2_log_argmax["return_per_episode"].mean()
csv_a2c_lstm4_mean = csv_a2c_lstm4["return_per_episode"].mean()
csv_a2c_lstm4_log_argmax_mean = csv_a2c_lstm4_log_argmax["return_per_episode"].mean()

plt.rc('font', family='Times New Roman', size=12)
plt.figure()

if args.algo == 'ppo':
    plt.plot(csv_ppo["episode"], csv_ppo["return_per_episode"], 'b', alpha=0.1)
    plt.plot(csv_ppo["episode"], mean_smooth(csv_ppo["return_per_episode"]),
             label='PPO (return mean: %.1f)' % csv_ppo_mean, color='blue')

    plt.plot(csv_ppo_log_argmax["episode"], csv_ppo_log_argmax["return_per_episode"], 'b', alpha=0.05)
    plt.plot(csv_ppo_log_argmax["episode"], mean_smooth(csv_ppo_log_argmax["return_per_episode"]),
             label='PPO argmax (return mean: %.1f)' % csv_ppo_log_argmax_mean, color='blue', alpha=0.5)

    plt.plot(csv_ppo_lstm2["episode"], csv_ppo_lstm2["return_per_episode"], 'g', alpha=0.1)
    plt.plot(csv_ppo_lstm2["episode"], mean_smooth(csv_ppo_lstm2["return_per_episode"]),
             label='PPO+LSTM 2 (return mean: %.1f)' % csv_ppo_lstm2_mean, color='g')

    plt.plot(csv_ppo_lstm2_log_argmax["episode"], csv_ppo_lstm2_log_argmax["return_per_episode"], 'g', alpha=0.05)
    plt.plot(csv_ppo_lstm2_log_argmax["episode"], mean_smooth(csv_ppo_lstm2_log_argmax["return_per_episode"]),
             label='PPO+LSTM 2 argmax (return mean: %.1f)' % csv_ppo_lstm2_log_argmax_mean, color='g', alpha=0.5)

    plt.plot(csv_ppo_lstm4["episode"], csv_ppo_lstm2["return_per_episode"], 'r', alpha=0.1)
    plt.plot(csv_ppo_lstm4["episode"], mean_smooth(csv_ppo_lstm2["return_per_episode"]),
             label='PPO+LSTM 4 (return mean: %.1f)' % csv_ppo_lstm4_mean, color='r')

    plt.plot(csv_ppo_lstm4_log_argmax["episode"], csv_ppo_lstm4_log_argmax["return_per_episode"], 'r', alpha=0.05)
    plt.plot(csv_ppo_lstm4_log_argmax["episode"], mean_smooth(csv_ppo_lstm4_log_argmax["return_per_episode"]),
             label='PPO+LSTM 4 argmax (return mean: %.1f)' % csv_ppo_lstm4_log_argmax_mean, color='r', alpha=0.5)

    plt.title('PPO(argmax)  Env:3-room  Metric: Return')
    plt.legend(loc='best')
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Return')
    plt.savefig('plots/{}/{}_3_room_evaluation_return.jpg'.format('PPO', 'ppo'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
elif args.algo == 'a2c':
    plt.plot(csv_a2c["episode"], csv_a2c["return_per_episode"], 'b', alpha=0.1)
    plt.plot(csv_a2c["episode"], mean_smooth(csv_a2c["return_per_episode"]),
             label='A2C (return mean: %.1f)' % csv_a2c_mean, color='blue')

    plt.plot(csv_a2c_log_argmax["episode"], csv_a2c_log_argmax["return_per_episode"], 'b', alpha=0.05)
    plt.plot(csv_a2c_log_argmax["episode"], mean_smooth(csv_a2c_log_argmax["return_per_episode"]),
             label='A2C argmax (return mean: %.1f)' % csv_a2c_log_argmax_mean, color='blue', alpha=0.5)

    plt.plot(csv_a2c_lstm2["episode"], csv_a2c_lstm2["return_per_episode"], 'g', alpha=0.1)
    plt.plot(csv_a2c_lstm2["episode"], mean_smooth(csv_a2c_lstm2["return_per_episode"]),
             label='A2C+LSTM 2 (return mean: %.1f)' % csv_a2c_lstm2_mean, color='g')

    plt.plot(csv_a2c_lstm2_log_argmax["episode"], csv_a2c_lstm2_log_argmax["return_per_episode"], 'g', alpha=0.05)
    plt.plot(csv_a2c_lstm2_log_argmax["episode"], mean_smooth(csv_a2c_lstm2_log_argmax["return_per_episode"]),
             label='A2C+LSTM 2 argmax (return mean: %.1f)' % csv_a2c_lstm2_log_argmax_mean, color='g', alpha=0.5)

    plt.plot(csv_a2c_lstm4["episode"], csv_a2c_lstm2["return_per_episode"], 'r', alpha=0.1)
    plt.plot(csv_a2c_lstm4["episode"], mean_smooth(csv_a2c_lstm2["return_per_episode"]),
             label='A2C+LSTM 4 (return mean: %.1f)' % csv_a2c_lstm4_mean, color='r')

    plt.plot(csv_a2c_lstm4_log_argmax["episode"], csv_a2c_lstm4_log_argmax["return_per_episode"], 'r', alpha=0.05)
    plt.plot(csv_a2c_lstm4_log_argmax["episode"], mean_smooth(csv_a2c_lstm4_log_argmax["return_per_episode"]),
             label='A2C+LSTM 4 argmax (return mean: %.1f)' % csv_a2c_lstm4_log_argmax_mean, color='r', alpha=0.5)

    plt.title('A2C(argmax)  Env:3-room  Metric: Return')
    plt.legend(loc='best')
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Return')
    plt.savefig('plots/{}/{}_3_room_evaluation_return.jpg'.format('A2C', 'a2c'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
