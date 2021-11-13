import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas

# frame = pandas.read_csv('storage/ppo/log.csv', engine='python')
# data = frame.drop_duplicates(subset=['update'], keep='first', inplace=False)
# data.to_csv('log1.csv', encoding='utf8', index=False)


parser = argparse.ArgumentParser()
parser.add_argument('--exps', nargs='+', type=str, default='storage')
parser.add_argument('--metric', type=str, default='return')
parser.add_argument('--save', type=str, default='return')
parser.add_argument('--smoothness', type=int, default=500)
args = parser.parse_args()


def mean_smooth(input_data):
    return pandas.Series(list(input_data)).rolling(args.smoothness, min_periods=5).mean()


plt.rc('font', family='Times New Roman', size=12)
f, ax = plt.subplots(1, 1)
log_filename = os.path.join(args.exps, 'ppo', 'log.csv')
# csv = pandas.read_csv(log_filename)
csv_ppo = pandas.read_csv('results/csvs/ppo.csv')
csv_ppo_lstm2 = pandas.read_csv('results/csvs/ppo_lstm2.csv')
csv_ppo_lstm4 = pandas.read_csv('results/csvs/ppo_lstm4.csv')
csv_a2c = pandas.read_csv('results/csvs/a2c.csv')
csv_a2c_lstm2 = pandas.read_csv('results/csvs/a2c_lstm2.csv')
csv_a2c_lstm4 = pandas.read_csv('results/csvs/a2c_lstm4.csv')
# csv_ppo = pandas.read_csv('results/csvs/ppo.csv')
# ax.plot(csv['frames'], csv['policy_loss']-0.01*csv['entropy']+0.5*csv['value_loss'], color='blue', alpha=0.2)
# ax.plot(csv['frames'], mean_smooth(csv['policy_loss']-0.01*csv['entropy']+0.5*csv['value_loss']), color='blue',
# label='entropy loss')
# ax.plot(csv['frames'], csv['entropy'], color='red', alpha=0.2)
# ax.plot(csv['frames'], mean_smooth(csv['entropy']), label='PPO', color='red')
# ax.title('Algorithm: PPO Metric: Value')
# ax.plot(csv['frames'], csv['policy_loss'], color='red', alpha=0.2)
# ax.plot(csv['frames'], mean_smooth(csv['policy_loss']), color='red', label='policy loss')
# ax.plot(csv['frames'], csv['value_loss'], color='green', alpha=0.2)
# ax.plot(csv['frames'], mean_smooth(csv['value_loss']), color='green', label='policy loss')
ax.plot(csv_ppo['frames'], csv_ppo['policy_loss'], color='green', alpha=0.15)
ax.plot(csv_ppo_lstm2['frames'], csv_ppo_lstm2['policy_loss'], color='blue', alpha=0.15)
ax.plot(csv_ppo_lstm4['frames'], csv_ppo_lstm4['policy_loss'], color='purple', alpha=0.15)
ax.plot(csv_ppo['frames'], mean_smooth(csv_ppo['policy_loss']), color='green', label='PPO')
ax.plot(csv_ppo_lstm2['frames'], mean_smooth(csv_ppo_lstm2['policy_loss']), color='blue', label='PPO+LSTM:2')
ax.plot(csv_ppo_lstm4['frames'], mean_smooth(csv_ppo_lstm4['policy_loss']), color='purple', label='PPO+LSTM:4')

print('csv_ppo', csv_ppo['return_mean'].mean())
print('csv_ppo_lstm2', csv_ppo_lstm2['return_mean'].mean())
print('csv_ppo_lstm4', csv_ppo_lstm4['return_mean'].mean())
# ax.fill_between(csv_ppo['frames'], mean_smooth((csv_ppo['return_mean'] - csv_ppo['return_std'])),
#                 mean_smooth((csv_ppo['return_mean'] + csv_ppo['return_std'])),
#                 color='green', alpha=0.1)
#
# ax.fill_between(csv_ppo_lstm2['frames'], mean_smooth((csv_ppo_lstm2['return_mean'] - csv_ppo_lstm2['return_std'])),
#                 mean_smooth((csv_ppo_lstm2['return_mean'] + csv_ppo_lstm2['return_std'])),
#                 color='blue', alpha=0.1)
#
# ax.fill_between(csv_ppo_lstm4['frames'], mean_smooth((csv_ppo_lstm4['return_mean'] - csv_ppo_lstm4['return_std'])),
#                 mean_smooth((csv_ppo_lstm4['return_mean'] + csv_ppo_lstm4['return_std'])),
#                 color='purple', alpha=0.1)

# ax.plot(csv_a2c['frames'], mean_smooth(csv_a2c['return_mean']), color='green', label='A2C')
# ax.plot(csv_a2c_lstm2['frames'], mean_smooth(csv_a2c_lstm2['return_mean']), color='blue', label='A2C+LSTM:2')
# ax.plot(csv_a2c_lstm4['frames'], mean_smooth(csv_a2c_lstm4['return_mean']), color='purple', label='A2C+LSTM:4')

# ax.fill_between(csv_a2c['frames'], mean_smooth((csv_a2c['return_mean'] - csv_a2c['return_std'])),
#                 mean_smooth((csv_a2c['return_mean'] + csv_a2c['return_std'])),
#                 color='green', alpha=0.15)
#
# ax.fill_between(csv_a2c_lstm2['frames'], mean_smooth((csv_a2c_lstm2['return_mean'] - csv_a2c_lstm2['return_std'])),
#                 mean_smooth((csv_a2c_lstm2['return_mean'] + csv_a2c_lstm2['return_std'])),
#                 color='blue', alpha=0.15)
#
# ax.fill_between(csv_a2c_lstm4['frames'], mean_smooth((csv_a2c_lstm4['return_mean'] - csv_a2c_lstm4['return_std'])),
#                 mean_smooth((csv_a2c_lstm4['return_mean'] + csv_a2c_lstm4['return_std'])),
#                 color='purple', alpha=0.15)


ax.set_title('Comparison Metric: Policy Loss')
ax.legend(loc='right')
ax.set_xlabel('Update Frames')
ax.set_ylabel('Policy Loss')

if args.save:
    os.makedirs('plots', exist_ok=True)
    f.savefig(os.path.join('plots', 'ppo_compare_policy_loss' + '.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)
else:
    plt.show()
