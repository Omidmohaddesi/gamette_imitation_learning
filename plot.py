import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
import os
from expert.config import config


def count_allocations(mydf):

    counts_df = pd.DataFrame(columns=['window'])

    for index, row in mydf.iterrows():
        counts_df = counts_df.append(pd.concat([row[list(range(21, 28))].value_counts(),    #/7,
                                                pd.Series({'window': '21-27'})]), ignore_index=True)
        counts_df = counts_df.append(pd.concat([row[list(range(28, 34))].value_counts(),    #/6,
                                                pd.Series({'window': '28-33'})]), ignore_index=True)
        counts_df = counts_df.append(pd.concat([row[list(range(34, 39))].value_counts(),    #/5,
                                                pd.Series({'window': '34-38'})]), ignore_index=True)
        counts_df = counts_df.append(pd.concat([row[list(range(39, 44))].value_counts(),    #/5,
                                                pd.Series({'window': '39-43'})]), ignore_index=True)
        counts_df = counts_df.append(pd.concat([row[list(range(44, 56))].value_counts(),    #/12,
                                                pd.Series({'window': '44-55'})]), ignore_index=True)

    counts_df = counts_df.fillna(0)

    return counts_df


def draw_plots(my_df, type, save_name, code_list=None, policy_list=None, deviation=None, disruption_start=28,
               disruption_end=33, title=None, ylabel=None, xlabel=None, outlier=True, hue=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    if type == 'agent_orders':
        # ax = sns.boxplot(x='variable', y='value', hue='code', data=my_df, showfliers=outlier)
        ax = sns.lineplot(x='variable', y='value', hue=hue, data=my_df)
        ax.set_ylim(bottom=-10, top=400)

    if type == 'agent_allocations':
        ax = sns.scatterplot(x='variable', y='value', hue=hue, data=my_df,
                             palette=['blue', 'orange', 'green'],
                             alpha=0.05)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(policy_list)
        ax.set_ylim(bottom=-0.1, top=2.1)

    if type == 'agent_rewards':
        ax = sns.boxplot(x='code', y=0, data=my_df[my_df[hue].isin(code_list)])

    if type == 'player_orders':
        # ax = sns.lineplot(x='variable', y='value', hue=my_df.index, data=my_df, ci=None)
        # ax = sns.lineplot(x='variable', y='value', data=my_df)
        ax = sns.boxplot(x='variable', y='value', data=my_df, showfliers=outlier)
        ax.set_ylim(bottom=-10, top=400)

    if type == 'ppo2_orders':
        ax = sns.lineplot(x='variable', y='value', data=my_df)
        ax.set_ylim(bottom=-10, top=400)

    if type == 'ppo2_allocations':
        ax = sns.scatterplot(x='variable', y='value', data=my_df, alpha=0.05)
        # my_df = my_df.groupby(['variable', 'value']).apply(lambda x: x['value'].count()).reset_index()
        # ax = sns.barplot(x='variable', y=0, hue='value', data=my_df)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(policy_list)
        ax.set_ylim(bottom=-0.1, top=2.1)

    if type == 'ppo2_allocation_time_window':
        df = my_df.pivot_table(values='value', index=['window', 'variable'], aggfunc='sum').reset_index()
        df['value'] = df['value'].transform(lambda x: x / x.sum())
        ax = sns.barplot(x='window', y='value', hue='variable', data=df)
        # ax.set_ylim(top=0.38)

    if type == 'ppo2_rewards':
        ax = sns.boxplot(data=my_df, showfliers=outlier)

    if type != 'agent_rewards' and type != 'ppo2_rewards':
        if type != 'player_orders' and type != 'ppo2_allocation_time_window':
            ax.set_xticks(list(range(21, 56)))
        ax.axvspan(disruption_start, disruption_end, alpha=0.5, color='pink', zorder=0)

        legend = ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='pink', label='Disruption'))
        labels.append('Disruption')
        legend._legend_box = None
        legend._init_legend_box(handles, labels)
        legend._set_loc(legend._loc)
        legend.set_title(legend.get_title().get_text())

    if type == 'ppo2_allocation_time_window':
        legend = ax.legend()
        handles, _ = ax.get_legend_handles_labels()
        legend._legend_box = None
        legend._init_legend_box(handles, policy_list)
        legend._set_loc(legend._loc)
        legend.set_title(legend.get_title().get_text())

    ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)

    fig.tight_layout()
    fig.savefig(f'plots/{save_name}.png', format="png", dpi=300)


order_data_0 = pd.read_csv(f'render/order_data_{config.condition}_code0.csv', header=None, names=list(range(21, 56)))
order_data_1 = pd.read_csv(f'render/order_data_{config.condition}_code1.csv', header=None, names=list(range(21, 56)))
order_data_2 = pd.read_csv(f'render/order_data_{config.condition}_code2.csv', header=None, names=list(range(21, 56)))
order_data_ppo2 = pd.read_csv(f'render/ppo2/order_data_{config.condition}.csv', header=None, names=list(range(21, 56)))

allocation_data_0 = pd.read_csv(f'render/allocation_data_{config.condition}_code0.csv', header=None, names=list(range(21, 56)))
allocation_data_1 = pd.read_csv(f'render/allocation_data_{config.condition}_code1.csv', header=None, names=list(range(21, 56)))
allocation_data_2 = pd.read_csv(f'render/allocation_data_{config.condition}_code2.csv', header=None, names=list(range(21, 56)))
allocation_data_ppo2 = pd.read_csv(f'render/ppo2/allocation_data_{config.condition}.csv', header=None, names=list(range(21, 56)))

rewards_data_0 = pd.read_csv(f'render/rewards_data_{config.condition}_code0.csv', header=None, names=list(range(21, 56)))
rewards_data_1 = pd.read_csv(f'render/rewards_data_{config.condition}_code1.csv', header=None, names=list(range(21, 56)))
rewards_data_2 = pd.read_csv(f'render/rewards_data_{config.condition}_code2.csv', header=None, names=list(range(21, 56)))
rewards_data_ppo2 = pd.read_csv(f'render/ppo2/rewards_data_{config.condition}.csv', header=None, names=list(range(21, 56)))

order_data = pd.concat([order_data_0, order_data_1, order_data_2],
                       keys=[0, 1, 2]).reset_index(level=0).rename(columns={'level_0': 'code'})
allocation_data = pd.concat([allocation_data_0, allocation_data_1, allocation_data_2],
                            keys=[0, 1, 2]).reset_index(level=0).rename(columns={'level_0': 'code'})
rewards_data = pd.concat([rewards_data_0.sum(axis=1), rewards_data_1.sum(axis=1), rewards_data_2.sum(axis=1)],
                         keys=[0, 1, 2]).reset_index(level=0).rename(columns={'level_0': 'code'})

order_melt = pd.melt(order_data, id_vars='code')
allocation_melt = pd.melt(allocation_data, id_vars='code')

order_ppo2_melt = pd.melt(order_data_ppo2)
allocation_ppo2_melt = pd.melt(allocation_data_ppo2)
rewards_ppo2 = rewards_data_ppo2.sum(axis=1)

# Loading players data:
players_order_data = pd.read_csv(
    os.path.join(os.path.abspath('../crisp_game_server/gamette_experiments/study_2/data_full'), 'order_data.csv'),
    index_col=0)
players_allocation_data = pd.read_csv(
    os.path.join(os.path.abspath('../crisp_game_server/gamette_experiments/study_2/data_full'), 'allocation_data.csv'),
    index_col=0)
players = pd.read_csv('expert/players.csv', index_col=0)

players_order_data = players_order_data[(players_order_data['player_id'].isin(players.values.ravel())) &
                                        (players_order_data['condition'] == config.condition)].reset_index(
    drop=True).drop(['player_id', 'condition'], axis=1)
players_allocation_data = players_allocation_data[(players_allocation_data['player_id'].isin(players.values.ravel())) &
                                                  (players_allocation_data['condition'] == config.condition)].\
    reset_index(drop=True).drop(['player_id', 'condition'], axis=1)

players_order_data.columns = players_order_data.columns.astype(int)
players_order_data_melt = pd.melt(players_order_data, ignore_index=False)

policies = ['Prefer HC1', 'Prefer HC2', 'Proportionally']

allocation_ppo2_time_count = count_allocations(allocation_data_ppo2)
allocation_ppo2_time_count_melt = pd.melt(allocation_ppo2_time_count, id_vars='window', value_vars=allocation_ppo2_time_count.columns[1:])

draw_plots(order_melt, 'agent_orders', f'agent_orders_Condition_{config.condition}',
           policy_list=None, disruption_start=28, disruption_end=33,
           title='Order Trajectories', ylabel='Order Amount', xlabel='Time (Week)', outlier=True, hue='code')

draw_plots(allocation_melt, 'agent_allocations', f'agent_allocations_Condition_{config.condition}',
           code_list=[0, 1, 2], policy_list=policies, disruption_start=28, disruption_end=33,
           title='Allocation Policies', ylabel='', xlabel='Time (Week)', outlier=True, hue='code')

draw_plots(rewards_data, 'agent_rewards', f'agent_rewards_Condition_{config.condition}',
           code_list=[0, 2], policy_list=None, title='Total Profit', ylabel='Amount',
           xlabel='Code', outlier=True, hue='code')

draw_plots(players_order_data_melt, 'player_orders', f'player_orders_Condition_{config.condition}',
           policy_list=None, title='Player Orders', ylabel='Order Amount', disruption_start=7, disruption_end=12,
           xlabel='Time (Week)', outlier=True)

draw_plots(order_ppo2_melt, 'ppo2_orders', f'ppo2_orders_Condition_{config.condition}',
           policy_list=None, title='Order Amounts selected by PPO2 Agent', ylabel='Order Amount', disruption_start=28, disruption_end=33,
           xlabel='Time (Week)', outlier=True)

draw_plots(allocation_ppo2_melt, 'ppo2_allocations', f'ppo2_allocations_Condition_{config.condition}',
           policy_list=policies, disruption_start=28, disruption_end=33,
           title='Allocation Policies selected by PPO2 Agent', ylabel='', xlabel='Time (Week)')

draw_plots(allocation_ppo2_time_count_melt, 'ppo2_allocation_time_window',
           f'ppo2_allocations_windowed_Condition_{config.condition}',
           policy_list=policies, disruption_start=0.5, disruption_end=1.5,
           title='Allocation Policies selected by PPO2 Agent', ylabel='', xlabel='Time (Week)')

draw_plots(rewards_ppo2, 'ppo2_rewards', f'ppo2_rewards_Condition_{config.condition}',
           policy_list=None, title='Total Profit of PPO2 Agent', ylabel='Amount',
           xlabel='', outlier=True)