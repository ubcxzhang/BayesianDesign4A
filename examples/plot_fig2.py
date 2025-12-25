import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# # Helper function for normalization
# def normalize(value, min_val, max_val):
#     return (value - min_val) / (max_val - min_val)

# NEW subplot setup
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex=True) # , sharey=True
plt.rcParams.update({'font.size': 20})

# fix Sigma1_coefficient k to be 4
fixSigma1_coeff_k = 4 
# fixIs = 211
# sample_size = fixIs

# Fixed parameters
fixt = np.array([0.25, 0.5, 0.75])
t_values = np.array([0.25, 0.50])
fixIs = 211
criteria = ["<5%", "in [5%,10%]", ">10%"]
Z_alpha = norm.ppf(1 - 0.025)
colors = [
    '#1f77b4',  # Dark Blue
    '#2ca02c',  # Dark Green
    '#ff7f0e',  # Dark Orange
    '#d62728',  # Dark Red
    '#9467bd'   # Purple
]
color_dict = {t: color for t, color in zip([0.25, 0.3, 0.5, 0.7, 0.75], colors)}

# p1_values = [0.25, 0.5, 0.75]
p1_dict =  {'none': [0.25, 0.5], 'weak': [0.5, 0.75]} 
delta_dict = {'none': 0.25, 'weak': 0.2}
effect_rows = ['none', 'weak']  # First row is 'none', second is 'weak'

for row_idx, effect in enumerate(effect_rows):
    delta = delta_dict[effect]
    tables = pd.read_csv(f'results/t_Test_iall_{effect}_combined_results.csv')
    deg_tables2 = tables 
    
#     # print('deg',deg_rtable['deg.power'])
#     # tables = tables[tables['Is'] == fixIs]
    # distinct_combinations = tables[['Sigma1_coeff_k']].drop_duplicates()
#     unique_dcofs = distinct_combinations['dcof'].unique()
#     dcof = unique_dcofs[0] #  if effect == 'none' else unique_dcofs[0]

#     unique_x1ts = distinct_combinations['X1_t_orig'].unique()
#     tables_dict = {x1t: tables[tables['X1_t_orig'] == x1t] for x1t in unique_x1ts}
#     deg_tables2_dict = {x1t: deg_tables2[abs(deg_tables2['X1_t_orig'] - x1t)<1e-6] for x1t in unique_x1ts}
    filtered_table = tables
    deg_filtered_table2=deg_tables2
    dcof = tables['dcof'].unique().item()
    fixIs = tables['Is'].unique().item()
    for col_idx, fixp1 in enumerate(p1_dict[effect]):
        ax = axs[row_idx, col_idx]
        
#         for x1t in unique_x1ts:
#             filtered_table = tables_dict.get(x1t, pd.DataFrame())
#             deg_filtered_table2 = deg_tables2_dict.get(x1t, pd.DataFrame())

#             if filtered_table.empty:
#                 continue

        label_added = False
        
        for t in t_values:
            color = color_dict[t]
            alpha = 0.6
            
            selected_rows = filtered_table[
                (filtered_table['p_1'] == fixp1) &
                (filtered_table['t'] == t) &
                (filtered_table['delta'] == delta) &
                (filtered_table['dcof'] == dcof) &
                (filtered_table['Sigma1_coeff_k'] == fixSigma1_coeff_k)  
            ]
            # if effect=='weak':
            #     rtable=selected_rows[selected_rows['r']==1]
            #     print(rtable[['p_1','t','power','X1_t_orig','success']])
            selected_rows=selected_rows[(selected_rows['success'] == True)]
            
            deg_selected_rows2 = deg_filtered_table2[
                (deg_filtered_table2['p_1'] == fixp1) &
                (deg_filtered_table2['t'] == t) &
                (deg_filtered_table2['delta'] == delta) &
                (deg_filtered_table2['dcof'] == dcof) &
                (deg_filtered_table2['Sigma1_coeff_k'] == fixSigma1_coeff_k) 
            ]
            
#             print(f"Effect: {effect}, fixp1: {fixp1}, t: {t}, delta: {delta}")
#             print(f"Selected rows shape: {selected_rows.shape}")
#             print(f"Deg selected rows shape: {deg_selected_rows2.shape}")

            # Even if selected_rows is empty, force a dummy entry for the legend
            ax.plot([], [], label=f't={t}', marker=' ', linestyle='None', markersize=15, color=color, alpha=alpha)
            if selected_rows.empty:
                ax.plot([], [], label=f'Joint: t={t}', color=color, marker='o', linestyle='None', alpha=alpha)
                continue

            else:
                # selected_rows = selected_rows[selected_rows['r'] < 1]
                deg_selected_rows_subg2 = deg_selected_rows2 #[deg_selected_rows2['r'] < 1]
                
                ax.plot(selected_rows['r'], selected_rows['power'], label=f'Joint', marker='o', color=color, alpha=alpha, markersize=10)
                ax.plot([], [], label='Joint (highest power)', marker='*', linestyle='None', markersize=15, color=color, alpha=1.00)
                
                ax.scatter(deg_selected_rows_subg2['r'], deg_selected_rows_subg2['deg.power'], label='Subgroup only', marker='x', color=color, alpha=alpha, s=200)
                ax.locator_params(axis='x', nbins=10)
                # Full population line
                deg_selected_rows_full2 = deg_selected_rows2[deg_selected_rows2['r'] == 1]
                if not deg_selected_rows_full2.empty:
                    full_pop_power = deg_selected_rows_full2['deg.power'].iloc[0]
                    ax.axhline(y=full_pop_power, label='Entire population', color=color, alpha=alpha, linestyle='--', linewidth=3)
                # ax.axhline(y=0.975, label=f'97.5% power', color=color, alpha=0.3, linestyle='-.', linewidth=3)

                # Highlighting joint > subgroup and full
                joint_power = selected_rows[['r','t','power','z','mu_2']].reset_index(drop=True)
                subgroup_power = deg_selected_rows_subg2[['r','deg.power','deg.z','mu_2','Sigma2']].reset_index(drop=True)
                joint_power['r'] = joint_power['r'].round(5)
                subgroup_power['r'] = subgroup_power['r'].round(5)
                merged_power = pd.merge(joint_power, subgroup_power, on='r', how='left').fillna({'deg.power': 0})
                
                if row_idx==1 and col_idx == 0:
                    # print(subgroup_power)
                    print(merged_power,"full",full_pop_power)
                if effect == 'weak':
                    for _, row in merged_power.iterrows():
                        r_val, jp, sp = row['r'], row['power'], row['deg.power']
                        if (jp > sp and jp > full_pop_power):
                            ax.scatter(r_val, jp, marker='*', color=color, s=300, zorder=5, alpha=1.0) #  if not highlighted else ""
                            highlighted = True
                        
            
            
            if row_idx == 0:
                ax.set_title(f"$p_1$={fixp1}, $\delta$={delta}, $I$={fixIs}, \n no-biomarker: $X_{{2,t}}=X_{{1,t}}\\sqrt{{r}}$")
                # ax.set_ylim(0.7, .77)
            if col_idx == 0:
                ax.set_ylabel("Power", fontsize=22)
            if row_idx == 1:
                ax.set_title(f"$p_1$={fixp1}, $\delta$={delta}, $I$={fixIs}, \n$X_{{2,t}}=\\sqrt{{r}}(X_{{1,t}}+{dcof}\\sqrt{{It}}(1-r))$")
                ax.set_xlabel("r", fontsize=22)

            # if col_idx == 1 and row_idx == 1:
            #     ax.legend(loc='upper left', bbox_to_anchor=(1, 1.5), ncol=1)
from matplotlib.lines import Line2D
color_handles = [
    Line2D([0], [0], color='#1f77b4', lw=2, label='t=0.25'),
    Line2D([0], [0], color='#ff7f0e', lw=2, label='t=0.5')
]

marker_handles = [
    Line2D([0], [0], color='black', marker='o', linestyle='-', label='Joint',markersize=10),
    Line2D([0], [0], color='black', marker='x', linestyle='None', label='Subgroup',markersize=10),
    Line2D([0], [0], color='black', marker='None', linestyle='--', label='Full',markersize=10),
    Line2D([0], [0], color='black', marker='*', linestyle='None', label='Joint is the best (when d>0)',markersize=20)
]

# First legend: models (colors)
legend1 = fig.legend(handles=color_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
ax.add_artist(legend1)  # Keep it on the axes

# Second legend: types (markers)
legend2 = fig.legend(handles=marker_handles, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=4)


plt.suptitle(f'Power vs r in True Estimation ($k$={fixSigma1_coeff_k}, $X_{{1,t}}=\delta \sqrt{{It}}$)', fontsize=24)
# plt.suptitle(f'Power vs r ($I$={fixIs}, $k$={fixSigma1_coeff_k}, $X_{{1,0.2}}$={np.round(unique_x1ts[0],5)}, i.e. $Z_{{0.95}}$)', fontsize=24)

plt.tight_layout()
plt.savefig(f'results/grid_power_vs_r_2x2_true_{fixSigma1_coeff_k}.pdf', facecolor='white', edgecolor='white', bbox_inches='tight')
plt.show()