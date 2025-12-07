#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:49:36 2025

@author: brunocaetanovidigal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 06:56:31 2025

@author: brunocaetanovidigal
"""

# libraries

import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen, VerticalPitch
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec



# open the data
parser = Sbopen()
df_match = parser.match(competition_id=55, season_id=282)
#our team
team = "Spain"
#get list of games by our team, either home or away
match_ids = df_match.loc[(df_match["home_team_name"] == team) | (df_match["away_team_name"] == team)]["match_id"].tolist()


# finding danger passes

#Open event data for all matches and concatenate them
df_all_events = pd.DataFrame()
for match_id in match_ids:
    df_events = parser.event(match_id)[0]
    df_all_events = pd.concat([df_all_events, df_events])


# get passes

#Take out the open play successful passes from the possession team
df_passes = df_all_events[(df_all_events['type_name'] == 'Pass')
                          & (df_all_events['outcome_name'].isnull())
                          & (df_all_events['possession_team_id'] == df_all_events['team_id'])
                          & (~df_all_events.sub_type_name.isin(['Throw-in','Corner','Free Kick', 'Kick Off', 'Goal Kick']))
                          ]

# Filter for our team
df_passes_our_team = df_passes[df_passes['team_name'] == team]
# Only keep necessary columns
df_passes_our_team = df_passes_our_team[['match_id', 'x', 'y', 'end_x', 'end_y', 'minute','second','player_name']]

# filter only passes made by specific player

chosen_player = 'Rodrigo Hernández Cascante'
player_short_name = 'Rodri'

#chosen_player = "Riccardo Calafiori"
#player_short_name = "Calafiori"

df_passes_player = df_passes_our_team[df_passes_our_team['player_name'] == chosen_player]

#calculate number of games played by the player

number_games = df_passes_player['match_id'].nunique()


# making heat map

#plot vertical pitch
pitch = Pitch(line_zorder=2, line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
#get the 2D histogram
bin_statistic = pitch.bin_statistic(df_passes_player.x, df_passes_player.y, statistic='count', bins=(6, 5), normalize=False)
#normalize by number of games played by player
bin_statistic["statistic"] = bin_statistic["statistic"]/number_games
#make a heatmap
pcm  = pitch.heatmap(bin_statistic, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Passes by ' + player_short_name + " per game", fontsize = 30)
plt.show()

# function to turn(x, y) into a zone index 0-29

# parameters of the grid
N_X = 6   # slices along length (x)
N_Y = 5   # slices along width (y)
PITCH_X = 120
PITCH_Y = 80

def coords_to_zone(x, y,
                   n_x=N_X, n_y=N_Y,
                   pitch_x=PITCH_X, pitch_y=PITCH_Y):
    """
    Map (x, y) to a zone index in [0, n_x*n_y-1].
    Zones are numbered by columns (x) then rows (y), e.g.:
    x_bin in [0..n_x-1], y_bin in [0..n_y-1]
    zone = x_bin * n_y + y_bin
    """
    # bin indices along each axis
    x_bin = np.floor(x / (pitch_x / n_x)).astype(int)
    y_bin = np.floor(y / (pitch_y / n_y)).astype(int)

    # safety: clip to valid range in case of x == 120 or y == 80
    x_bin = np.clip(x_bin, 0, n_x - 1)
    y_bin = np.clip(y_bin, 0, n_y - 1)

    # flatten 2D bin -> 1D zone index
    zone = x_bin * n_y + y_bin    # 0..29
    return zone

# start zone
df_passes_player['start_zone'] = coords_to_zone(df_passes_player['x'],
                                             df_passes_player['y'])

# end zone
df_passes_player['end_zone'] = coords_to_zone(df_passes_player['end_x'],
                                           df_passes_player['end_y'])

# build 30x30 transition matrix

# raw counts
transition = pd.crosstab(df_passes_player['start_zone'],
                         df_passes_player['end_zone'])

# ensure full 30×30 layout (0..29 on both axes)
all_zones = np.arange(N_X * N_Y)
transition = transition.reindex(index=all_zones,
                                columns=all_zones,
                                fill_value=0)

print(transition.shape)  # (30, 30)

# Row-normalize so each row sums to 1: “given the ball starts in zone i, where does it go?”

transition_prob = transition.div(transition.sum(axis=1).replace(0, np.nan),
                                 axis=0)

# making heat map of 30x30

# ---------- assumes: transition_prob is a 30x30 DataFrame ----------

PITCH_X = 120
PITCH_Y = 80
GOAL_WIDTH = 7.32
GOAL_Y_CENTER = PITCH_Y / 2
GOAL_LEFT_Y1 = GOAL_Y_CENTER - GOAL_WIDTH/2
GOAL_LEFT_Y2 = GOAL_Y_CENTER + GOAL_WIDTH/2

PENALTY_AREA_X = 18
PENALTY_AREA_Y = 44
GOAL_AREA_X = 6
GOAL_AREA_Y = 20

N_X = 6
N_Y = 5

# -------------------------------------------------------------------
# 2) CREATE FIGURE LAYOUT: top = matrix, bottom = legend
# -------------------------------------------------------------------
fig = plt.figure(figsize=(12, 16))

gs = GridSpec(
    2, 2,
    height_ratios=[1.6, 0.4],   # heatmap taller than legend
    width_ratios=[1.0, 0.05],   # main plots + thin column for colorbars
    hspace=0.3, wspace=0.15
)

ax_heat      = fig.add_subplot(gs[0, 0])  # matrix
ax_heat_cbar = fig.add_subplot(gs[0, 1])  # matrix colorbar
ax_legend    = fig.add_subplot(gs[1, 0])  # MPLSoccer legend
ax_leg_cbar  = fig.add_subplot(gs[1, 1])  # legend colorbar

# -------------------------------------------------------------------
# 3) TOP: TRANSITION MATRIX HEATMAP (unchanged logic)
# -------------------------------------------------------------------
matrix = transition_prob.copy()
matrix_to_plot = matrix.fillna(0)

im = ax_heat.imshow(
    matrix_to_plot.values,
    origin='lower',
    aspect='equal',
    cmap='Reds',
    interpolation='nearest',
    vmin=0,
    vmax=np.nanmax(matrix_to_plot.values)
)

ax_heat.set_title("Pass Transition Probability (30×30) by " + player_short_name,
                  fontsize=14, pad=12)
ax_heat.set_xlabel("To zone (end of pass)")
ax_heat.set_ylabel("From zone (start of pass)")

# Tick labels 1–30 (every 2)
positions = np.arange(0, 30, 2)
labels    = np.arange(1, 31, 2)

ax_heat.set_xticks(positions)
ax_heat.set_xticklabels(labels, fontsize=9)

ax_heat.set_yticks(positions)
ax_heat.set_yticklabels(labels, fontsize=9)

# Thin grid for each cell
ax_heat.set_xticks(np.arange(-0.5, 30, 1), minor=True)
ax_heat.set_yticks(np.arange(-0.5, 30, 1), minor=True)
ax_heat.grid(which="minor", color="black", linewidth=0.25, alpha=0.25)

# Thick lines at third boundaries: 1–10, 11–20, 21–30 → matrix indices 0,10,20,30
third_bounds = [0, 10, 20, 30]
for k in third_bounds:
    ax_heat.axvline(k - 0.5, color='black', linewidth=2.5)
    ax_heat.axhline(k - 0.5, color='black', linewidth=2.5)

# Colorbar for matrix
cbar_heat = plt.colorbar(im, cax=ax_heat_cbar)
cbar_heat.set_label("Probability", fontsize=11)

# -------------------------------------------------------------------
# 4) BOTTOM: MPLSOCCER HEATMAP LEGEND ON ax_legend
# -------------------------------------------------------------------
pitch = Pitch(line_zorder=2, line_color='black')
pitch.draw(ax=ax_legend)

# 2D histogram on the same pitch
bin_statistic = pitch.bin_statistic(
    df_passes_player.x, df_passes_player.y,
    statistic='count', bins=(N_X, N_Y), normalize=False
)
bin_statistic["statistic"] = bin_statistic["statistic"] / number_games

pcm = pitch.heatmap(
    bin_statistic, cmap='Reds', edgecolor='grey', ax=ax_legend
)

# bin edges in pitch coordinates
x_bins = np.linspace(pitch.dim.left, pitch.dim.right, N_X + 1)
y_bins = np.linspace(pitch.dim.bottom, pitch.dim.top, N_Y + 1)

# 4.1 Zone numbers (1–30)
zone = 1
for i in range(N_X):      # x bins
    for j in range(N_Y):  # y bins
        xc = (x_bins[i] + x_bins[i+1]) / 2
        yc = (y_bins[j] + y_bins[j+1]) / 2
        ax_legend.text(
            xc, yc, str(zone),
            ha='center', va='center',
            fontsize=12, color='black', weight='bold'
        )
        zone += 1

# 4.2 Thick third boundaries (1–10, 11–20, 21–30)
third_lines = [x_bins[2], x_bins[4]]  # after 2 and 4 columns
for x_third in third_lines:
    ax_legend.plot(
        [x_third, x_third],
        [pitch.dim.bottom, pitch.dim.top],
        color='black', linewidth=3
    )

# 4.3 Third labels above pitch
third_centers = [
    (x_bins[0] + x_bins[2]) / 2,   # first third
    (x_bins[2] + x_bins[4]) / 2,   # middle third
    (x_bins[4] + x_bins[6]) / 2    # final third
]

pitch_height = pitch.dim.top - pitch.dim.bottom
y_label = pitch.dim.top + 0.03 * pitch_height

ax_legend.text(third_centers[0], y_label, "First third",
               ha='center', va='bottom', fontsize=11)
ax_legend.text(third_centers[1], y_label, "Middle third",
               ha='center', va='bottom', fontsize=11)
ax_legend.text(third_centers[2], y_label, "Final third",
               ha='center', va='bottom', fontsize=11)

# 4.4 ATTACK DIRECTION ARROW OUTSIDE THE PITCH (below), using axes fraction
y_arrow_ax = -0.15    # 15% below axes
x_start_ax = 0.20
x_end_ax   = 0.80

ax_legend.annotate(
    '',
    xy=(x_end_ax, y_arrow_ax),
    xytext=(x_start_ax, y_arrow_ax),
    xycoords='axes fraction',
    textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', linewidth=2.5, color='black')
)
ax_legend.text(
    0.5, y_arrow_ax - 0.05,
    "Attacking direction",
    ha='center', va='top',
    fontsize=11, color='black',
    transform=ax_legend.transAxes
)

# Colorbar for legend
cbar_leg = plt.colorbar(pcm, cax=ax_leg_cbar)
cbar_leg.set_label("Passes per game", fontsize=11)

# -------------------------------------------------------------------
# 5) GLOBAL TITLE / LAYOUT
# -------------------------------------------------------------------
fig.suptitle('Pass flow by zones – ' + player_short_name + ' (' + team + ')' , fontsize=18, y=0.98)

plt.tight_layout(rect=[0, 0, 0.96, 0.96])
plt.show()