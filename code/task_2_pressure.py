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
# team = "Italy"
#get list of games by our team, either home or away
match_ids = df_match["match_id"]


# finding danger passes

#Open event data for all matches and concatenate them
df_all_events = pd.DataFrame()
for match_id in match_ids:
    df_events = parser.event(match_id)[0]
    df_all_events = pd.concat([df_all_events, df_events])


###########


# -----------------------------------------------------
# 1) CONFIGURAÇÕES
# -----------------------------------------------------
DEF_ACTIONS = ["Pressure", "Interception", "Ball Recovery", "Block", "Duel"]
#DEF_ACTIONS = ["Interception", "Ball Recovery", "Block"]

# zona típica de volante
#X_MIN, X_MAX = 35, 75
#Y_MIN, Y_MAX = 20, 60


# -----------------------------------------------------
# 2) IDENTIFICAR VOLANTES (defensive midfielders)
# -----------------------------------------------------

dm_positions = ["Left Defensive Midfield", "Center Defensive Midfield", "Right Defensive Midfield"]


DEF_ACTIONS = ["Pressure", "Interception", "Ball Recovery", "Block", "Duel"]


# filtrar só eventos onde a posição é de volante defensivo
mask_dm = df_all_events["position_name"].isin(dm_positions)

dm_info = (
    df_all_events
    .loc[mask_dm, ["player_id", "player_name", "team_id", "team_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

dm_info.head()

minutes_per_match = (
    df_all_events.groupby(["player_id", "match_id"])["minute"]
    .max()
    .reset_index(name="match_minutes")
)

minutes_player = (
    minutes_per_match.groupby("player_id")["match_minutes"]
    .sum()
    .reset_index(name="minutes_played")
)

# somente eventos defensivos
df_def = df_all_events[df_all_events["type_name"].isin(DEF_ACTIONS)].copy()

# somente defensivos na zona de volante
#df_def_zone = df_def[
 #   df_def["x"].between(X_MIN, X_MAX) &
  #  df_def["y"].between(Y_MIN, Y_MAX)
#].copy()

# analysing the entire pitch
df_def_zone = df_def.copy()

# contar ações defensivas na zona por jogador
def_counts = (
    df_def_zone.groupby("player_id")
    .size()
    .reset_index(name="def_actions_zone")
)

# juntar DMs com minutos e ações defensivas
dm_metrics = (
    dm_info
    .merge(minutes_player, on="player_id", how="left")
    .merge(def_counts, on="player_id", how="left")
)

dm_metrics["minutes_played"] = dm_metrics["minutes_played"].fillna(0)
dm_metrics["def_actions_zone"] = dm_metrics["def_actions_zone"].fillna(0)

# excluir volantes que jogaram muito pouco
dm_metrics = dm_metrics[dm_metrics["minutes_played"] >= 270].copy()

# calcular ações defensivas por 90 min
dm_metrics["def_actions_per90"] = (
    dm_metrics["def_actions_zone"] / (dm_metrics["minutes_played"] / 90)
)

# ordenar ranking
dm_metrics = dm_metrics.sort_values("def_actions_per90", ascending=False)

dm_metrics.head(20)


# calcular média e desvio padrão
mean_val = dm_metrics["def_actions_per90"].mean()
std_val  = dm_metrics["def_actions_per90"].std()

# calcular z-score
dm_metrics["z_def_actions"] = (dm_metrics["def_actions_per90"] - mean_val) / std_val

dm_rank = dm_metrics.sort_values("z_def_actions", ascending=False).reset_index(drop=True)

dm_rank[["player_name", "team_name", "def_actions_per90", "z_def_actions"]].head(15)

dm_rank[dm_rank["player_name"] == "Rodrigo Hernández Cascante"]
dm_rank[dm_rank["player_name"] == "Toni Kroos"]


### Create metric based on danger passes

#Identify danger passes
#Add time in seconds column
df_all_events["time_seconds"] = df_all_events["minute"]*60 + df_all_events["second"]

#Take out the shots
df_shots = df_all_events[(df_all_events['type_name'] == 'Shot')]

#Only keep the necessary columns about shots
df_shots = df_shots[['match_id', 'possession', 'time_seconds']]

#Take out the open play successful passes from the possession team
df_passes = df_all_events[(df_all_events['type_name'] == 'Pass')
                          & (df_all_events['outcome_name'].isnull())
                          & (df_all_events['possession_team_id'] == df_all_events['team_id'])
                          & (~df_all_events.sub_type_name.isin(['Throw-in','Corner','Free Kick', 'Kick Off', 'Goal Kick']))
                          ]

# Merge shots and passes on possession and match_id
# Use a inner join to keep only passes that have a matching shot in the same possession

df_merged = df_shots.merge(df_passes, on=['possession', 'match_id'], how='inner',suffixes=('_shot',''))


# Calculate time difference between pass and shot
df_merged['time_diff'] = df_merged['time_seconds_shot'] - df_merged['time_seconds']

# Keep only passes that occurred within 15 seconds before the shot
df_danger_passes = df_merged[df_merged['time_diff'].between(0,15)]

# Some possessions may have multiple shots, keep only the shot with the smallest time_diff to each pass
first_shot = df_danger_passes.groupby('id')['time_diff'].idxmin()
df_danger_passes = df_danger_passes.loc[first_shot].reset_index(drop=True)


# filter only defensive midfielders

df_danger_passes = df_danger_passes[df_danger_passes["position_name"].isin(dm_positions)].copy()

#count passes by player and normalize them
pass_count =  (
    df_danger_passes
    .groupby(["player_id", "player_name", "team_name"])
    .size()
    .reset_index(name="danger_passes")
)

pass_count = pass_count.merge(minutes_player, on=['player_id'], how='inner')

# excluir volantes que jogaram muito pouco
pass_count = pass_count[pass_count["minutes_played"] >= 270].copy()


pass_count["danger_passes_per_min"] = (
    pass_count["danger_passes"] / pass_count["minutes_played"]
)

pass_count["danger_passes_per_90"] = (
    pass_count["danger_passes_per_min"] * 90
)

# calcular média e desvio padrão
mean_danger_passes_per_90 = pass_count["danger_passes_per_90"].mean()
std_danger_passes_per_90  = pass_count["danger_passes_per_90"].std()

# calcular z-score
pass_count["z_danger_passes_per_90"] = (pass_count["danger_passes_per_90"] - mean_danger_passes_per_90) / std_danger_passes_per_90

pass_count_rank = pass_count.sort_values("z_danger_passes_per_90", ascending=False).reset_index(drop=True)

pass_count_rank[["player_name", "team_name", "danger_passes_per_90", "z_danger_passes_per_90"]].head(15)

pass_count_rank[pass_count_rank["player_name"] == "Rodrigo Hernández Cascante"]


df_all_events[(df_all_events["player_name"] == "Luka Modrić")][['player_name', 'position_name']]

# merge both datasets

z_score_player = (
    dm_rank
    .merge(pass_count, on=["player_id", "player_name", "team_name", "minutes_played"], how="left")
)


## plot

highlight_names = [
    "Rodrigo Hernández Cascante",  # Rodri
    "Toni Kroos",
    "Granit Xhaka",
    "Vitor Machado Ferreira",
    "João Maria Lobo Alves Palhinha Gonçalves",
    "Adam Gnezda Čerin",
    "Nicolas Seiwald",
    "N'Golo Kanté",
    "Jude Bellingham",
    "Aurélien Djani Tchouaméni",
    "Pierre-Emile Højbjerg",
    "Fabián Ruiz Peña"
]

short_names = {
    "Rodrigo Hernández Cascante": "Rodri",
    "Toni Kroos": "Kroos",
    "Granit Xhaka": "Xhaka",
    "Vitor Machado Ferreira": "Vitinha",
    "João Maria Lobo Alves Palhinha Gonçalves": "Palhinha",
   "Adam Gnezda Čerin": "Čerin",
   "Nicolas Seiwald": "Seiwald",
   "N'Golo Kanté": "Kanté",
   "Jude Bellingham": "Bellingham",
   "Aurélien Djani Tchouaméni": "Tchouaméni",
   "Pierre-Emile Højbjerg": "Højbjerg",
   "Fabián Ruiz Peña": "Fabián"
}

# base scatter (all players, light)
plt.figure(figsize=(8, 6))
plt.scatter(
    z_score_player["z_def_actions"],
    z_score_player["z_danger_passes_per_90"],
    alpha=0.2
)

# highlight subset
mask_hi = z_score_player["player_name"].isin(highlight_names)
hi = z_score_player[mask_hi]

# Colors
background_color = "#C7C7C7"   # light grey
highlight_color = "#1f77b4"    # strong blue

plt.figure(figsize=(10, 7))

# --- Background players ---
plt.scatter(
    z_score_player["z_def_actions"],
    z_score_player["z_danger_passes_per_90"],
    color=background_color,
    alpha=0.5,
    s=40
)

# --- Highlight players ---
plt.scatter(
    hi["z_def_actions"],
    hi["z_danger_passes_per_90"],
    s=80,
    color=highlight_color,
    edgecolor="black",
    linewidth=1
)

# --- Labels ---
x_offset = 0.06

for _, row in hi.iterrows():
    label = short_names.get(row["player_name"], row["player_name"])
    plt.text(
        row["z_def_actions"] + x_offset,
        row["z_danger_passes_per_90"],
        label,
        fontsize=9,
        ha="left",
        va="center"
    )

# Axes & formatting
plt.axhline(0, color="grey", linestyle="--", linewidth=1)
plt.axvline(0, color="grey", linestyle="--", linewidth=1)
plt.xlabel("Z-score: defensive actions per 90")
plt.ylabel("Z-score: dangerous passes per 90")
plt.title("Defensive vs offensive contribution (DMs)")
plt.tight_layout()
plt.show()










