# ============================================================
# 7-ZONE PASS TRANSITION MATRIX + 7-ZONE LEGEND HEATMAP
# (StatsBomb Open Data via mplsoccer.Sbopen)
# - Top: 7x7 transition probability matrix (start_zone -> end_zone)
# - Bottom: pitch legend with passes-per-game heatmap split into 7 zones
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplsoccer import Pitch, Sbopen

# -----------------------------
# USER SETTINGS
# -----------------------------
COMPETITION_ID = 55   # EURO
SEASON_ID      = 282  # 2024
TEAM_NAME      = "Spain"

CHOSEN_PLAYER_FULL  = "Rodrigo Hernández Cascante"
PLAYER_SHORT_NAME   = "Rodri"

# Set to None to not filter by player (then it uses all team passes)
# CHOSEN_PLAYER_FULL = None
# PLAYER_SHORT_NAME  = TEAM_NAME

# -----------------------------
# HELPERS
# -----------------------------
OPEN_PLAY_SET_PIECES = {'Throw-in', 'Corner', 'Free Kick', 'Kick Off', 'Goal Kick'}

PITCH_X, PITCH_Y = 120, 80

ZONE_ORDER = [
    "Own Half",
    "Left Wing",
    "Left Half-space",
    "Central",
    "Right Half-space",
    "Right Wing",
    "Box"
]

ZONE_TO_ID = {name: i for i, name in enumerate(ZONE_ORDER)}
ID_TO_ZONE = {i: name for name, i in ZONE_TO_ID.items()}

def to_zone7(x, y):
    """
    Map StatsBomb (x,y) into 7 zones:
      0 Own Half
      1 Left Wing
      2 Left Half-space
      3 Central
      4 Right Half-space
      5 Right Wing
      6 Box (opponent penalty area)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Box (opponent penalty area): last 18m and central 44m
    in_box = (x >= 102) & (y >= 18) & (y <= 62)

    # Own half
    in_own_half = x < 60

    # Attacking half lanes
    lane = np.full_like(x, fill_value=np.nan)

    lane[(y >= 0)  & (y < 16)] = ZONE_TO_ID["Left Wing"]
    lane[(y >= 16) & (y < 32)] = ZONE_TO_ID["Left Half-space"]
    lane[(y >= 32) & (y < 48)] = ZONE_TO_ID["Central"]
    lane[(y >= 48) & (y < 64)] = ZONE_TO_ID["Right Half-space"]
    lane[(y >= 64) & (y <= 80)] = ZONE_TO_ID["Right Wing"]

    zone = np.full_like(x, fill_value=-1, dtype=int)
    zone[in_box] = ZONE_TO_ID["Box"]
    zone[~in_box & in_own_half] = ZONE_TO_ID["Own Half"]

    mask_att = (~in_box) & (~in_own_half)
    zone[mask_att] = lane[mask_att].astype(int)

    return zone

def get_team_match_ids(df_match, team_name):
    return df_match.loc[
        (df_match["home_team_name"] == team_name) | (df_match["away_team_name"] == team_name),
        "match_id"
    ].tolist()

def filter_open_play_successful_passes(df_events):
    # Successful pass: type_name == Pass and outcome_name is NA
    # Open play: sub_type_name not in set pieces
    # Possession team matches event team (so you're looking at team in-possession actions)
    df_passes = df_events[
        (df_events["type_name"] == "Pass") &
        (df_events["outcome_name"].isna()) &
        (df_events["possession_team_id"] == df_events["team_id"]) &
        (~df_events["sub_type_name"].isin(OPEN_PLAY_SET_PIECES))
    ].copy()

    # keep essential columns
    keep_cols = ["match_id", "team_name", "player_name", "x", "y", "end_x", "end_y", "minute", "second"]
    keep_cols = [c for c in keep_cols if c in df_passes.columns]
    return df_passes[keep_cols].copy()

# -----------------------------
# LOAD DATA (all matches for team)
# -----------------------------
parser = Sbopen()
df_match = parser.match(competition_id=COMPETITION_ID, season_id=SEASON_ID)

match_ids = get_team_match_ids(df_match, TEAM_NAME)
if len(match_ids) == 0:
    raise ValueError(f"No matches found for TEAM_NAME='{TEAM_NAME}'. Check spelling.")

# concat all event data for team matches
df_all_events = []
for mid in match_ids:
    ev = parser.event(mid)[0]
    df_all_events.append(ev)
df_all_events = pd.concat(df_all_events, ignore_index=True)

# passes (open play, successful)
df_passes = filter_open_play_successful_passes(df_all_events)

# keep only passes by the selected team
df_passes_team = df_passes[df_passes["team_name"] == TEAM_NAME].copy()

# optional: filter by player
if CHOSEN_PLAYER_FULL is not None:
    df_passes_team = df_passes_team[df_passes_team["player_name"] == CHOSEN_PLAYER_FULL].copy()
    plot_name = PLAYER_SHORT_NAME
else:
    plot_name = TEAM_NAME

if df_passes_team.empty:
    raise ValueError("No passes after filtering. Try a different team/player or relax filters.")

# number of games played (for passes-per-game heatmap)
number_games = df_passes_team["match_id"].nunique()

# -----------------------------
# 1) BUILD 7x7 TRANSITION MATRIX
# -----------------------------
df_passes_team["start_zone7"] = to_zone7(df_passes_team["x"].values, df_passes_team["y"].values)
df_passes_team["end_zone7"]   = to_zone7(df_passes_team["end_x"].values, df_passes_team["end_y"].values)

# raw counts 7x7
transition_counts = pd.crosstab(df_passes_team["start_zone7"], df_passes_team["end_zone7"])

# ensure full 7x7
all7 = np.arange(7)
transition_counts = transition_counts.reindex(index=all7, columns=all7, fill_value=0)

# row-normalize -> probabilities
row_sums = transition_counts.sum(axis=1).replace(0, np.nan)
transition_prob = transition_counts.div(row_sums, axis=0).fillna(0)

# label rows/cols
transition_prob.index = [ID_TO_ZONE[i] for i in transition_prob.index]
transition_prob.columns = [ID_TO_ZONE[i] for i in transition_prob.columns]

# for plotting as array in ZONE_ORDER order
matrix = transition_prob.loc[ZONE_ORDER, ZONE_ORDER].values

# -----------------------------
# 2) BUILD LEGEND: PASSES PER GAME BY 7 ZONES
# -----------------------------
zones_start = to_zone7(df_passes_team["x"].values, df_passes_team["y"].values)
counts7 = np.bincount(zones_start, minlength=7).astype(float)
passes_per_game = counts7 / number_games

# -----------------------------
# 3) PLOT: MATRIX (TOP) + LEGEND HEATMAP (BOTTOM)
# -----------------------------
fig = plt.figure(figsize=(12, 14))
gs = GridSpec(
    2, 2,
    height_ratios=[1.8, 0.9],
    width_ratios=[1.0, 0.06],
    hspace=0.35,
    wspace=0.15
)

ax_mat   = fig.add_subplot(gs[0, 0])
ax_matcb = fig.add_subplot(gs[0, 1])

ax_leg   = fig.add_subplot(gs[1, 0])
ax_legcb = fig.add_subplot(gs[1, 1])

# ---- TOP: 7x7 transition matrix
im = ax_mat.imshow(
    matrix,
    origin="lower",
    aspect="equal",
    cmap="Reds",
    vmin=0,
    vmax=np.nanmax(matrix) if np.nanmax(matrix) > 0 else 1
)

ax_mat.set_title(f"7-zone pass transition probability — {plot_name} ({TEAM_NAME})", fontsize=14, pad=12)
ax_mat.set_xlabel("To zone (end of pass)")
ax_mat.set_ylabel("From zone (start of pass)")

ax_mat.set_xticks(np.arange(7))
ax_mat.set_yticks(np.arange(7))
ax_mat.set_xticklabels(ZONE_ORDER, rotation=35, ha="right")
ax_mat.set_yticklabels(ZONE_ORDER)

# gridlines for readability
ax_mat.set_xticks(np.arange(-0.5, 7, 1), minor=True)
ax_mat.set_yticks(np.arange(-0.5, 7, 1), minor=True)
ax_mat.grid(which="minor", color="black", linewidth=0.6, alpha=0.25)

cbar1 = plt.colorbar(im, cax=ax_matcb)
cbar1.set_label("Probability", fontsize=11)

# ---- BOTTOM: pitch legend with 7-zone passes-per-game heatmap
pitch = Pitch(line_zorder=2, line_color="black")
pitch.draw(ax=ax_leg)

# Create 7 rectangles (same geometry as to_zone7)
from matplotlib.patches import Rectangle

def add_zone_rect(ax, x0, y0, w, h, value, vmin, vmax):
    cmap = plt.cm.Reds
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ax.add_patch(Rectangle((x0, y0), w, h,
                           facecolor=cmap(norm(value)),
                           edgecolor="black", lw=2.2, zorder=0))

vmin2 = 0
vmax2 = max(passes_per_game.max(), 1e-6)

# Own half
add_zone_rect(ax_leg, 0, 0, 60, 80, passes_per_game[ZONE_TO_ID["Own Half"]], vmin2, vmax2)

# Attacking half lanes
add_zone_rect(ax_leg, 60,  0, 60, 16, passes_per_game[ZONE_TO_ID["Left Wing"]], vmin2, vmax2)
add_zone_rect(ax_leg, 60, 16, 60, 16, passes_per_game[ZONE_TO_ID["Left Half-space"]], vmin2, vmax2)
add_zone_rect(ax_leg, 60, 32, 60, 16, passes_per_game[ZONE_TO_ID["Central"]], vmin2, vmax2)
add_zone_rect(ax_leg, 60, 48, 60, 16, passes_per_game[ZONE_TO_ID["Right Half-space"]], vmin2, vmax2)
add_zone_rect(ax_leg, 60, 64, 60, 16, passes_per_game[ZONE_TO_ID["Right Wing"]], vmin2, vmax2)

# Box (overrides)
add_zone_rect(ax_leg, 102, 18, 18, 44, passes_per_game[ZONE_TO_ID["Box"]], vmin2, vmax2)

# Labels
label_style = dict(ha="center", va="center", fontsize=12, fontweight="bold", color="black")
ax_leg.text(30, 40, "Own Half", **label_style)

ax_leg.text(90,  8, "Left Wing", **label_style)
ax_leg.text(90, 24, "Left Half-space", **label_style)
ax_leg.text(90, 40, "Central", **label_style)
ax_leg.text(90, 56, "Right Half-space", **label_style)
ax_leg.text(90, 72, "Right Wing", **label_style)

ax_leg.text(111, 40, "Box", **label_style)

# Attack direction arrow OUTSIDE the pitch
ax_leg.annotate(
    "",
    xy=(0.80, -0.16),
    xytext=(0.20, -0.16),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=2.5, color="black"),
)
ax_leg.text(0.5, -0.24, "Attacking direction", transform=ax_leg.transAxes,
            ha="center", va="top", fontsize=11)

ax_leg.set_title("Passes per game by zone (7-zone split)", fontsize=13, pad=10)

# Colorbar for legend
sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin2, vmax=vmax2))
sm.set_array([])
cbar2 = plt.colorbar(sm, cax=ax_legcb)
cbar2.set_label("Passes per game", fontsize=11)

plt.tight_layout()
plt.show()
