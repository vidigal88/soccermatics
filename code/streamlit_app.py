import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch, Sbopen
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------
# 1. LOAD DATA (all matches in EURO 2024)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    parser = Sbopen()
    df_match = parser.match(competition_id=55, season_id=282)
    match_ids = df_match["match_id"].tolist()

    df_all_events = pd.DataFrame()
    for match_id in match_ids:
        df_events = parser.event(match_id)[0]
        df_all_events = pd.concat([df_all_events, df_events], ignore_index=True)

    return df_match, df_all_events


# ------------------------------------------------------------
# 2. PASS-FLOW HELPERS: compute transition matrix + legend
# ------------------------------------------------------------
@st.cache_data
def compute_transition_and_passes(df_all_events, team, chosen_player):
    """
    Filter passes for a given team & player, compute 30x30 transition_prob
    and return (transition_prob, df_passes_player, number_games).
    """
    # Only successful open-play passes by possession team
    df_passes = df_all_events[
        (df_all_events['type_name'] == 'Pass')
        & (df_all_events['outcome_name'].isnull())
        & (df_all_events['possession_team_id'] == df_all_events['team_id'])
        & (~df_all_events['sub_type_name'].isin(
            ['Throw-in', 'Corner', 'Free Kick', 'Kick Off', 'Goal Kick']
        ))
    ]

    # Filter by team
    df_passes_our_team = df_passes[df_passes['team_name'] == team]
    df_passes_our_team = df_passes_our_team[
        ['match_id', 'x', 'y', 'end_x', 'end_y', 'minute', 'second', 'player_name']
    ]

    # Filter by player
    df_passes_player = df_passes_our_team[
        df_passes_our_team['player_name'] == chosen_player
    ].copy()

    if df_passes_player.empty:
        raise ValueError(f"No passes found for {chosen_player} in {team}.")

    number_games = df_passes_player['match_id'].nunique()

    # Grid parameters
    N_X = 6
    N_Y = 5
    PITCH_X = 120
    PITCH_Y = 80

    def coords_to_zone(x, y,
                       n_x=N_X, n_y=N_Y,
                       pitch_x=PITCH_X, pitch_y=PITCH_Y):
        x_bin = np.floor(x / (pitch_x / n_x)).astype(int)
        y_bin = np.floor(y / (pitch_y / n_y)).astype(int)
        x_bin = np.clip(x_bin, 0, n_x - 1)
        y_bin = np.clip(y_bin, 0, n_y - 1)
        return x_bin * n_y + y_bin   # 0..29

    df_passes_player['start_zone'] = coords_to_zone(
        df_passes_player['x'], df_passes_player['y']
    )
    df_passes_player['end_zone'] = coords_to_zone(
        df_passes_player['end_x'], df_passes_player['end_y']
    )

    # 30x30 transition matrix
    transition = pd.crosstab(
        df_passes_player['start_zone'],
        df_passes_player['end_zone']
    )

    all_zones = np.arange(N_X * N_Y)
    transition = transition.reindex(
        index=all_zones,
        columns=all_zones,
        fill_value=0
    )

    transition_prob = transition.div(
        transition.sum(axis=1).replace(0, np.nan),
        axis=0
    )

    return transition_prob, df_passes_player, number_games


def build_pitch_legend(df_passes_player, number_games, team, chosen_player):
    """
    Create a matplotlib figure with the mplsoccer pitch heatmap legend.
    """
    N_X = 6
    N_Y = 5

    fig, ax = plt.subplots(figsize=(9, 6))
    pitch = Pitch(line_zorder=2, line_color='black')
    pitch.draw(ax=ax)

    # 2D histogram on the pitch
    bin_statistic = pitch.bin_statistic(
        df_passes_player.x,
        df_passes_player.y,
        statistic='count',
        bins=(N_X, N_Y),
        normalize=False
    )
    bin_statistic["statistic"] = bin_statistic["statistic"] / number_games

    pcm = pitch.heatmap(
        bin_statistic,
        cmap='Reds',
        edgecolor='grey',
        ax=ax
    )

    # Zone labels 1–30
    x_bins = np.linspace(pitch.dim.left, pitch.dim.right, N_X + 1)
    y_bins = np.linspace(pitch.dim.bottom, pitch.dim.top, N_Y + 1)

    zone = 1
    for i in range(N_X):
        for j in range(N_Y):
            xc = (x_bins[i] + x_bins[i+1]) / 2
            yc = (y_bins[j] + y_bins[j+1]) / 2
            ax.text(
                xc, yc, str(zone),
                ha='center', va='center',
                fontsize=12, color='black', weight='bold'
            )
            zone += 1

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Passes per game", fontsize=11)

    player_short = chosen_player.split()[0]
    fig.suptitle(
        f"Pass volume per zone – {player_short} ({team})",
        fontsize=16
    )

    fig.tight_layout()
    return fig


def build_interactive_matrix(transition_prob, chosen_player, team):
    """
    Build a Plotly interactive 30x30 heatmap from transition_prob,
    including vertical + horizontal separator lines.
    """
    matrix_to_plot = transition_prob.fillna(0).values
    zones = np.arange(1, 31)

    fig = px.imshow(
        matrix_to_plot,
        x=zones,
        y=zones,
        color_continuous_scale="Reds",
        origin="lower",
        labels=dict(
            x="To zone (end of pass)",
            y="From zone (start of pass)",
            color="Probability"
        ),
    )

    player_short = chosen_player.split()[0]

    fig.update_layout(
        title=f"Pass Transition Probability (30×30) – {player_short} ({team})",
        xaxis=dict(tickmode="array", tickvals=zones, ticktext=zones),
        yaxis=dict(tickmode="array", tickvals=zones, ticktext=zones),
        width=750,
        height=750,
    )

    # Improve hover
    fig.update_traces(
        hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.3f}<extra></extra>"
    )

    # ---------------------------------------------------------
    # ADD THIRD-BOUNDARY LINES (same as matplotlib version)
    # matrix indices are 1–30 → separators at 10.5 and 20.5 
    # ---------------------------------------------------------
    third_lines = [10.5, 20.5]

    for x in third_lines:
        # Vertical line
        fig.add_shape(
            type="line",
            x0=x, x1=x,
            y0=0.5, y1=30.5,
            line=dict(color="black", width=3)
        )

    for y in third_lines:
        # Horizontal line
        fig.add_shape(
            type="line",
            x0=0.5, x1=30.5,
            y0=y, y1=y,
            line=dict(color="black", width=3)
        )

    # Keep axis square
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig



# ------------------------------------------------------------
# 3. Z-SCORE DM SCATTER PLOT
# ------------------------------------------------------------
@st.cache_data
def build_zscore_data(df_all_events):
    """Prepare the z-score dataset for DMs."""
    dm_positions = [
        "Left Defensive Midfield",
        "Center Defensive Midfield",
        "Right Defensive Midfield"
    ]
    DEF_ACTIONS = ["Pressure", "Interception", "Ball Recovery", "Block", "Duel"]

    # DMs
    mask_dm = df_all_events["position_name"].isin(dm_positions)
    dm_info = (
        df_all_events.loc[mask_dm, ["player_id", "player_name", "team_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Minutes per match and total minutes
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

    # Defensive actions
    df_def = df_all_events[df_all_events["type_name"].isin(DEF_ACTIONS)].copy()
    def_counts = (
        df_def.groupby("player_id")
        .size()
        .reset_index(name="def_actions_zone")
    )

    dm_metrics = (
        dm_info
        .merge(minutes_player, on="player_id", how="left")
        .merge(def_counts, on="player_id", how="left")
    )

    dm_metrics["minutes_played"] = dm_metrics["minutes_played"].fillna(0)
    dm_metrics["def_actions_zone"] = dm_metrics["def_actions_zone"].fillna(0)

    dm_metrics = dm_metrics[dm_metrics["minutes_played"] >= 270].copy()

    dm_metrics["def_actions_per90"] = (
        dm_metrics["def_actions_zone"] / (dm_metrics["minutes_played"] / 90)
    )

    mean_def = dm_metrics["def_actions_per90"].mean()
    std_def  = dm_metrics["def_actions_per90"].std()
    dm_metrics["z_def_actions"] = (dm_metrics["def_actions_per90"] - mean_def) / std_def

    # Dangerous passes
    df_all_events = df_all_events.copy()
    df_all_events["time_seconds"] = (
        df_all_events["minute"]*60 + df_all_events["second"]
    )

    df_shots = df_all_events[df_all_events["type_name"] == "Shot"]
    df_shots = df_shots[["match_id", "possession", "time_seconds"]]

    df_passes = df_all_events[
        (df_all_events["type_name"] == "Pass")
        & (df_all_events["outcome_name"].isnull())
        & (df_all_events["possession_team_id"] == df_all_events["team_id"])
        & (~df_all_events["sub_type_name"].isin(
            ['Throw-in','Corner','Free Kick','Kick Off','Goal Kick']
        ))
    ]

    df_merged = df_shots.merge(
        df_passes,
        on=["possession", "match_id"],
        how="inner",
        suffixes=("_shot", "")
    )

    df_merged["time_diff"] = (
        df_merged["time_seconds_shot"] - df_merged["time_seconds"]
    )
    df_danger_passes = df_merged[df_merged["time_diff"].between(0, 15)]

    first_shot = df_danger_passes.groupby("id")["time_diff"].idxmin()
    df_danger_passes = df_danger_passes.loc[first_shot]

    df_danger_passes = df_danger_passes[
        df_danger_passes["position_name"].isin(dm_positions)
    ]

    pass_count = (
        df_danger_passes
        .groupby(["player_id", "player_name", "team_name"])
        .size()
        .reset_index(name="danger_passes")
    )

    pass_count = pass_count.merge(minutes_player, on="player_id", how="left")
    pass_count = pass_count[pass_count["minutes_played"] >= 270]

    pass_count["danger_passes_per_90"] = (
        pass_count["danger_passes"] / pass_count["minutes_played"] * 90
    )

    mean_off = pass_count["danger_passes_per_90"].mean()
    std_off  = pass_count["danger_passes_per_90"].std()

    pass_count["z_danger_passes_per_90"] = (
        (pass_count["danger_passes_per_90"] - mean_off) / std_off
    )

    zdf = dm_metrics.merge(
        pass_count[
            ["player_id", "player_name", "team_name",
             "minutes_played", "danger_passes_per_90", "z_danger_passes_per_90"]
        ],
        on=["player_id", "player_name", "team_name", "minutes_played"],
        how="left"
    )

    return zdf

def build_zscore_plot_interactive(zdf):
    """
    Interactive Plotly scatter: z_def_actions vs z_danger_passes_per_90
    for all defensive midfielders.
    """
    highlight_names = [
        "Rodrigo Hernández Cascante",
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

    fig = go.Figure()

    # ---- Background: all DMs ----
    fig.add_trace(
        go.Scatter(
            x=zdf["z_def_actions"],
            y=zdf["z_danger_passes_per_90"],
            mode="markers",
            name="All DMs",
            marker=dict(color="#CCCCCC", size=7),
            hovertemplate=(
                "Player: %{customdata[0]}<br>"
                "Team: %{customdata[1]}<br>"
                "Def actions/90 (z): %{x:.2f}<br>"
                "Danger passes/90 (z): %{y:.2f}<br>"
                "Def actions/90: %{customdata[2]:.2f}<br>"
                "Danger passes/90: %{customdata[3]:.2f}<br>"
                "Minutes: %{customdata[4]:.0f}<extra></extra>"
            ),
            customdata=np.stack([
                zdf["player_name"],
                zdf["team_name"],
                zdf["def_actions_per90"],
                zdf["danger_passes_per_90"],
                zdf["minutes_played"]
            ], axis=-1)
        )
    )

    # ---- Highlighted players ----
    hi = zdf[zdf["player_name"].isin(highlight_names)].copy()
    hi["short_label"] = hi["player_name"].map(short_names).fillna(hi["player_name"])

    fig.add_trace(
        go.Scatter(
            x=hi["z_def_actions"],
            y=hi["z_danger_passes_per_90"],
            mode="markers+text",
            name="Highlighted",
            marker=dict(
                color="#1f77b4",
                size=10,
                line=dict(color="black", width=1)
            ),
            text=hi["short_label"],
            textposition="middle right",
            hovertemplate=(
                "Player: %{customdata[0]}<br>"
                "Team: %{customdata[1]}<br>"
                "Def actions/90 (z): %{x:.2f}<br>"
                "Danger passes/90 (z): %{y:.2f}<br>"
                "Def actions/90: %{customdata[2]:.2f}<br>"
                "Danger passes/90: %{customdata[3]:.2f}<br>"
                "Minutes: %{customdata[4]:.0f}<extra></extra>"
            ),
            customdata=np.stack([
                hi["player_name"],
                hi["team_name"],
                hi["def_actions_per90"],
                hi["danger_passes_per_90"],
                hi["minutes_played"]
            ], axis=-1)
        )
    )

    # ---- Axes, quadrants, layout ----
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.add_vline(x=0, line_color="gray", line_dash="dash")

    fig.update_layout(
        title="DM Comparison: Defensive vs Offensive Contribution (z-scores)",
        xaxis_title="Z-score: defensive actions per 90",
        yaxis_title="Z-score: dangerous passes per 90",
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=650
    )

    return fig


# ------------------------------------------------------------
# 4. STREAMLIT APP
# ------------------------------------------------------------
st.title("EURO 2024 – Defensive Midfielder Analysis")

df_match, df_all_events = load_data()

# Sidebar: team & player selection (used in tab 1)
team_options = sorted(df_all_events["team_name"].dropna().unique())
default_team_idx = team_options.index("Spain") if "Spain" in team_options else 0
team = st.sidebar.selectbox("Select team", team_options, index=default_team_idx)

players_team = (
    df_all_events[df_all_events["team_name"] == team]["player_name"]
    .dropna()
    .unique()
)
players_team = sorted(players_team)

default_player_idx = 0
if team == "Spain":
    for i, p in enumerate(players_team):
        if p == "Rodrigo Hernández Cascante":
            default_player_idx = i
            break

chosen_player = st.sidebar.selectbox("Select player", players_team, index=default_player_idx)

tab1, tab2 = st.tabs(["Pass-flow Map", "DM Z-Score Comparison"])

with tab1:
    st.subheader("Pass-flow Map (Interactive 30×30 Matrix + Pitch Legend)")
    st.write(
        "The heatmap shows the probability of Rodri (or chosen player) moving the ball "
        "from one zone (row) to another (column). The legend below shows where on the pitch "
        "those passes start, normalized per game."
    )
    try:
        transition_prob, df_passes_player, number_games = compute_transition_and_passes(
            df_all_events, team, chosen_player
        )

        # Interactive matrix
        matrix_fig = build_interactive_matrix(transition_prob, chosen_player, team)
        st.plotly_chart(matrix_fig, use_container_width=True)

        # Pitch legend (matplotlib)
        legend_fig = build_pitch_legend(df_passes_player, number_games, team, chosen_player)
        st.pyplot(legend_fig)

    except ValueError as e:
        st.error(str(e))

with tab2:
    st.subheader("Defensive vs Offensive Contribution (DMs)")
    st.write(
        "Each point represents a defensive midfielder (≥270 minutes). "
        "The x-axis is defensive actions per 90 (z-score) and the y-axis is "
        "dangerous passes per 90 (z-score). Hover over a point to see player details."
    )
    zdf = build_zscore_data(df_all_events)
    z_fig = build_zscore_plot_interactive(zdf)
    st.plotly_chart(z_fig, use_container_width=True)

