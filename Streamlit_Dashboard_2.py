pip install statsbombpy
pip install mplsoccer
from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Sbopen, pitch
import seaborn as sns
from mplsoccer import VerticalPitch, add_image
from matplotlib.colors import to_rgba
from mplsoccer import Pitch, FontManager, Sbopen
import unicodedata
import streamlit as st
parser = Sbopen()
competitions = parser.competition()
competition_id = competitions.competition_id
competition_id = competition_id.unique()
def get_string_element(shot_events, column_name, condition_column, condition_value):
    """
    Get a string element from a particular column in the DataFrame based on the given condition.

    Parameters:
        shot_events (pd.DataFrame): The input Pandas DataFrame.
        column_name (str): The name of the column from which to extract the string element.
        condition_column (str): The name of the column to apply the condition.
        condition_value: The value used for the condition to filter the DataFrame.

    Returns:
        str or None: The string element from the specified column that meets the condition.
                     Returns None if no matching element is found.
    """

    filtered_df =   shot_events[shot_events[condition_column] == condition_value]
    
    if not filtered_df.empty:
        return str(filtered_df[column_name].iloc[0])
    else:
        return None
def get_season_ids(competition_id):
    season_ids = competitions[competitions.competition_id==competition_id].season_id
    return season_ids
def get_match_ids(competition_id,season_id):
    matches = parser.match(competition_id,season_id)
    match_ids = matches.match_id
    return match_ids
def get_team_name(match_id):
    events, related, freeze, players = parser.event(match_id)
    team_names = events.team_name.unique()
    return team_names
def get_formation(match_id,team_names):
    events, related, freeze, players = parser.event(match_id)
    formations = events[events['team_name']==team_names].tactics_formation.unique()
    return formations
def get_shots_id(match_id):
    df_event = parser.event(match_id)[0]
    shot_events = df_event[(df_event.type_name=='Shot')].copy()
    shot_events = shot_events[['id','minute','player_name','team_name']]
    shot_events['player_name_minute']=shot_events['minute'].astype(str)+'-'+shot_events['player_name']
    shot_events=shot_events.reset_index()
    return shot_events
def get_shot(match_id,SHOT_ID):
# get event and lineup dataframes for game 7478
# event data

    df_event, df_related, df_freeze, df_tactics = parser.event(match_id)

    # lineup data
    df_lineup = parser.lineup(match_id)
    df_lineup = df_lineup[['player_id', 'jersey_number', 'team_name']].copy()

    ##############################################################################
    # Subset a shot
    df_freeze_frame = df_freeze[df_freeze.id == SHOT_ID].copy()
    df_shot_event = df_event[df_event.id == SHOT_ID].dropna(axis=1, how='all').copy()

    # add the jersey number
    df_freeze_frame = df_freeze_frame.merge(df_lineup, how='left', on='player_id')

    ##############################################################################
    # Subset the teams

    # strings for team names
    team1 = df_shot_event.team_name.iloc[0]
    team2 = list(set(df_event.team_name.unique()) - {team1})[0]

    # subset the team shooting, and the opposition (goalkeeper/ other)
    df_team1 = df_freeze_frame[df_freeze_frame.team_name == team1]
    df_team2_goal = df_freeze_frame[(df_freeze_frame.team_name == team2) &
                                    (df_freeze_frame.position_name == 'Goalkeeper')]
    df_team2_other = df_freeze_frame[(df_freeze_frame.team_name == team2) &
                                    (df_freeze_frame.position_name != 'Goalkeeper')]

    ##############################################################################
    # Plotting

    # Setup the pitch
    pitch = VerticalPitch(half=True, goal_type='box', pad_bottom=-20)

    # We will use mplsoccer's grid function to plot a pitch with a title axis.
    fig, axs = pitch.grid(figheight=8, endnote_height=0,  # no endnote
                        title_height=0.1, title_space=0.02,
                        # Turn off the endnote/title axis. I usually do this after
                        # I am happy with the chart layout and text placement
                        axis=False,
                        grid_height=0.83)

    # Plot the players
    sc1 = pitch.scatter(df_team1.x, df_team1.y, s=600, c='#727cce', label='Attacker', ax=axs['pitch'])
    sc2 = pitch.scatter(df_team2_other.x, df_team2_other.y, s=600,
                        c='#5ba965', label='Defender', ax=axs['pitch'])
    sc4 = pitch.scatter(df_team2_goal.x, df_team2_goal.y, s=600,
                        ax=axs['pitch'], c='#c15ca5', label='Goalkeeper')

    # plot the shot
    sc3 = pitch.scatter(df_shot_event.x, df_shot_event.y, marker='football',
                        s=600, ax=axs['pitch'], label='Shooter', zorder=1.2)
    line = pitch.lines(df_shot_event.x, df_shot_event.y,
                    df_shot_event.end_x, df_shot_event.end_y, comet=True,
                    label='shot', color='#cb5a4c', ax=axs['pitch'])

    # plot the angle to the goal
    pitch.goal_angle(df_shot_event.x, df_shot_event.y, ax=axs['pitch'], alpha=0.2, zorder=1.1,
                    color='#cb5a4c', goal='right')

    # fontmanager for google font (robotto)
    robotto_regular = FontManager()

    # plot the jersey numbers
    for i, label in enumerate(df_freeze_frame.jersey_number):
        pitch.annotate(label, (df_freeze_frame.x[i], df_freeze_frame.y[i]),
                    va='center', ha='center', color='white',
                    fontproperties=robotto_regular.prop, fontsize=15, ax=axs['pitch'])

    # add a legend and title
    legend = axs['pitch'].legend(loc='center left', labelspacing=1.5)
    for text in legend.get_texts():
        text.set_fontproperties(robotto_regular.prop)
        text.set_fontsize(20)
        text.set_va('center')

    # title
    axs['title'].text(0.5, 0.5, f'{df_shot_event.player_name.iloc[0]}\n{team1} vs. {team2}',
                    va='center', ha='center', color='black',
                    fontproperties=robotto_regular.prop, fontsize=25)

    st.pyplot(fig) 
def generate_cumulative_xg_plot(match_id):
    # Fetch events data for the specified match_id
    df = sb.events(match_id=match_id)

    # Filter out rows where xG value is not available
    df = df[df.shot_statsbomb_xg.notna()]

    # Select relevant columns for analysis
    df = df[['team', 'timestamp', 'type', 'under_pressure', 'shot_body_part', 'player', 'shot_outcome',
             'shot_statsbomb_xg', 'play_pattern', 'period', 'location']].reset_index(drop=True)

    # Convert timestamp column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate the minute of each data point
    df['minute'] = (df['period'] - 1) * 45 + df['timestamp'].dt.minute

    # Adjust for periods 3 and 4
    df.loc[df['period'] > 2, 'minute'] = 90 + (df['period'] - 3) * 15 + df['timestamp'].dt.minute

    # Extract x and y coordinates from the location column
    df['x'] = df['location'].apply(lambda x: x[0])
    df['y'] = df['location'].apply(lambda x: x[1])

    # Set plot size
    fig = plt.figure(figsize=(14, 10))

    team_name = df['team'].unique()

    # Calculate cumulative sum of shot_statsbomb_xg for each team
    team1_xg_cumulative = df[(df['team'] == team_name[0]) & (df['period'] < 3)]['shot_statsbomb_xg'].cumsum()
    team2_xg_cumulative = df[(df['team'] == team_name[1]) & (df['period'] < 3)]['shot_statsbomb_xg'].cumsum()

    # Plot the cumulative sum of shot_statsbomb_xg for each team against the minute values
    plt.plot(team1_xg_cumulative, label=team_name[0], color='red')
    plt.plot(team2_xg_cumulative, label=team_name[1], color='black')

    # Add title, x-axis and y-axis labels, and legend to the plot
    plt.title('Cumulative xG Plot', fontsize=16)
    plt.xlabel('Minute', fontsize=14)
    plt.ylabel('Expected Goals', fontsize=14)
    plt.legend(prop={'size': 16})

    # Remove the x-ticks text
    plt.xticks([])

    # Add text showing the maximum value of each team's cumulative xG
    max_team1_xg = max(team1_xg_cumulative)
    max_team2_xg = max(team2_xg_cumulative)
    plt.text(43, max_team1_xg, f"Max {team_name[0]} xG: {max_team1_xg:.2f}", fontsize=14, color='red', ha='right')
    plt.text(43, max_team2_xg, f"Max {team_name[1]} xG: {max_team2_xg:.2f}", fontsize=14, color='black', ha='right')
    st.pyplot(fig)

def shot_map(match_id,team):
    df = sb.events(match_id = match_id)
    df = df[df.shot_statsbomb_xg.isna()==False]
    df = df[['team', 'timestamp', 'type','under_pressure','shot_body_part','shot_statsbomb_xg','player','location']].reset_index(drop=True)
    df = df[df.team==team]
    df['x'] =   df['location'].apply(lambda x: x[0])
    df['y'] =   df['location'].apply(lambda x: x[1])
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, goal_type='box', goal_alpha=0.01, pitch_color='white', line_color='black')
    fig, axs = pitch.grid(figheight=10, title_height=0, endnote_space=0, axis=False,title_space=0, grid_height=0.82)
    fig.set_facecolor("#22312b")
    #plotting
    scatter_shots = pitch.scatter(df.x, df.y, s=(df.shot_statsbomb_xg * 800)+30, c=df.shot_statsbomb_xg, edgecolors='black', marker='o', ax=axs['pitch'])
    # add text
    axs['pitch'].text(35, 121,'Shots map of '+team,color='Black',size=14)
    cbar = fig.colorbar(scatter_shots, ax=axs['pitch'])
    cbar.ax.set_ylabel('Expected Goal Value', fontsize=14)
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(labelcolor='white')
    st.pyplot(fig)

def passing_network(match_id,team_name,formation):

    ##############################################################################
    # Set team and match info, and get event and tactics dataframes for the defined match_id
    parser = Sbopen()
    events, related, freeze, players = parser.event(match_id)
    TEAM = team_name

    ##############################################################################
    # Adding on the last tactics id and formation for the team for each event

    events.loc[events.tactics_formation.notnull(), 'tactics_id'] = events.loc[
        events.tactics_formation.notnull(), 'id']
    events[['tactics_id', 'tactics_formation']] = events.groupby('team_name')[[
        'tactics_id', 'tactics_formation']].ffill()

    ##############################################################################
    # Add the abbreviated player position to the players dataframe

    formation_dict = {1: 'GK', 2: 'RB', 3: 'RCB', 4: 'CB', 5: 'LCB', 6: 'LB', 7: 'RWB',
                    8: 'LWB', 9: 'RDM', 10: 'CDM', 11: 'LDM', 12: 'RM', 13: 'RCM',
                    14: 'CM', 15: 'LCM', 16: 'LM', 17: 'RW', 18: 'RAM', 19: 'CAM',
                    20: 'LAM', 21: 'LW', 22: 'RCF', 23: 'ST', 24: 'LCF', 25: 'SS'}
    players['position_abbreviation'] = players.position_id.map(formation_dict)

    ##############################################################################
    # Add on the subsitutions to the players dataframe, i.e. where players are subbed on
    # but the formation doesn't change

    sub = events.loc[events.type_name == 'Substitution',
                    ['tactics_id', 'player_id', 'substitution_replacement_id',
                    'substitution_replacement_name']]
    players_sub = players.merge(sub.rename({'tactics_id': 'id'}, axis='columns'),
                                on=['id', 'player_id'], how='inner', validate='1:1')
    players_sub = (players_sub[['id', 'substitution_replacement_id', 'position_abbreviation']]
                .rename({'substitution_replacement_id': 'player_id'}, axis='columns'))
    players = pd.concat([players, players_sub])
    players.rename({'id': 'tactics_id'}, axis='columns', inplace=True)
    players = players[['tactics_id', 'player_id', 'position_abbreviation']]

    ##############################################################################
    # Add player position information to the events dataframe

    # add on the position the player was playing in the formation to the events dataframe
    events = events.merge(players, on=['tactics_id', 'player_id'], how='left', validate='m:1')
    # add on the position the receipient was playing in the formation to the events dataframe
    events = events.merge(players.rename({'player_id': 'pass_recipient_id'},
                                        axis='columns'), on=['tactics_id', 'pass_recipient_id'],
                        how='left', validate='m:1', suffixes=['', '_receipt'])

    ##############################################################################
    # Show the formations used in the match

    ##############################################################################
    # Filter passes by chosen formation, then group all passes and receipts to
    # calculate avg x, avg y, count of events for each slot in the formation

    FORMATION = formation
    pass_cols = ['id', 'position_abbreviation', 'position_abbreviation_receipt']
    passes_formation = events.loc[(events.team_name == TEAM) & (events.type_name == 'Pass') &
                                (events.tactics_formation == FORMATION) &
                                (events.position_abbreviation_receipt.notnull()), pass_cols].copy()
    location_cols = ['position_abbreviation', 'x', 'y']
    location_formation = events.loc[(events.team_name == TEAM) &
                                    (events.type_name.isin(['Pass', 'Ball Receipt'])) &
                                    (events.tactics_formation == FORMATION), location_cols].copy()

    # average locations
    average_locs_and_count = (location_formation.groupby('position_abbreviation')
                            .agg({'x': ['mean'], 'y': ['mean', 'count']}))
    average_locs_and_count.columns = ['x', 'y', 'count']

    # calculate the number of passes between each position (using min/ max so we get passes both ways)
    passes_formation['pos_max'] = (passes_formation[['position_abbreviation',
                                                    'position_abbreviation_receipt']]
                                .max(axis='columns'))
    passes_formation['pos_min'] = (passes_formation[['position_abbreviation',
                                                    'position_abbreviation_receipt']]
                                .min(axis='columns'))
    passes_between = passes_formation.groupby(['pos_min', 'pos_max']).id.count().reset_index()
    passes_between.rename({'id': 'pass_count'}, axis='columns', inplace=True)

    # add on the location of each player so we have the start and end positions of the lines
    passes_between['pos_min']=passes_between['pos_min'].astype(object)
    passes_between['pos_max']=passes_between['pos_max'].astype(object)
    passes_between = passes_between.merge(average_locs_and_count, left_on='pos_min', right_index=True)
    passes_between = passes_between.merge(average_locs_and_count, left_on='pos_max', right_index=True,
                                        suffixes=['', '_end'])

    ##############################################################################
    # Calculate the line width and marker sizes relative to the largest counts

    MAX_LINE_WIDTH = 18
    MAX_MARKER_SIZE = 3000
    passes_between['width'] = (passes_between.pass_count / passes_between.pass_count.max() *
                            MAX_LINE_WIDTH)
    average_locs_and_count['marker_size'] = (average_locs_and_count['count']
                                            / average_locs_and_count['count'].max() * MAX_MARKER_SIZE)

    ##############################################################################
    # Set color to make the lines more transparent when fewer passes are made

    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('white'))
    color = np.tile(color, (len(passes_between), 1))
    c_transparency = passes_between.pass_count / passes_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    ##############################################################################

    ##############################################################################
    # Plot the chart again with a title.
    # We will use mplsoccer's grid function to plot a pitch with a title and endnote axes.
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')

    fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0,
                        axis=False,
                        title_space=0, grid_height=0.82, endnote_height=0.05)
    fig.set_facecolor("#22312b")
    pass_lines = pitch.lines(passes_between.x, passes_between.y,
                            passes_between.x_end, passes_between.y_end, lw=passes_between.width,
                            color=color, zorder=1, ax=axs['pitch'])
    pass_nodes = pitch.scatter(average_locs_and_count.x, average_locs_and_count.y,
                            s=average_locs_and_count.marker_size,
                            color='red', edgecolors='black', linewidth=1, alpha=1, ax=axs['pitch'])
    for index, row in average_locs_and_count.iterrows():
        pitch.annotate(row.name, xy=(row.x, row.y), c='white', va='center',
                    ha='center', size=16, weight='bold', ax=axs['pitch'])

    # Load a custom font.
    URL = 'https://raw.githubusercontent.com/google/fonts/main/apache/roboto/Roboto%5Bwdth,wght%5D.ttf'
    robotto_regular = FontManager(URL)

    # endnote /title
    axs['endnote'].text(1, 0.5, '@sreerajsrikanth', color='#c7d5cc',
                        va='center', ha='right', fontsize=15,
                        fontproperties=robotto_regular.prop)
    TITLE_TEXT = f'{TEAM}, {FORMATION} formation'
    axs['title'].text(0.5, 0.7, TITLE_TEXT, color='#c7d5cc',
                    va='center', ha='center', fontproperties=robotto_regular.prop, fontsize=30)

    # sphinx_gallery_thumbnail_path = 'gallery/pitch_plots/images/sphx_glr_plot_pass_network_002.png'
    st.pyplot(fig)





st.set_page_config(page_title='Match Analysis', page_icon=':soccer:', initial_sidebar_state='expanded')
st.sidebar.markdown('## Select Football Game')
cc1,cc2 = st.columns([1,1])
select_competition = st.sidebar.selectbox('Select Competition', competition_id,index=0)
if select_competition:
    seasons = get_season_ids(select_competition)
    select_season = st.sidebar.selectbox('Select season',seasons,index=0)
if select_competition and select_season:
    matches = get_match_ids(select_competition,select_season)
    select_match = st.sidebar.selectbox('Select Match',matches,index=0)

if select_match:
    shots = get_shots_id(select_match)
    shot_player_minute = shots.player_name_minute
    select_shot = st.sidebar.selectbox('Select Shot Map',shot_player_minute,index=0)
    get_shot_id = get_string_element(shot_events=shots,column_name='id',condition_column='player_name_minute',condition_value=select_shot)
    teams = get_team_name(select_match)
    select_team = st.sidebar.selectbox('Select Team',teams,index=0)
    formations = get_formation(select_match,select_team)
    select_formations = st.selectbox('Select Formation',formations,index=0)
    with cc1:
        generate_cumulative_xg_plot(select_match)
        if select_team:
            shot_map(select_match,select_team)
        
    with cc2:   
        if select_team and select_formations:    
            passing_network(select_match,select_team,select_formations)
        if select_shot:
            get_shot(select_match,get_shot_id)
    
    
    
    

