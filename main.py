from os import read
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from Data_viz.final_data_viz import *


# Reading data charts for plots
def read_chartsdf():
    # read and process the charts dataset
    charts_df = pd.read_csv('data/spotify_daily_charts.csv')
    # transform date column into a datetime column
    charts_df['date'] = pd.to_datetime(charts_df['date'])

    return charts_df

def read_tracksdf():
    tracks_df = pd.read_csv('data/spotify_daily_charts_tracks.csv')
    return tracks_df

def read_zoodf():
    zoo_df=pd.read_csv("data_sp/zoo_tracks_data.csv")
    return zoo_df

def read_mainstaydf():
    mainstay_df=pd.read_csv("data_sp/tracks_data.csv")
    return mainstay_df

def read_streamsdf(charts_df = read_chartsdf(), tracks_df = read_tracksdf()):
    streams_df = charts_df.merge(tracks_df, on='track_id', how='left')
    streams_df = streams_df.drop(columns='track_name_y')
    streams_df = streams_df.rename(columns={'track_name_x': 'track_name'})
    streams_df['date']=pd.to_datetime(streams_df['date'])
    streams_df.set_index("date", inplace=True)
    
    return streams_df

def read_streams_df_opm(streams_df = read_streamsdf()):
    artists_df = pd.read_csv('data/spotify_daily_charts_artists_edited.csv')
    artists_df["OPM"]=np.where(artists_df.genres.str.contains('opm'),1,0)

    streams_df_opm=pd.merge(streams_df.reset_index(),artists_df[["artist_id","OPM"]], on="artist_id", how="left")
    streams_df_opm.set_index("date")
    streams_df_opm=streams_df_opm[streams_df_opm.OPM==1]
    streams_df_opm.set_index('date',inplace=True)

    return streams_df_opm

def read_songs_reco():
    songs_reco = pd.read_csv('data/Song_Recommendations_Final.csv')
    return songs_reco

# Streamlit codes
sns.set_theme(style="white", palette=sns.color_palette("Set2"))

def project_overview():
    st.image('images/front_page.png')

def artist_overview():
    col1, col2, col3 = st.columns([1.5,2,0.5])
    with col1:
        st.text('')
    with col2:
        st.title('Artist Background')
    with col3:
        st.text('')

    col1, col2, col3 = st.columns([0.2,1,1])
    with col1:
        st.text('')
    with col2:
        st.image('images/page2.png')
    with col3:
        st.subheader('I Belong to the Zoo')
        st.text(
            '''
            - Filipino indie pop rock band that became active on 2014
            - Expanded to a five piece band after their first album I Belong to the Zoo 
            - Became popular, mainly because of their hits Sana on June 2018 
              and Balang Araw on October 2018 (4th Wish 107.5 Music Awards)
            ''')
    st.markdown('***')
    st.subheader('Recent Struggles after their Initial Hits')
    st.pyplot(plot_4charts("I Belong to the Zoo", read_streamsdf()))
    st.text(
        '''
        - Streams peaked in 2019
        - Only 2 songs charted
            - Sana
            - Balang Araw
        - Sharp decrease in monthly streams and average position
        '''
    )
    st.markdown('***')
    st.subheader('How Did they Come up with Hits Before?')
    col1, col2 = st.columns([2,0.4])
    with col1:
        st.pyplot(plot_2boxcharts(read_zoodf()))
    with col2:
        st.text('')
        st.text('')
        st.text('''
        Their previous hits were:
        - High tempo
        - Acoustic
        - Danceable Songs
        ''')

def opm_mainstay():
    st.title('OPM Mainstay Artists')
    title = '<p style="font-size: 30px;"><b>Mainstay Tracks:</b></p>'            
    st.markdown(
        f'''
        {title} \n
        - Top 15th percentile of appearances in Top 200 \n
        - Average position is in the top 25% of the Top 200''', unsafe_allow_html=True)
    option = st.selectbox(
            'I Belong to the Zoo vs Mainstay Artists: ',
            ('I Belong to the Zoo', 'Zack Tabudlo', 'Adie', 'Arthur Nery', 'NOBITA', 'Ben&Ben'))
    col1, col2 = st.columns([0.5,1])
    if option == 'I Belong to the Zoo':
        with col1:
            st.image('images/page2.png')
        with col2:
            st.pyplot(plot_marketing_strat("I Belong to the Zoo",13, 1, read_streamsdf()))
    elif option == 'Zack Tabudlo':
        with col1:
            st.image('images/zack.png')
        with col2:
            st.pyplot(plot_marketing_strat("Zack Tabudlo",13, 5, read_streamsdf()))
    elif option == 'Adie':
        with col1:
            st.image('images/adic.png')
        with col2:
            st.pyplot(plot_marketing_strat("Adie",10, 3, read_streamsdf()))
    elif option == 'Arthur Nery':
        with col1:
            st.image('images/nery.png')
        with col2:
            st.pyplot(plot_marketing_strat("Arthur Nery",13, 5, read_streamsdf()))
    elif option == 'NOBITA':
        with col1:
            st.image('images/nobita.png')
        with col2:
            st.pyplot(plot_marketing_strat("NOBITA",13, 2, read_streamsdf()))
    elif option == 'Ben&Ben':
        with col1:
            st.image('images/ben.png')
        with col2:
            st.pyplot(plot_marketing_strat("Ben&Ben",13, 8, read_streamsdf()))
    
    with st.expander('Area Chart of 5 Mainstay Artists vs Other OPM Artists Streams'):
        st.pyplot(plot_market_eda(read_tracksdf(),read_streams_df_opm()))
        
def mainstay_learning():
    st.title('What can we learn from these Mainstay Artists?')
    st.markdown('***')
    st.subheader('Song Composition')
    st.pyplot(plot_6boxcharts(read_tracksdf(), read_mainstaydf()))
    st.text(
        '''
        - Try slower tempo, and lower volume (less loud) songs to make the listeners feel the “hugot” more.
        - Create more live tracks, very important for bands to have a solid audience following.

        '''
        )
    st.markdown('***')
    st.subheader('Marketing Strategies')
    st.image('images/market_insights.png')
    st.text('')
    st.text('')
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('**Strategic Release/Timing of Music Videos**')
        st.image('images/timeline.png')
        st.text('Besides the initial release of the mainstay artists’ songs, these strategies \nhave been utilized to extend the “hype” of their songs:')
        st.text(
            '''
            - Release of a lyric video followed by an official music video
            - Featuring famous actresses/actors in their music video
            - Promoting their song as a movie/teleserye soundtrack
            - Live performances of their songs
            ''')
    with col2:
        st.markdown('**Collaboration with famous actresses/actors in music videos**')
        st.image('images/collab1.png')
        st.image('images/collab2.png')

    st.text('')
    st.text('')
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown('**Movie/Teleserye Soundtracks**')
        st.image('images/soundtracks1.png')
        st.image('images/soundtracks2.png')
    
    with col2:
        st.markdown('**Live Performances/Exposure in Important Events**')
        st.image('images/performance1.png')
        st.image('images/performance2.png')


def recommender_engine():
    st.title('Recommended Artists for Collaboration')
    st.text(
        '''
        To look for the best artists for the band to collaborate with, we built a genre classification model trained with audio features of OPM Data. The model 
        had the following features:

        •    Support Vector Machine algorithm using radial kernel and gamma = 1
        •    OPM Genre classifications: Rock, Rap, Reggae, Jazz, Dance, Acoustic
        •    Audio features used for training: danceability, energy, acousticness, valence, speechiness

        A representative track was then made using the top ten popular songs of mainstay artists Arthur Nery, Adie, Zack Tabudlo, Ben&Ben, Nobita 
        along with the top ten of I Belong To The Zoo. This was chosen to increase the likelihood of creating mainstay songs, while still retaining the 
        authentic musicality of our band. 

        Song similarity was compared between the representative track and other popular OPM songs based on the cosine distance of their audio features and 
        predicted genre possibilities. 

        The data shows Acoustic, Rock, and Jazz to be the most similar genre to the representative track. For each of these three genres, we chose the top five songs 
        with the greatest similarity to constitute our playlist recommendation.
        
        Based on the recommendation track, a list of artists were named as possible collaborators with our band; with Acoustic, Rock, or Jazz as 
        the recommended genre for the collab project.
        '''
        )

    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',\
                'liveness', 'valence', 'tempo']
    df = read_songs_reco()
    with st.expander('Recommended Artists Dataframe'):
        st.dataframe(df[['track_name','artist_name', 'predicted_genre', 'cosine_dist_mod2'] + feature_cols])

def recommendations():
    st.title('The Playlist')
    components.iframe("https://open.spotify.com/embed/playlist/20R3vDgunfzviJmCbJsh3e", height=380, scrolling=True)


list_of_pages = [
    "Title Page",
    "Artist Overview",
    "OPM Mainstay Artists",
    "Mainstay Insights",
    "Recommender Engine",
    "Playlist"
]

st.sidebar.title(':scroll: Main Pages')
selection = st.sidebar.radio("Go to: ", list_of_pages)

if selection == "Title Page":
    project_overview()

elif selection == "Artist Overview":
    artist_overview()

elif selection == "OPM Mainstay Artists":
    opm_mainstay()

elif selection == "Mainstay Insights":
    mainstay_learning()

elif selection == "Recommender Engine":
    recommender_engine()

elif selection == "Playlist":
    recommendations()