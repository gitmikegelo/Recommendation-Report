import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt


# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler

#Color Pallette
COLOR1="#F2EFDC"
COLOR2="#1F1F1F"
COLOR3="#FFFFFF"

#Secondary Colors
COLOR4="skyblue"
COLOR5="#58D68D"
COLOR6="#F5CBA7"
COLOR7="#F9E79F"
COLOR8="#C39BD3"
COLOR9="FDD4FD"

# def read_chartsdf():
#     # read and process the charts dataset
#     charts_df = pd.read_csv('../data/spotify_daily_charts.csv')
#     # transform date column into a datetime column
#     charts_df['date'] = pd.to_datetime(charts_df['date'])

#     return charts_df

# def read_tracksdf():
#     tracks_df = pd.read_csv('../data/spotify_daily_charts_tracks.csv')
#     return tracks_df

# def read_streamsdf(charts_df = read_chartsdf(), tracks_df = read_tracksdf()):
#     streams_df = charts_df.merge(tracks_df, on='track_id', how='left')
#     streams_df = streams_df.drop(columns='track_name_y')
#     streams_df = streams_df.rename(columns={'track_name_x': 'track_name'})
#     streams_df['date']=pd.to_datetime(streams_df['date'])
#     streams_df.set_index("date", inplace=True)
    
#     return streams_df

# def read_streams_df_opm(streams_df = read_streamsdf):
#     artists_df = pd.read_csv('../data/spotify_daily_charts_artists_edited.csv')
#     artists_df["OPM"]=np.where(artists_df.genres.str.contains('opm'),1,0)

#     streams_df_opm=pd.merge(streams_df.reset_index(),artists_df[["artist_id","OPM"]], on="artist_id", how="left")
#     streams_df_opm.set_index("date")
#     streams_df_opm=streams_df_opm[streams_df_opm.OPM==1]
#     streams_df_opm.set_index('date',inplace=True)

#     return streams_df_opm

def plot_4charts(artist, streams_df, selected_color=COLOR2, bg_color=COLOR1):
    data1=streams_df[streams_df.artist==artist]['streams'].resample("MS").sum()/1000000
    #data2=streams_df[streams_df.artist==artist]['track_id'].resample("MS").count()
    data2=streams_df[streams_df.artist==artist]['streams'].resample("MS").sum().cumsum()/1000000
    data3=data1.pct_change()
    data4=streams_df[streams_df.artist==artist]['position'].resample("MS").mean()

    #line chart of monthly streams
    figure = plt.figure(figsize=(15,4), dpi=200)
    ax1 = figure.add_subplot(141)
    ax2 = figure.add_subplot(142)
    ax3 = figure.add_subplot(143)
    ax4 = figure.add_subplot(144)
    #default is line so you can omit kind= parameter
    
    '''
    data1[:-1].plot(ax=ax1, kind='line')
    data2.plot(ax=ax2, kind='line')
    data3[:-1].plot(ax=ax3, kind='line')
    data4[:-1].plot(ax=ax4, kind='line')
    '''
    
    
    #'''
    #selected_color=color3
    data1.plot(ax=ax1, kind='line', color=selected_color)
    data2.plot(ax=ax2, kind='line' , color=selected_color)
    data3.plot(ax=ax3, kind='line', color=selected_color)
    data4.plot(ax=ax4, kind='line', color=selected_color)
    #'''
    


    #Uncomment for cleaner x labels
    #ax1.set_xticklabels([x.strftime('%Y-%m') for x in ar_streams.index])

    #ax1.set_xlabel('Date')
    ax1.set_ylabel('common ylabel')

    ax1.set_title('Monthly Streams (%s)' % artist)
    ax2.set_title('Cumulative Streams')
    ax3.set_title('Monthly % Change')
    ax4.set_title('Avg Position in the Top 200')

    ax1.set_facecolor(bg_color)
    ax2.set_facecolor(bg_color)
    ax3.set_facecolor(bg_color)
    ax4.set_facecolor(bg_color)
    
    ax4.set_yticks([1]+np.arange(0, 210, 50).tolist())
    ax4.set_ylim([200, 1])

    return figure

def plot_2boxcharts(zoo_df):
    # zoo_df=pd.read_csv("../data_sp/zoo_tracks_data.csv")

    scaler = MinMaxScaler()
    zoo_df['loudness'] = scaler.fit_transform(zoo_df[['loudness']])
    zoo_df['tempo'] =  scaler.fit_transform(zoo_df[['tempo']])

    features = ['tempo','acousticness','danceability', 'loudness','energy','valence'  , 'liveness',]  

    columns_to_view = ['track_id'] + features

    df_features = zoo_df[columns_to_view].copy()

    #df_features['artist'] = [artist if artist in comb_names else 'all else'
                        #for artist in df_features['artist_name'].values]
        
    # set multiindex
    df_features = df_features.set_index(['track_id'])

    # reshape by pd.stack to achieve shape demanded by boxplot
    df_features_stacked = pd.DataFrame({'value': df_features.stack()})
    df_features_stacked = df_features_stacked.reset_index()
    df_features_stacked = df_features_stacked.rename(columns={'level_1': 'feature'})

    df_features_stacked["track_id"]=df_features_stacked["track_id"].apply(lambda x:
                                                                        "Hit Songs" if ( (x=="1X4l4i472kW5ofFP8Xo0x0") | \
                                                                            (x=="5NXdUJ3Z2jhlp2u1cj6f7m")) else 
                                                                        "Other Songs")

        
    df_features_stacked

    figure = plt.figure(figsize=(15, 6), dpi=200)
    ax = plt.subplot(111)

    sns.boxplot(data=df_features_stacked, x='feature', y='value',  hue='track_id', ax=ax,
                hue_order=['Hit Songs',"Other Songs"] , palette=[COLOR4, COLOR1])#, boxprops=dict(alpha=.3))

    ax.legend(loc='upper center', bbox_to_anchor=(
    0.5, -0.1), frameon=False, ncol=3)
    ax.axvspan(-0.625, 2.5, facecolor=COLOR6, alpha=0.1)

    plt.title("Audio Features- I Belong to the Zoo Hit Songs vs Their Other Songs",size=15)

    return figure

def plot_market_eda(tracks_df, streams_df_opm):
    artist_dict=pd.Series(tracks_df.artist_name.values,index=(tracks_df.artist_id)).to_dict()
    df_3_groups = streams_df_opm\
        .groupby(['artist_id'])['streams']\
        .resample('M').sum().reset_index()\
        .sort_values('streams', ascending=False)

    #df_ed = df_ed.set_index('date')
    df_3_groups['album_name'] = df_3_groups['artist_id'].apply(lambda x:
                                                'Big3' if ( (artist_dict.get(x) == 'Arthur Nery') | \
                                                            (artist_dict.get(x) == 'Adie') |
                                                            (artist_dict.get(x) == 'Zack Tabudlo') ) else
                                                'Dominant Bands' if ((artist_dict.get(x) == 'Ben&Ben') |
                                                            (artist_dict.get(x) == 'NOBITA'))
                                                            else 'OPM Others')

    #df_3_groups = df_3_groups.set_index('date')
    df_3_groups=df_3_groups.groupby(["date","album_name"])[["streams"]].sum().reset_index()   

    figure = plt.figure(figsize=(4, 2),dpi=200)
    ax = plt.subplot(111)

    color_list = ['skyblue', COLOR2, COLOR1]
    # reshape
    data = df_3_groups.pivot(index='date', columns='album_name', values='streams')
    # normalize with monthly sums
    data[data.columns] = 100*data[data.columns].div(data.sum(axis=1), axis=0)
    data=data.loc['2021-01-01':'2022-12-31']

    # plot
    data.plot.area(ax=ax, lw=0, color=color_list)

    # custom ticks
    plt.yticks(np.arange(0, 120, 20), [str(x)+'%' for x in np.arange(0, 120, 20)])
    plt.ylabel('Streams')
    plt.ylim([0, 100])

    plt.xlabel('')
    plt.title("5 Mainstays vs Other OPM Artists")
    # Put a legend below current axis
    #ax.legend(loc='upper center', bbox_to_anchor=(
        #0.5, -0.25), frameon=False, ncol=3)

    ax.legend(loc='right',ncol=1,prop={'size': 6},bbox_to_anchor=(
        0.35,0.85))
        
    return figure

def plot_6boxcharts(tracks_df, mainstay_df):

    tracks_dict=pd.Series(tracks_df.track_name.values,index=(tracks_df.track_id)).to_dict()
    top5_names=["Arthur Nery", "Adie","Zack Tabudlo","Ben&Ben","NOBITA"]
    artist_name=["I Belong to the Zoo"]
    comb_names=top5_names

    # mainstay_df=pd.read_csv("../data_sp/tracks_data.csv")

    scaler = MinMaxScaler()
    mainstay_df['loudness'] = scaler.fit_transform(mainstay_df[['loudness']])
    mainstay_df['tempo'] =  scaler.fit_transform(mainstay_df[['tempo']])

    features = ['danceability', 'energy', 'loudness', 'acousticness', 'liveness', 'valence', 'tempo']
    columns_to_view = ['artist_name', 'track_name'] + features

    df_features = mainstay_df[columns_to_view].copy()

    #df_features['artist'] = [artist if artist in comb_names else 'all else'
                        #for artist in df_features['artist_name'].values]
        
    # set multiindex
    df_features = df_features.set_index(['track_name', 'artist_name'])

    # reshape by pd.stack to achieve shape demanded by boxplot
    df_features_stacked = pd.DataFrame({'value': df_features.stack()})
    df_features_stacked = df_features_stacked.reset_index()
    df_features_stacked = df_features_stacked.rename(columns={'level_2': 'feature'})

    df_features_stacked['artist_name']=df_features_stacked['artist_name'].apply(lambda x: x if (x=="I Belong to the Zoo") else ( \
                                                                            "Big3" if ((x=="Arthur Nery") | \
                                                                           (x=="Adie") | (x=="Zack Tabudlo")) 
                                                                                            else "Dominant Bands"))

    figure = plt.figure(figsize=(15, 6), dpi=200)
    ax = plt.subplot(111)

    sns.boxplot(data=df_features_stacked, x='feature', y='value',  hue='artist_name', ax=ax,
                hue_order=['I Belong to the Zoo', 'Big3', 'Dominant Bands'], palette=[COLOR4,COLOR1,COLOR5,"#F5CBA7","#F9E79F","#C39BD3"])


    ax.legend(loc='upper center', bbox_to_anchor=(
    0.5, -0.1), frameon=False, ncol=3)

    ax.set_ylim([-0.0999, 1.2])
    #text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))

    ax.text(0.2, 0.95,'Similar',fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
        horizontalalignment='left',
        verticalalignment='top',
        transform = ax.transAxes)

    ax.text(0.68, 0.95,'Different',fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
        horizontalalignment='left',
        verticalalignment='top',
        transform = ax.transAxes)


    ax.axvspan(2.5, 6.8, facecolor=COLOR6, alpha=0.15)
    ax.set_title("Comparison of Audio Features- I Belong to the Zoo vs Mainstay Artists",fontsize=15)

    return figure

def plot_marketing_strat(artist, figx, figy, streams_df):

    colors=["#f9f8ef","#ddd6a4","#c8bd6c","#a89c3f"]
    streams_df_no_dup=streams_df.copy()
    #streams_df_no_dup["track_name"]=streams_df_no_dup["track_name"].str.strip()
    df_artist = streams_df_no_dup[streams_df_no_dup['artist'] == artist].groupby('track_name')[['streams']]\
    .resample('M').sum()
    df_artist = df_artist.reset_index()
    df_artist['track_name'] = df_artist['track_name'].apply(lambda x: x.split('(')[0])\
        .apply(lambda x: x.split(' - ')[0])
    df_artist["track_name"]=df_artist["track_name"].str.strip()
    
    df_artist=df_artist.groupby(["date","track_name"])[["streams"]].sum().reset_index()
    #print(df_artist)

    #------------------------------------------------------
    arr_df = df_artist.pivot(index='track_name', columns='date', values='streams')
    # divide by 1M to show streams in millions
    arr_df = arr_df/1000000
    arr_df.fillna(0, inplace=True)
    arr_df['total_streams'] = arr_df.sum(axis=1)
    #arr_df = arr_df.sort_values('total_streams',ascending=False)
    arr_df

    #----------------------------------------------------------------------------
    figure = plt.figure(figsize=(figx, figy),dpi=200)
    ax = plt.subplot(111)

    # get all month columns and specify format for xticks
    moncols = arr_df.columns[:-1]
    yymm_cols = pd.Series(moncols.values).apply(lambda x: x.strftime('%Y-%m'))

    sns.heatmap(arr_df[moncols], ax=ax,
                #vmin=0, vmax=0.5,
                #cmap='Greens',
                cmap=sns.color_palette(colors),
                #cmap=ccmap,
                cbar_kws={'label': 'million streams', 'ticks': np.arange(0, 20, 1)},
                xticklabels=yymm_cols, yticklabels=True, linecolor='0.8',linewidths=0.1)

    plt.ylabel('')
    plt.xlabel('')
    return figure

