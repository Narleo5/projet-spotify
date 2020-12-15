#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:30:57 2020

@author: Nadir
"""

import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

cid ="d6440636c8de42f9a31db8c168ecfd01" 
secret = "ad9b78a3088b4acbbd61540f90405692"

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#scrapping TOP 100 


def clean(data):
    df = data.copy()
    df['Rang']=data.index
    df.reset_index(inplace=True)
    df.drop(['index'], axis = 1, inplace = True)
    df[['Artiste', 'Titre']] = df[['Artiste', 'Titre']].apply(lambda x: x.str.lower().str.replace(',','').str.replace('feat.', '').str.replace('featuring','').str.replace('&','').str.replace('.',' ').str.strip())
    return(df)



def get_top_fr(year, size = 100):
    page = requests.get('https://snepmusique.com/les-tops/le-top-de-lannee/top-singles-annee/?annee='+str(year))
    soup = BeautifulSoup(page.content, 'html.parser')
    artistes = []
    titres = []
    
    n = min(size, len(soup.find_all('div', class_='artiste')))
    
    for i in range(n):
        artiste = soup.find_all('div', class_='artiste')[i].get_text()
        titre = soup.find_all('div', class_='titre')[i].get_text()
        
        artistes.append(artiste)
        titres.append(titre)
        
    df = pd.DataFrame({'Artiste' : artistes, 'Titre' : titres, 'Année' : [year]*n}, index=range(1,n+1))
    
    return(df)   


def get_tops_fr(year_inf=2000, year_sup=2020, size = 100):
    if year_inf == year_sup:
        return(get_top_fr(year_inf, size))
    else: 
        return(pd.concat([get_tops_fr(year_inf,year_sup -1,size), get_top_fr(year_sup,size)]))
    
def get_top_us(year, size = 100):
    page = requests.get('https://www.billboard.com/charts/year-end/'+str(year)+'/hot-100-songs')
    soup = BeautifulSoup(page.content, 'html.parser')
    artistes = []
    titres = []
    
    n = min(size, len(soup.find_all('div', class_= "ye-chart-item__title" )))
    
    for i in range(n):
        artiste = soup.find_all('div', class_="ye-chart-item__artist")[i].get_text()[1:-1]
        titre = soup.find_all('div', class_="ye-chart-item__title")[i].get_text()[1:-1]
        
        artistes.append(artiste)
        titres.append(titre)
        
    df = pd.DataFrame({'Artiste' : artistes, 'Titre' : titres, 'Année' : [year]*n}, index=range(1,n+1))
    
    return(df)


def get_tops_us(year_inf=2006, year_sup=2019, size = 100):
    if year_inf == year_sup:
        return(get_top_us(year_inf, size))
    else: 
        return(pd.concat([get_tops_us(year_inf,year_sup -1,size), get_top_us(year_sup,size)]))


# Spotify track IDs 


def search_id_track(df, market):
    
    d = {}
    dff = df.copy()

    for x in list(dff.index):

        artiste = dff['Artiste'][x]
        titre = dff['Titre'][x]
        year = dff['Année'][x]
        
        track_results = sp.search(q=f'track:{titre}', type='track', market = str(market))
    
        for i in range(min(5,len(track_results['tracks']['items']))):
            if x in d.keys():
                break
            for word in artiste.split():
                if word in track_results['tracks']['items'][i]['artists'][0]['name'].lower():
                    d[x] = track_results['tracks']['items'][i]['id'], track_results['tracks']['items'][i]['popularity']
                    break
                 
    dff = df.drop(d.keys(),axis=0)

    
    for x in list(dff.index):
        
    
        artiste = dff['Artiste'][x]
        titre = dff['Titre'][x]
        year = dff['Année'][x]
        
        track_results = sp.search(q=f'artist:{artiste}' , type='track', market = str(market), limit = 30)
        
        for i in range(min(30,len(track_results['tracks']['items']))):
            
            if x in d.keys():
                break
            
            else:
                
                n = min(3, len(titre.split()))
                titre_reduce = ' '.join(titre.split()[:n])
                if titre_reduce in track_results['tracks']['items'][i]['name'].lower() or track_results['tracks']['items'][i]['name'].lower() in titre_reduce:
                    d[x] = track_results['tracks']['items'][i]['id'], track_results['tracks']['items'][i]['popularity']
    
    dff = df.drop(d.keys(),axis=0)
                    
    print(f"pourcentage de données manquantes : {100*dff.shape[0]/df.shape[0]} %")
    
    ids = pd.DataFrame(d.values(), index = d.keys(), columns =['id', 'popularity'])
    
    df_id = df.drop(list(dff.index),axis=0)
    
    return(pd.concat([df_id,ids],axis=1))


#Spotify features with IDs 

def get_features(df):
    rows = []
    batchsize = 100
    None_counter = 0
    for i in range(0,df.shape[0],batchsize):
        batch = df['id'][i:i+batchsize]
        feature_results = sp.audio_features(batch)
        
        for i, t in enumerate(feature_results):
            if t == None:
                None_counter = None_counter + 1
            else:
                rows.append(t)
    print(None_counter)
    features = pd.DataFrame.from_dict(rows,orient='columns')
    columns_to_drop = ['analysis_url','track_href','type','uri']
    features.drop(columns_to_drop, axis=1,inplace=True)
    musics = pd.merge(df,features,on='id')
    
    return(musics.drop_duplicates(keep='first'))


#Create DF with playlists 

def create_df_playlist(dico):
    categorie = []
    artiste = []
    ids = []
    titre = []
    for j in dico.keys():
        for idd in dico[j]:
            playlist = sp.user_playlist(cid, idd, 'tracks')
            data = playlist['tracks']
            for track in data['items']:
                ids.append(track['track']['id'])
                titre.append(track['track']['name'])
                b=''
                for artist in track['track']['artists']:
                    b+=str(' '+artist['name'])
                artiste.append(b)
                categorie.append(j)
    df = pd.DataFrame(list(zip(artiste,titre,categorie,ids)),columns = ['Artiste', 'Titre', 'Catégorie', 'id'])
    return(get_features(df)) 

# Chargement des playlist

def playlist_top100fr():
    df = get_tops_fr() 
    df = clean(df)
    dff = search_id_track(df, market='FR')
    return(get_features(dff))



def playlist_top100us():
    df = get_tops_us() 
    df = clean(df)
    dff = search_id_track(df, market='US')
    return(get_features(dff))



rnbrock = {'rnb' : ['1wpgZuFXRTykFnMglhPuz9','78tPw9siuiq0XkhByNnYdR','1rLuENR0J3kQ2DNXJ0bp6x','7jC31nUrv3LWjulJfEFti8','28AOnXdRpkeBGMH3zMehz1','4oFXq56HSbMUrTT8PELyWO','37i9dQZF1DX04mASjTsvf0'], 
           'rock' : ['37i9dQZF1DWXRqgorJj26U','37i9dQZF1DWXTHBOfJ8aI7','4jOqGKvV7iu0ojea2pt9Te','37i9dQZF1DWWSuZL7uNdVA','3Ho3iO0iJykgEQNbjB2sic','37i9dQZF1DX2aneNMeYHQ8','37i9dQZF1DX68H8ZujdnN7']}

def playlist_rnbrock():
    return(create_df_playlist(rnbrock))


dico = {'funk' : ['37i9dQZF1DWUS3jbm4YExP', '37i9dQZF1DX4WgZiuR77Ef','37i9dQZF1DWWvhKV4FBciw'],
        'jazz' : ['37i9dQZF1DXbITWG1ZJKYt', '37i9dQZF1DX4wta20PHgwo','37i9dQZF1DX2vYju3i0lNX','37i9dQZF1DXdziGPHNE40t'], 
        'rock' : ['37i9dQZF1DWXRqgorJj26U', '37i9dQZF1DX2xKCIjknUoQ','37i9dQZF1DWXTHBOfJ8aI7','37i9dQZF1DWWSuZL7uNdVA'], 
        'classique' : ['7IFEOepzcIUmsD5aUTivyX', '62MgIbD9qUykkaD8dbRNK4','7wNzpJ4xwH5GVZ4ONYQZnr','7uWLdgDnH26OE02He6G2qF'],
        'rap_fr' : ['37i9dQZF1DWSrqNVMcxGKc', '6cZmC8TNyvDc7FQR9JIaH0','4dx099NEMVYP8l2YFq8xTL'],
        'rnb' : ['37i9dQZF1DX9UuQbl12Nmb', '37i9dQZF1DX6VDO8a6cQME','1wpgZuFXRTykFnMglhPuz9','78tPw9siuiq0XkhByNnYdR']
}

def playlist_6clusters():
    return(create_df_playlist(dico))

