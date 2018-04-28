import codecs
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from io import StringIO
from sklearn.cluster import KMeans as km
from sklearn.cluster import AffinityPropagation as ap
from sklearn.cluster import Birch as b
from sklearn.cluster import FeatureAgglomeration as fa

def arrayToDict(array):
	tempDict = {}
	for i in range(0, len(array), 2):
		tempDict[array[i]]=array[i+1]
	return tempDict

def getFile(fileName):
	file = codecs.open(fileName, 'r', encoding='utf8', errors='ignore')#gets file
	file = file.read()#reads file into 1 string
	file = file.split()#makes it into an array
	return file

def addEmojiCode(dimensions, cluster, emojiDict):
    labels = cluster.predict(dimensions)
    clusters = {}
    n = 0
    for item in labels:#goes in order by coords put in which go in order by emoji code so will work
	    if item in clusters:
	        clusters[item].append(emojiDict[n])#if has two in same spot(doubt it does but just in case)(looking back this is quite useless)
	    else:
	        clusters[item] = [emojiDict[n]]#for each new set of coords get the emoji code for it and ad to dictionary
	    n += 2#skips icons
    return clusters

def organizeEnumeratedDictionary(dictionary):
	newDict = {}
	for i in range(0, len(dictionary)):
		newDict[i]=dictionary[i]
	return newDict

def emojiCodeToEmoji(clusterDict, emojiDict):
	for i in clusterDict:
		for x in range(0, len(clusterDict[i])):
			clusterDict[i][x] = emojiDict[clusterDict[i][x]]
	return clusterDict


tsvToEmojiDict = arrayToDict(getFile('emoji_lookup.tsv'))#dictionary to translate emojis over
emojiDataFrame = pd.read_csv(StringIO(codecs.open('emojis.txt', 'r', encoding='utf8', errors='ignore').read()), sep='\s+')#creates pandas system for holding data
dimensions = np.array(emojiDataFrame.as_matrix(columns=emojiDataFrame.columns[1:]))#gets dimensions in an array perfect for cluster
#			auto random state
#cluster = km(n_clusters = 250, max_iter=10000000)#kmeans clustering
cluster = ap(max_iter=1000000)#Affinity propogation clustering
#cluster = b(n_clusters=200)#birch clustering
cluster.fit(dimensions)
codedCluster = addEmojiCode(dimensions, cluster, getFile('emoji_lookup.tsv'))
organizedClusters = organizeEnumeratedDictionary(codedCluster)
emojiClusters = emojiCodeToEmoji(organizedClusters, tsvToEmojiDict)
for z in emojiClusters:
	print("Clusters" + str(z))
	print(emojiClusters[z])
	print("\n")
