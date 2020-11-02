# -*- coding: utf-8 -*-
# python3 datasetConvert.py --vsqxDir All-song/ --seqLen 7 --stride 2

import pykakasi
from midiutil import MIDIFile

import xml.dom.minidom
import sys
from pathlib import PurePath
import os

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Commands for converting a directory of vsqx files to a csv dataset')
parser.add_argument('--vsqxDir', dest="vsqxDir",action="store",default='All-songs/',
                   help='Path to the dirctory that contains all of the vsqx files to be used in dataset')
parser.add_argument('--seqLen', dest="seqLen",action="store",type=int,default=7,
                   help='The length of each note vector that will iterate through each song')
parser.add_argument('--stride', dest="stride",action="store",type=int,default=2,
                   help='The rate the note vector will iterate through each song.')

args = parser.parse_args()



seqLen = args.seqLen
stride = args.stride
rootDir = args.vsqxDir

assert os.path.exists(rootDir)


def getNoteData(note, key):
	return int(note.getElementsByTagName(key)[0].firstChild.data)


def printNotesTokens(vsqxPath,printNoteLim=-1):
  path = PurePath(vsqxPath)

  vsqx = xml.dom.minidom.parse(str(path))

  TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])

  #mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)

  time = 0
  beginInt = 5
  tokList = []
  for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
    for i,note in enumerate(track.getElementsByTagName('note')):
      if i == 0:
        timeOffSet = getNoteData(note,'t')-beginInt

      if printNoteLim > 0 and i > printNoteLim:
        break
        
      noteTok =  'n'+str(getNoteData(note,'n'))+'/d'+str(getNoteData(note, 'dur'))
      print(noteTok,'   note: ',getNoteData(note,'n'),'   time: ',getNoteData(note,'t')-timeOffSet,'  duration: ', getNoteData(note, 'dur'), '  velocity: ',getNoteData(note, 'v'))
      tokList.append(noteTok)
  return tokList

def createNote(note,params):
  
  noteDict = {}
  tokenDict = {}

  tokenSeq = [] # For keeping track of order of tokens

  tokenStr = ""
  if params is None:
    # First note step
    i = 0
    timeOffset = getNoteData(note,'t') - 5
    prevTime = getNoteData(note,'t')-timeOffset
    durTime = prevTime
  else:
    prevTime,durTime,timeOffset,noteDict,tokenDict,tokenSeq,i = params
  

  currTime = getNoteData(note,'t')-timeOffset

  if durTime < currTime:
      # This means there is a gap between ending duration of note and new note
      # So we add a value of 0
      #print('note: ',0,'   time: ',durTime,'  duration: ', currTime-durTime, '  velocity: ',0)


    

      if 0 in noteDict:
        noteDict[0] += 1
      else:
        noteDict[0] = 1

      tokenStr = "n"+str(0)+"/d"+str(currTime-durTime)+"|"
      tokenSeq.append(tokenStr)
      if tokenStr in tokenDict:
        tokenDict[tokenStr] += 1
      else:
        tokenDict[tokenStr] = 1


      #mf.addNote(trackNo, 0, 0, durTime / 480, (currTime-durTime) / 480, 0)

    
  #print('note: ',getNoteData(note,'n'),'   time: ',currTime,'  duration: ', getNoteData(note, 'dur'), '  velocity: ',getNoteData(note, 'v'))
    
  durTime = currTime + getNoteData(note, 'dur')


  tokenStr = "n"+str(getNoteData(note,'n'))+"/d"+str(getNoteData(note, 'dur'))+"|"
  tokenSeq.append(tokenStr)

  if tokenStr in tokenDict:
    tokenDict[tokenStr] += 1
  else:
    tokenDict[tokenStr] = 1
  # Count frequency
  if getNoteData(note,'n') not in noteDict:
    noteDict[getNoteData(note,'n')] = 1
  else:
    noteDict[getNoteData(note,'n')] += 1
  i+= 1
  return (prevTime,durTime,timeOffset,noteDict,tokenDict,tokenSeq,i)

def generateNoteData(tokenSeq):
  seqLen = 7
  stride = 2


  currInd = 0
  dfListCurr = []
  dfListTar = []

  for i in range(0,int(len(tokenSeq)/7)):
    currSeq = tokenSeq[currInd:(currInd+seqLen)]
    currStr = ''.join(currSeq)
    dfListCurr.append(currStr)

    tarInd = currInd+seqLen
    tarSeq = tokenSeq[tarInd:(tarInd+seqLen)]
    tarStr = ''.join(tarSeq)
    dfListTar.append(tarStr)

    currInd += stride
    #print(currStr)
    #print(tarStr)
    #print('-------------')


  dfsrc = pd.DataFrame(dfListCurr)
  df2trg = pd.DataFrame(dfListTar)

  frames = [dfsrc, df2trg]
  df = pd.concat(frames,axis=1)
  #print(df)

  return df


def generateCSVFile(df):
  df.to_csv("entireNotes.csv", index=False)
  #print(df)

  msk = np.random.rand(len(df)) < 0.8
  train_df = df[msk]
  test_df = df[~msk]
  #print(test_df)

  train_df.to_csv("trainNotes.csv", index=False)
  test_df.to_csv("valNotes.csv", index=False)





############






# Encode info into vector
df = None
params = None

for i,fileName in enumerate(os.listdir(rootDir)):
  # This is for implementing all songs
  print(i,': ', fileName)
  vsqxPath = rootDir + fileName 
  path = PurePath(vsqxPath)
  vsqx = xml.dom.minidom.parse(str(path))
  try:
    TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
  except:
    print('   Failure with ', fileName)
    continue
  mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)
  time = 0




  for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
    mf.addTrackName(trackNo, time, "Track {}".format(str(trackNo)))
    i = 0
    timeOffset = 0

    prevTime = 0
    durTime = 0

    #params = None
    #print(len(track.getElementsByTagName('note')))
    for note in track.getElementsByTagName('note'):
      params = createNote(note,params)
      #mf.addNote(trackNo, 0, getNoteData(note, 'n'), currTime / 480, getNoteData(note, 'dur') / 480, 64)
      
    #mf.addTempo(trackNo, time, TEMPO)

  if params is None:
    print('   Unable to extract notes from ', fileName)
    continue


  prevTime,durTime,timeOffset,noteDict,tokenDict,tokenSeq,i = params
  # We construct pandas dataframe
  dfNew = generateNoteData(tokenSeq)

  if df is None:
    df = dfNew
  else:
    frames = [df, dfNew]
    df = pd.concat(frames)
  #print(df)

  # n-63|d-480


#generateCSVFile(df)


"""Here, we generate our note data csv file with a note sequence and their following note sequence being put into csv file"""


currInd = 0
dfListCurr = []
dfListTar = []

for i in range(0,int(len(tokenSeq)/7)):
  currSeq = tokenSeq[currInd:(currInd+seqLen)]
  currStr = ''.join(currSeq)
  dfListCurr.append(currStr)

  tarInd = currInd+seqLen
  tarSeq = tokenSeq[tarInd:(tarInd+seqLen)]
  tarStr = ''.join(tarSeq)
  dfListTar.append(tarStr)

  currInd += stride


dfsrc = pd.DataFrame(dfListCurr)
df2trg = pd.DataFrame(dfListTar)

frames = [dfsrc, df2trg]
df = pd.concat(frames,axis=1)

df.to_csv("entireNotes.csv", index=False)
#print(df)

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]
#print(test_df)

train_df.to_csv("trainNotes.csv", index=False)
test_df.to_csv("valNotes.csv", index=False)

print('Dataset has been converted')






