# -*- coding: utf-8 -*-

# Example command:
# python3 main.py --seed inputs/noteSeed.txt

import numpy as np
import fugashi
import os
import pykakasi
import jaconv
from googletrans import Translator
import time
import pickle
import string


from midiutil import MIDIFile

import xml.dom.minidom
import sys
from pathlib import PurePath
import numpy as np

# Imported scripts
import support.json2vsqx as json2vsqx
from support.parsingHelper import *

import torch
import support.songDecoder as songDecoder
import random
import argparse

outputDir = './outputs/'




parser = argparse.ArgumentParser(description='Commands for the vocaloid generator')
parser.add_argument('--seed', dest="seed",action="store",default=None,
                   help='Use beginning notes to initalize melody generation')

args = parser.parse_args()

seedNotePath = args.seed




# We feed in a text file that contains starter notes

mainList = []

if seedNotePath is not None:

  try:
    assert os.path.exists(seedNotePath)
  except:
    print("The path to the file contianing inital notes, represented as arguument --seed, does not exist")
    raise

  with open(seedNotePath,"r") as noteSeedHandle:
    for line in noteSeedHandle:
      mainList.append(line)



time = 0
#mainList = ['n63/d150', 'n65/d30', 'n65/d90', 'n65/d30', 'n65/d30', 'n64/d300', 'n64/d150', 'n64/d60']


"""Now we proceed to generate lyrics"""

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


"""Now we will start decoding and construct a midi / vsqx file"""


modelPath = savedModelDir+'9-22-music.pt'

model = songDecoder.initalizeModel()
#model.load_state_dict(torch.load('melodyModel-9-20-20.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(modelPath,map_location=device))
print(f'The model has {songDecoder.count_parameters(model):,} trainable parameters')

from torchtext.data import Field, BucketIterator
import torchtext

SRC = Field(tokenize = songDecoder.tokenize_notes, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

TRG = Field(tokenize = songDecoder.tokenize_notes, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

data_fields = [('src', SRC), ('trg', TRG)]

train_data, test_data = torchtext.data.TabularDataset.splits(path='./', train='colab/pipeline/trainNotes.csv', validation='colab/pipeline/valNotes.csv', format='csv', fields=data_fields)

valid_data = test_data

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


#vsqxData = json2vsqx(vsqxJson)

# We start to construct the midi file
setNum = 200
#setNum = 50

enableDuplicate = False
#enableDuplicate = True

lenOfTokenList = len(TRG.vocab.itos)
dupList = {}

if len(mainList) <= 0:
  mainList = [TRG.vocab.itos[i] for i in np.random.uniform(0, high=lenOfTokenList-1, size=(7,)).astype(int).tolist()]


prevSeq = mainList
for i in range(0,setNum):

  translation, attention = translate_sentence(prevSeq, SRC, TRG, model, device)

  token = '|'.join(translation)
  dupList = addTokensToDup(translation,dupList,i)  # For individual tokens
  
  if not enableDuplicate:
    
    #if isDuplicateSeq(token ,dupList,i):
    #if isDuplicate(translation,dupList,i):

    dup,dupList = isDuplicateSeq(translation ,dupList,i,3)
    if dup:
      # We generate a random token to predict on
      randSeq = np.random.uniform(0, high=lenOfTokenList-1, size=(7,)).astype(int)
      prevSeq = [TRG.vocab.itos[i] for i in randSeq]
      translation, attention = translate_sentence(prevSeq, SRC, TRG, model, device)
      
      
      tokenNew = '|'.join(translation)
      #dupList[tokenNew] = i  # For sequence tokens
      
      dupList = addTokensToDup(translation,dupList,i) # For individual tokens
    
  
  mainList = appendToMainList(mainList,translation)

  # For sequence tokens
  #dupList[token] = i 

  prevSeq = translation
  print(prevSeq)

  #print(f'predicted trg = {translation}')
#print(mainList)

vsqxJson = {u'tracks': 1, 
            u'resolution': 480, 
            u'stream': [],
            u'format': 1}



mf = MIDIFile(2, removeDuplicates=False)

trackNo = 0  # Added
mf.addTrackName(trackNo, time, "Track {}".format(str(trackNo)))

currTime = 5

for noteTok in mainList:
  try:
    note,duration = noteTok.split('/')
  except:
    continue
  note = int(note[1:])
  duration = int(duration[1:])

  if note == 0:
    if duration > 2000:
      duration = 2000
    currTime += duration
    continue
  
  duration += 150
  
  #print(duration)

  vsqxJson['stream'].append({u'velocity': 64, u'tick': 1 , u'sub_type': u'noteOn', u'channel': 1, u'note_num': note})
  vsqxJson['stream'].append({u'velocity': 0, u'tick': duration+1, u'sub_type': u'noteOff', u'channel': 1, u'note_num': note, u'lyrics': 'み'})

  mf.addNote(trackNo, 0, note, currTime / 480, duration / 480, 64)
  
  currTime += duration


with open(outputDir+"out.mid", 'wb') as outf:
	mf.writeFile(outf)
 
# We write the vsqx file
vsqxData = json2vsqx.json2vsqx(vsqxJson)
f = open(outputDir +'output.vsqx', 'wb')
f.write(vsqxData.toprettyxml('', '', 'utf-8'))
f.close()

"""From here, we can generate the lyrics to our song"""


"""We start making the corpus"""

import time

tagger = fugashi.Tagger()

# Trump's speeches here: https://github.com/ryanmcdermott/trump-speeches
#trump = open('speeches.txt', encoding='utf8').read()
newCorpus = False

lyricDir = './lyric-data/'
corpus = []
#lastWordDict = {}
#firstWordDict = {}
begin = time.time()

if newCorpus:
  ##################
  # To retrieve the double pairs in front and end of sentence
  for i,file in enumerate(os.listdir(lyricDir)):
    print(i,' : ', file)
    txtPath = lyricDir + file
    #print(txtPath)
    trump = open(txtPath, encoding='utf8').read()
    trump  = lyricProcess(punctPreprocess(trump))

    corpusTmp = [word for word in pykakasiTagDoubleWord(trump,kks,tagger)]
    corpus += corpusTmp 

  print(len(corpus))
  print(time.time()-begin,'s')
  begin = time.time()


  ###################


  for i,file in enumerate(os.listdir(lyricDir)):
    print(i,' : ', file)
    txtPath = lyricDir + file
    #print(txtPath)
    trump = open(txtPath, encoding='utf8').read()
    trump  = lyricProcess(punctPreprocess(trump))

    #lastWordDict = lastWordFromCorpus(trump,lastWordDict,kks,tagger)
    #firstWordDict = firstWordFromCorpus(trump,firstWordDict,kks,tagger)

    #corpus = trump.split()
    corpusTmp = [word.surface for word in tagger(trump)]
    #print(len(corpus))
    corpus += corpusTmp 
  pickle.dump(corpus, open(savedModelDir+"corpus.pkl", "wb" ))
else:
  corpus = pickle.load(open(savedModelDir+"corpus.pkl", "rb" ))

print(len(corpus))
print(time.time()-begin,'s')

def make_pairs(corpus):
    for i in range(len(corpus)-1):
        yield (corpus[i], corpus[i+1])
        
pairs = make_pairs(corpus)

word_dict = {}

for word_1, word_2 in pairs:
    if word_1 in word_dict.keys():
        if word_2 in word_dict[word_1]:
          word_dict[word_1][word_2] += 1
        else:
          word_dict[word_1][word_2] = 1
    else:
        word_dict[word_1] = {word_2: 1}

first_word = str(np.random.choice(corpus))
while first_word.islower():
    first_word = np.random.choice(corpus)
chain = [first_word]
n_words = 50


word = chain[-1]


'''
# This is to convert VSQX to midi
from midiutil import MIDIFile

import xml.dom.minidom
import sys
from pathlib import PurePath

#####
import os

####
import pandas as pd
import numpy as np
'''


	

#vsqxPath = '/content/Crossing-Fields.vsqx'
#vsqxPath = '/content/AiDee-simplified.vsqx'
#vsqxPath = '/content/seed-crossingField-0.vsqx'
#vsqxPath = '/content/random-0.vsqx'
vsqxPath = outputDir +'output.vsqx'

assert os.path.exists(vsqxPath)

path = PurePath(vsqxPath)

vsqx = xml.dom.minidom.parse(str(path))

TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])

mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)

time = 0

for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
	mf.addTrackName(trackNo, time, "Track {}".format(str(trackNo)))
	
	for i,note in enumerate(track.getElementsByTagName('noteNum')):
		#mf.addNote(trackNo, 0, getNoteData(note, 'n'), getNoteData(note, 't') / 480, getNoteData(note, 'dur') / 480, getNoteData(note, 'v'))
		#print('note: ',note)
		mf.addNote(trackNo, 0, getNoteData(note, 'n',i,track), getNoteData(note, 't',i,track) / 480, getNoteData(note, 'dur',i,track) / 480, 64)
		#print('note: ',getNoteData(note,'n'),'   time: ',getNoteData(note,'t'),'  duration: ', getNoteData(note, 'dur'), '  velocity: ',getNoteData(note, 'v'))
	mf.addTempo(trackNo, time, TEMPO)

#with open(str(path.parents[0]) +'\\'+ path.stem + ".mid", 'wb') as outf:
with open(outputDir +"out.mid", 'wb') as outf:
	mf.writeFile(outf)




params,noteClusterList = getNoteGroupCluster(vsqxPath)
prevTime,durTime,timeOffset,noteDict,tokenDict,tokenSeq,i = params


#print(noteClusterList[:10])
#print(noteNewClusterList[:10])

noteNewClusterList = divideListCluster(noteClusterList)
countList = combineSmallNoteClusterCount(noteNewClusterList)

kanjiTxt = open(outputDir +'kanji-lyrics.txt','w')
hiraTxt = open(outputDir +'hira-lyrics.txt','w')

hiraList = []

lengthWord = 3
initalWord = None
for countNum in countList:
  #print(countNum)
  if countNum < lengthWord:
    resStr,sumNum = generateLyric(countNum,countNum,None,kks)
  else:
    resStr,sumNum  = generateLyric(countNum,lengthWord,initalWord,kks)
  
  initalWord = str(getLastWord(resStr,tagger))
  resStr = resStr.replace('\n','')

  hiraTxt.write(convertToHira(resStr, kks))
  kanjiTxt.write(resStr)

  # Now we add the resulting lyrics to our hiragana lyric List
  hiraStr = jaconv.kata2hira(convertToHira(resStr, kks))
  tokenizerList = hiraTokenizer(hiraStr)
  hiraList.append(tokenizerList)

  #print('[',resStr,' [',countNum,':',sumNum,']')
  print('[',resStr,']')

hiraTxt.close()
kanjiTxt.close()


"""We try to generate the song again, but with the lyrics"""


vsqxJson = {u'tracks': 1, 
            u'resolution': 480, 
            u'stream': [],
            u'format': 1}

path = PurePath(vsqxPath)
vsqx = xml.dom.minidom.parse(str(path))

# From our constructed list, we create a midi file
TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
mf = MIDIFile(2, removeDuplicates=False)

trackNo = 1  # Added
timeInt = 0
mf.addTrackName(trackNo, timeInt, "Track {}".format(str(trackNo)))

currTime = 5

rowIndexHira = 0
colIndexHira = 0

tokenSeq
#for i,noteTok in enumerate(mainList):
for i,noteTok in enumerate(tokenSeq):
  noteTok = noteTok.replace('|','')
  #print(noteTok)
  try:
    note,duration = noteTok.split('/')
  except:
    continue
    
  note = int(note[1:])
  duration = int(duration[1:])

  

  if note == 0:
    if duration > 2000:
      duration = 2000
    currTime += duration
    continue
  
  duration += 50
  
  #print(duration)


  # We add note and generated hiragana letter to our vsqx file

  vsqxJson['stream'].append({u'velocity': 64, u'tick': 1 , u'sub_type': u'noteOn', u'channel': 0, u'note_num': note})
  #print(noteTok)
  if note == 0:
    # We do not insert lyrics and increment row
    vsqxJson['stream'].append({u'velocity': 0, u'tick': duration+1, u'sub_type': u'noteOff', u'channel': 0, u'note_num': note})
    
    if len(hiraList[rowIndexHira])-1 > colIndexHira:
      colIndexHira += 1
    else:
      rowIndexHira += 1
      colIndexHira = 0
  else:
    #print('row: ', rowIndexHira, '   col: ', colIndexHira)
    #print(colIndexHira, '   ', hiraList[rowIndexHira])
    try:
      #print('     ',hiraList[rowIndexHira][colIndexHira])
      lyricLetter = hiraList[rowIndexHira][colIndexHira]
    except:
      rowIndexHira += 1
      colIndexHira = 0
      lyricLetter = hiraList[rowIndexHira][colIndexHira]

    vsqxJson['stream'].append({u'velocity': 0, u'tick': duration+1, u'sub_type': u'noteOff', u'channel': 0, u'note_num': note, u'lyrics': lyricLetter})
    colIndexHira += 1

  mf.addNote(trackNo, 0, note, currTime / 480, duration / 480, 64)
  
  currTime += duration


with open(outputDir +"out.mid", 'wb') as outf:
	mf.writeFile(outf)
 


# We write the vsqx file
vsqxData = json2vsqx.json2vsqx(vsqxJson)
f = open(outputDir +'output.vsqx', 'wb')
f.write(vsqxData.toprettyxml('', '', 'utf-8'))
f.close()

