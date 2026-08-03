# -*- coding: utf-8 -*-

# Example command:
# python3 main.py --modelPath savedModels/9-22-music.pt --seed inputs/noteSeed.txt --dupThresh 3 --numOfNotes 50

import numpy as np
import fugashi
import os
import pykakasi
import jaconv
import time as time_module
import pickle
import string

from midiutil import MIDIFile

import xml.dom.minidom
import sys
from pathlib import PurePath

# Imported scripts
import support.json2vsqx as json2vsqx
from support.parsingHelper import *

import torch
import support.songDecoder as songDecoder
import random
import argparse

outputDir = './outputs/'


parser = argparse.ArgumentParser(description='Commands for the vocaloid generator')
parser.add_argument('--seed', dest="seed", action="store", default=None,
                    help='Use beginning notes to initalize melody generation.')
parser.add_argument('--modelPath', dest="modelPath", action="store", default='savedModels/9-22-music.pt',
                    help='Path to the trained model')
parser.add_argument('--dupThresh', dest="dupThresh", action="store", type=int, default=3,
                    help='Threshold to balance between note harmony and repeating melodies.')
parser.add_argument('--numOfNotes', dest="numOfNotes", action="store", type=int, default=50,
                    help='Determines the number of notes/midi commands in the generated song')

args = parser.parse_args()

seedNotePath = args.seed
dupThresh = args.dupThresh
modelPath = args.modelPath
setNum = args.numOfNotes

mainList = []

os.makedirs(outputDir, exist_ok=True)

if seedNotePath is not None:
    if not os.path.exists(seedNotePath):
        print("The path to the file containing initial notes (--seed) does not exist")
        sys.exit(1)

    with open(seedNotePath, "r") as noteSeedHandle:
        for line in noteSeedHandle:
            mainList.append(line.strip())


tick_time = 0


def printNotesTokens(vsqxPath, printNoteLim=-1):
    path = PurePath(vsqxPath)
    vsqx = xml.dom.minidom.parse(str(path))
    TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
    tokList = []
    for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
        for i, note in enumerate(track.getElementsByTagName('note')):
            if i == 0:
                timeOffSet = getNoteData(note, 't') - 5
            if printNoteLim > 0 and i > printNoteLim:
                break
            noteTok = 'n' + str(getNoteData(note, 'n')) + '/d' + str(getNoteData(note, 'dur'))
            print(noteTok, '   note: ', getNoteData(note, 'n'), '   time: ',
                  getNoteData(note, 't') - timeOffSet, '  duration: ', getNoteData(note, 'dur'),
                  '  velocity: ', getNoteData(note, 'v'))
            tokList.append(noteTok)
    return tokList


def createNote(note, params):
    noteDict = {}
    tokenDict = {}
    tokenSeq = []
    tokenStr = ""
    if params is None:
        i = 0
        timeOffset = getNoteData(note, 't') - 5
        prevTime = getNoteData(note, 't') - timeOffset
        durTime = prevTime
    else:
        prevTime, durTime, timeOffset, noteDict, tokenDict, tokenSeq, i = params

    currTime = getNoteData(note, 't') - timeOffset

    if durTime < currTime:
        if 0 in noteDict:
            noteDict[0] += 1
        else:
            noteDict[0] = 1
        tokenStr = "n" + str(0) + "/d" + str(currTime - durTime) + "|"
        tokenSeq.append(tokenStr)
        if tokenStr in tokenDict:
            tokenDict[tokenStr] += 1
        else:
            tokenDict[tokenStr] = 1

    durTime = currTime + getNoteData(note, 'dur')
    tokenStr = "n" + str(getNoteData(note, 'n')) + "/d" + str(getNoteData(note, 'dur')) + "|"
    tokenSeq.append(tokenStr)

    if tokenStr in tokenDict:
        tokenDict[tokenStr] += 1
    else:
        tokenDict[tokenStr] = 1

    if getNoteData(note, 'n') not in noteDict:
        noteDict[getNoteData(note, 'n')] = 1
    else:
        noteDict[getNoteData(note, 'n')] += 1
    i += 1
    return (prevTime, durTime, timeOffset, noteDict, tokenDict, tokenSeq, i)


def generateNoteData(tokenSeq, seqLen=7, stride=2):
    currInd = 0
    dfListCurr = []
    dfListTar = []

    for i in range(0, int(len(tokenSeq) / seqLen)):
        currSeq = tokenSeq[currInd:(currInd + seqLen)]
        currStr = ''.join(currSeq)
        dfListCurr.append(currStr)

        tarInd = currInd + seqLen
        tarSeq = tokenSeq[tarInd:(tarInd + seqLen)]
        tarStr = ''.join(tarSeq)
        dfListTar.append(tarStr)

        currInd += stride

    dfsrc = pd.DataFrame(dfListCurr)
    df2trg = pd.DataFrame(dfListTar)
    df = pd.concat([dfsrc, df2trg], axis=1)
    return df


def generateCSVFile(df):
    df.to_csv("entireNotes.csv", index=False)
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk]
    test_df = df[~msk]
    train_df.to_csv("trainNotes.csv", index=False)
    test_df.to_csv("valNotes.csv", index=False)


"""Now we will start decoding and construct a midi / vsqx file"""

model = songDecoder.initalizeModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(modelPath, map_location=device, weights_only=True))
print(f'The model has {songDecoder.count_parameters(model):,} trainable parameters')

from torchtext.data import Field, BucketIterator
import torchtext

SRC = Field(tokenize=songDecoder.tokenize_notes,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=songDecoder.tokenize_notes,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

data_fields = [('src', SRC), ('trg', TRG)]

train_data, test_data = torchtext.data.TabularDataset.splits(
    path='./', train='dataset/trainNotes.csv', validation='dataset/valNotes.csv',
    format='csv', fields=data_fields)

valid_data = test_data

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

lenOfTokenList = len(TRG.vocab.itos)
dupList = {}

if len(mainList) <= 0:
    mainList = [TRG.vocab.itos[i] for i in np.random.uniform(0, high=lenOfTokenList - 1, size=(7,)).astype(int).tolist()]

enableDuplicate = False

prevSeq = mainList
for i in range(0, setNum):
    translation, attention = translate_sentence(prevSeq, SRC, TRG, model, device)

    token = '|'.join(translation)
    dupList = addTokensToDup(translation, dupList, i)

    if not enableDuplicate:
        dup, dupList = isDuplicateSeq(translation, dupList, i, dupThresh)
        if dup:
            randSeq = np.random.uniform(0, high=lenOfTokenList - 1, size=(7,)).astype(int)
            prevSeq = [TRG.vocab.itos[i] for i in randSeq]
            translation, attention = translate_sentence(prevSeq, SRC, TRG, model, device)
            dupList = addTokensToDup(translation, dupList, i)

    mainList = appendToMainList(mainList, translation)
    prevSeq = translation
    print(prevSeq)


vsqxJson = {u'tracks': 1,
            u'resolution': 480,
            u'stream': [],
            u'format': 1}

mf = MIDIFile(2, removeDuplicates=False)

trackNo = 0
mf.addTrackName(trackNo, tick_time, "Track {}".format(str(trackNo)))

currTime = 5

for noteTok in mainList:
    try:
        note, duration = noteTok.split('/')
    except ValueError:
        continue
    note = int(note[1:])
    duration = int(duration[1:])

    if note == 0:
        if duration > 2000:
            duration = 2000
        currTime += duration
        continue

    duration += 150

    vsqxJson['stream'].append({u'velocity': 64, u'tick': 1, u'sub_type': u'noteOn', u'channel': 1, u'note_num': note})
    vsqxJson['stream'].append({u'velocity': 0, u'tick': duration + 1, u'sub_type': u'noteOff', u'channel': 1, u'note_num': note, u'lyrics': 'み'})

    mf.addNote(trackNo, 0, note, currTime / 480, duration / 480, 64)
    currTime += duration


with open(outputDir + "out.mid", 'wb') as outf:
    mf.writeFile(outf)

vsqxData = json2vsqx.json2vsqx(vsqxJson)
f = open(outputDir + 'output.vsqx', 'wb')
f.write(vsqxData.toprettyxml('', '', 'utf-8'))
f.close()

"""From here, we can generate the lyrics to our song"""

"""We start making the corpus"""

tagger = fugashi.Tagger()
newCorpus = False

lyricDir = './lyric-data/'
corpus = []
savedModelDir = './savedModels/'
begin = time_module.time()

if newCorpus:
    for i, file in enumerate(os.listdir(lyricDir)):
        print(i, ' : ', file)
        txtPath = lyricDir + file
        trump = open(txtPath, encoding='utf8').read()
        trump = lyricProcess(punctPreprocess(trump))
        corpusTmp = [word for word in pykakasiTagDoubleWord(trump, kks, tagger)]
        corpus += corpusTmp

    print(len(corpus))
    print(time_module.time() - begin, 's')
    begin = time_module.time()

    for i, file in enumerate(os.listdir(lyricDir)):
        print(i, ' : ', file)
        txtPath = lyricDir + file
        trump = open(txtPath, encoding='utf8').read()
        trump = lyricProcess(punctPreprocess(trump))
        corpusTmp = [word.surface for word in tagger(trump)]
        corpus += corpusTmp
    pickle.dump(corpus, open(savedModelDir + "corpus.pkl", "wb"))
else:
    corpus = pickle.load(open(savedModelDir + "corpus.pkl", "rb"))

print(len(corpus))
print(time_module.time() - begin, 's')


def make_pairs(corpus):
    for i in range(len(corpus) - 1):
        yield (corpus[i], corpus[i + 1])


pairs = make_pairs(corpus)

word_dict = {}

for word_1, word_2 in pairs:
    if word_1 in word_dict:
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

vsqxPath = outputDir + 'output.vsqx'

assert os.path.exists(vsqxPath)

path = PurePath(vsqxPath)
vsqx = xml.dom.minidom.parse(str(path))

TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])

mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)

midi_time = 0

for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
    mf.addTrackName(trackNo, midi_time, "Track {}".format(str(trackNo)))

    for i, note in enumerate(track.getElementsByTagName('noteNum')):
        mf.addNote(trackNo, 0, getNoteData(note, 'n', i, track), getNoteData(note, 't', i, track) / 480,
                   getNoteData(note, 'dur', i, track) / 480, 64)
    mf.addTempo(trackNo, midi_time, TEMPO)

with open(outputDir + "out.mid", 'wb') as outf:
    mf.writeFile(outf)


params, noteClusterList = getNoteGroupCluster(vsqxPath)
prevTime, durTime, timeOffset, noteDict, tokenDict, tokenSeq, i = params

noteNewClusterList = divideListCluster(noteClusterList)
countList = combineSmallNoteClusterCount(noteNewClusterList)

kanjiTxt = open(outputDir + 'kanji-lyrics.txt', 'w')
hiraTxt = open(outputDir + 'hira-lyrics.txt', 'w')

hiraList = []

lengthWord = 3
initalWord = None
for countNum in countList:
    if countNum < lengthWord:
        resStr, sumNum = generateLyric(countNum, countNum, None, kks)
    else:
        resStr, sumNum = generateLyric(countNum, lengthWord, initalWord, kks)

    initalWord = str(getLastWord(resStr, tagger))
    resStr = resStr.replace('\n', '')

    hiraTxt.write(convertToHira(resStr, kks))
    kanjiTxt.write(resStr)

    hiraStr = jaconv.kata2hira(convertToHira(resStr, kks))
    tokenizerList = hiraTokenizer(hiraStr)
    hiraList.append(tokenizerList)

    print('[', resStr, ']')

hiraTxt.close()
kanjiTxt.close()


"""We try to generate the song again, but with the lyrics"""

vsqxJson = {u'tracks': 1,
            u'resolution': 480,
            u'stream': [],
            u'format': 1}

path = PurePath(vsqxPath)
vsqx = xml.dom.minidom.parse(str(path))

TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
mf = MIDIFile(2, removeDuplicates=False)

trackNo = 1
timeInt = 0
mf.addTrackName(trackNo, timeInt, "Track {}".format(str(trackNo)))

currTime = 5

rowIndexHira = 0
colIndexHira = 0

for i, noteTok in enumerate(tokenSeq):
    noteTok = noteTok.replace('|', '')
    try:
        note, duration = noteTok.split('/')
    except ValueError:
        continue

    note = int(note[1:])
    duration = int(duration[1:])

    if note == 0:
        if duration > 2000:
            duration = 2000
        currTime += duration
        if len(hiraList[rowIndexHira]) - 1 > colIndexHira:
            colIndexHira += 1
        else:
            rowIndexHira += 1
            colIndexHira = 0
        continue

    duration += 50

    vsqxJson['stream'].append({u'velocity': 64, u'tick': 1, u'sub_type': u'noteOn', u'channel': 0, u'note_num': note})

    try:
        lyricLetter = hiraList[rowIndexHira][colIndexHira]
    except IndexError:
        rowIndexHira += 1
        colIndexHira = 0
        lyricLetter = hiraList[rowIndexHira][colIndexHira]

    vsqxJson['stream'].append({u'velocity': 0, u'tick': duration + 1, u'sub_type': u'noteOff',
                                u'channel': 0, u'note_num': note, u'lyrics': lyricLetter})
    colIndexHira += 1

    mf.addNote(trackNo, 0, note, currTime / 480, duration / 480, 64)
    currTime += duration


with open(outputDir + "out.mid", 'wb') as outf:
    mf.writeFile(outf)

vsqxData = json2vsqx.json2vsqx(vsqxJson)
f = open(outputDir + 'output.vsqx', 'wb')
f.write(vsqxData.toprettyxml('', '', 'utf-8'))
f.close()
