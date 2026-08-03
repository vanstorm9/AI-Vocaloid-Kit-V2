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

DATASET_SEED = 42

parser = argparse.ArgumentParser(description='Commands for converting a directory of vsqx files to a csv dataset')
parser.add_argument('--vsqxDir', dest="vsqxDir", action="store", default='All-songs/',
                    help='Path to the directory that contains all of the vsqx files to be used in dataset')
parser.add_argument('--seqLen', dest="seqLen", action="store", type=int, default=7,
                    help='The length of each note vector that will iterate through each song')
parser.add_argument('--stride', dest="stride", action="store", type=int, default=2,
                    help='The rate the note vector will iterate through each song.')

args = parser.parse_args()

seqLen = args.seqLen
stride = args.stride
rootDir = args.vsqxDir

assert os.path.exists(rootDir)


def getNoteData(note, key):
    return int(note.getElementsByTagName(key)[0].firstChild.data)


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


def generateNoteData(tokenSeq, seqLen, stride):
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


def generateCSVFile(df, seed=DATASET_SEED, out_dir='dataset/'):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "entireNotes.csv"), index=False)
    rng = np.random.default_rng(seed)
    msk = rng.random(len(df)) < 0.8
    train_df = df[msk]
    test_df = df[~msk]
    train_df.to_csv(os.path.join(out_dir, "trainNotes.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "valNotes.csv"), index=False)


df = None
params = None

for i, fileName in enumerate(os.listdir(rootDir)):
    print(i, ': ', fileName)
    vsqxPath = rootDir + fileName
    path = PurePath(vsqxPath)
    vsqx = xml.dom.minidom.parse(str(path))
    try:
        TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
    except (IndexError, ValueError):
        print('   Failure with ', fileName)
        continue
    mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)
    time = 0

    for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
        mf.addTrackName(trackNo, time, "Track {}".format(str(trackNo)))
        for note in track.getElementsByTagName('note'):
            params = createNote(note, params)

    if params is None:
        print('   Unable to extract notes from ', fileName)
        continue

    prevTime, durTime, timeOffset, noteDict, tokenDict, tokenSeq, i = params
    dfNew = generateNoteData(tokenSeq, seqLen, stride)

    if df is None:
        df = dfNew
    else:
        df = pd.concat([df, dfNew])

generateCSVFile(df)
print('Dataset has been converted')
