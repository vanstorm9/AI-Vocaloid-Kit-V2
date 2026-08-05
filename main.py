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
parser.add_argument('--modelPath', dest="modelPath", action="store", default='savedModels/music-model.pt',
                    help='Path to the trained model')
parser.add_argument('--dupThresh', dest="dupThresh", action="store", type=int, default=3,
                    help='Threshold to balance between note harmony and repeating melodies.')
parser.add_argument('--numOfNotes', dest="numOfNotes", action="store", type=int, default=50,
                    help='Determines the number of notes/midi commands in the generated song')
parser.add_argument('--temperature', dest="temperature", action="store", type=float, default=1.0,
                    help='Sampling temperature (higher = more random, lower = more conservative)')
parser.add_argument('--theme', dest="theme", action="store", default='青春',
                    help='Lyric generation theme (Japanese text, e.g. 青春, 夜, 恋)')
parser.add_argument('--llmModel', dest="llmModel", action="store", default='qwen2.5:7b',
                    help='Ollama model name for lyric generation')
parser.add_argument('--noRepeatNgram', dest="noRepeatNgram", type=int, default=4,
                    help='Block n-grams of this size from repeating during generation (0 to disable)')

args = parser.parse_args()

seedNotePath = args.seed
dupThresh = args.dupThresh
modelPath = args.modelPath
setNum = args.numOfNotes
temperature = args.temperature
theme = args.theme
llmModel = args.llmModel
noRepeatNgram = args.noRepeatNgram

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

from support.vocalVocab import VocaloidVocab, initialize_model, SOS_IDX, EOS_IDX, PAD_IDX

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

vocab = VocaloidVocab()

checkpoint = torch.load(modelPath, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    hp = checkpoint.get('hparams', {})
    model = initialize_model(
        len(vocab), device,
        hid_dim=hp.get('hid_dim', 512),
        n_layers=hp.get('n_layers', 6),
        n_heads=hp.get('n_heads', 8),
        pf_dim=hp.get('pf_dim', 1024),
    )
    model.load_state_dict(checkpoint['model_state'])
else:
    print('Warning: loading legacy checkpoint format')
    model = initialize_model(len(vocab), device)
    model.load_state_dict(checkpoint)

model.eval()
print(f'The model has {songDecoder.count_parameters(model):,} trainable parameters')

CONTEXT_LEN = 64


def _block_ngrams(logits, context, ngram_size):
    """Set logits to -inf for any token that would repeat a seen n-gram."""
    if ngram_size <= 0 or len(context) < ngram_size - 1:
        return logits
    prefix = tuple(context[-(ngram_size - 1):])
    for i in range(len(context) - ngram_size + 1):
        if tuple(context[i:i + ngram_size - 1]) == prefix:
            logits[context[i + ngram_size - 1]] = float('-inf')
    return logits


def generate_notes(model, vocab, seed_tokens, num_notes, dup_thresh, device,
                   temperature=1.0, context_len=CONTEXT_LEN, no_repeat_ngram=4,
                   ticks_per_beat=480):
    """Autoregressively sample note tokens from the decoder-only model.

    Bar-position tokens (B1-B32) are injected automatically when accumulated
    note/rest duration crosses a bar boundary, matching the training sequence format.
    """
    ticks_per_bar = ticks_per_beat * 4  # assume 4/4

    result = list(seed_tokens)
    dup_list = {}

    context = [vocab.encode(t) for t in seed_tokens]
    context = context[-context_len:]

    # Accumulate ticks from seed so we start at the right bar position
    cumulative_ticks = 0
    for tok in seed_tokens:
        if '/' in tok and not tok.startswith('B'):
            try:
                cumulative_ticks += int(tok.split('/')[1][1:])
            except (ValueError, IndexError):
                pass
    current_bar = cumulative_ticks // ticks_per_bar

    for step in range(num_notes):
        # Inject bar token when we cross into a new bar
        new_bar = cumulative_ticks // ticks_per_bar
        if new_bar != current_bar:
            bar_tok = f'B{(new_bar % 32) + 1}'
            bar_idx = vocab.encode(bar_tok)
            result.append(bar_tok)
            context.append(bar_idx)
            current_bar = new_bar

        ctx = torch.tensor([context[-context_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ctx)         # (1, T, vocab)
        logits = logits[0, -1, :] / temperature

        # n-gram blocking: prevent phrase-level loops
        logits = _block_ngrams(logits, context, no_repeat_ngram)

        probs = torch.softmax(logits, dim=-1)
        tok_idx = torch.multinomial(probs, 1).item()
        tok = vocab.decode(tok_idx)

        # duplicate check: re-sample once at higher temperature
        dup_list = addTokensToDup([tok], dup_list, step)
        dup, dup_list = isDuplicateSeq([tok], dup_list, step, dup_thresh)
        if dup:
            logits_hot = logits / (temperature * 1.5)
            probs_hot = torch.softmax(logits_hot, dim=-1)
            tok_idx = torch.multinomial(probs_hot, 1).item()
            tok = vocab.decode(tok_idx)

        if tok in ('<eos>', '<pad>', '<unk>'):
            continue

        result.append(tok)
        context.append(tok_idx)

        # Accumulate duration for bar tracking
        if '/' in tok and not tok.startswith('B'):
            try:
                cumulative_ticks += int(tok.split('/')[1][1:])
            except (ValueError, IndexError):
                pass

    return result


dupList = {}

if len(mainList) <= 0:
    # Start from a random anchor token
    import random as _random
    anchor_tokens = [t for t in vocab.idx2tok if t.startswith('a')]
    mainList = [_random.choice(anchor_tokens)]

mainList = generate_notes(model, vocab, mainList, setNum, dupThresh, device,
                          temperature=temperature, no_repeat_ngram=noRepeatNgram)
print(mainList)


vsqxJson = {u'tracks': 1,
            u'resolution': 480,
            u'stream': [],
            u'format': 1}

mf = MIDIFile(2, removeDuplicates=False)

trackNo = 0
mf.addTrackName(trackNo, tick_time, "Track {}".format(str(trackNo)))

currTime = 5
curr_pitch = 60  # running absolute pitch for interval decoding

for noteTok in mainList:
    try:
        kind, dur_str = noteTok.split('/')
        duration = int(dur_str[1:])
    except ValueError:
        continue

    if kind.startswith('r'):
        if duration > 2000:
            duration = 2000
        currTime += duration
        continue

    if kind.startswith('a'):
        note = int(kind[1:])
        curr_pitch = note
    elif kind.startswith('i'):
        interval = int(kind[1:])  # handles '+N' and '-N'
        curr_pitch = max(0, min(127, curr_pitch + interval))
        note = curr_pitch
    else:
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

from support.lyricGenerator import QwenLyricGenerator, count_morae

lyric_gen = QwenLyricGenerator(model=llmModel, theme=theme)

kanjiTxt = open(outputDir + 'kanji-lyrics.txt', 'w')
hiraTxt = open(outputDir + 'hira-lyrics.txt', 'w')

hiraList = []

for countNum in countList:
    resStr = lyric_gen.generate_phrase(countNum)
    resStr = resStr.replace('\n', '')

    hiraStr = jaconv.kata2hira(convertToHira(resStr, kks))
    hiraTxt.write(hiraStr)
    kanjiTxt.write(resStr)

    tokenizerList = hiraTokenizer(hiraStr)
    hiraList.append(tokenizerList)

    actual_morae = count_morae(resStr)
    print(f'[{resStr}]  ({actual_morae}/{countNum} morae)')

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
        # Lyrics exhausted — wrap around to the beginning
        rowIndexHira = 0
        colIndexHira = 0
        lyricLetter = hiraList[0][0] if hiraList and hiraList[0] else 'あ'

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
