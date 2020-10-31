import fugashi
import os
import pykakasi
from googletrans import Translator
import torch

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
import pickle
import string



savedModelDir = './savedModels/'

lastWordDict = pickle.load( open( savedModelDir+"lastWord_dict.pkl", "rb" ) )
firstWordDict = pickle.load( open( savedModelDir+"firstWord_dict.pkl", "rb" ) )
word_dict = pickle.load( open( savedModelDir+"word_dict.pkl", "rb" ) )


kks = pykakasi.kakasi()
tagger = fugashi.Tagger()
translator = Translator()



def japaneseTokenizer(text):
  for kanaStr in kks.convert(text):
    hiraStr = kanaStr['hira']
    print(hiraStr)
  return

def removeSoundPunct(text):
  punctList = ['っ','ょう','ゅう ','ょ','ゅ','ー','ゃ']
  for punct in punctList:
    text = text.replace(punct,'')
  return text

def punctPreprocess(text):
 
 
 
  punctList = ['！','？','、','!','?',',','(',')',"'",'"',"（","）","｣","｢",'･','､','」','』','…','。','～','〜','.','-','‥','“','「','『','【','<EOS>']
  text = text.translate(str.maketrans('', '', string.punctuation))
  for punct in punctList:
    text = text.replace(punct,'')
  return text
  
def lyricProcess(text):
  text = text.strip()
  text = text.replace('　','')
  text = text.replace('  ','')
  text = text.replace('	','')

  #text = text.replace('\n','')
  
  return text

def hasEnglishWords(word,tokenizer):
  resStr = ''
  for kanaStr in tokenizer(word):
    kanaStr = str(kanaStr)
    kanaStr = lyricProcess(punctPreprocess(kanaStr)).replace('\n','')
    #print(kanaStr)
    #if kanaStr and isEnglish(kanaStr):
    if isEnglish(kanaStr):
      return True
  return False

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def getLengthJPWord(word, hiraParser):
  resStr = ''
  word = lyricProcess(punctPreprocess(word))
  for kanaStr in hiraParser.convert(word):
    resStr += removeSoundPunct(kanaStr['hira'])
  resStr = resStr.replace('\n','')
  return len(resStr)

def convertToHira(word, hiraParser):
  resStr = ''
  for kanaStr in hiraParser.convert(word):
    resStr += kanaStr['hira']
  return resStr

def getSyllable(wordList, hiraParser):
  probDict = {}
  for k, v in wordList.items():
    probDict[k] = v / sum(wordList.values())
  for k, v in probDict.items():
    resStr = convertToHira(k, hiraParser)
    hiraLen = getLengthJPWord(k, hiraParser)

    print(k,' : ', hiraLen)
####
def getLastWord(verse,tokenizer):
  if not verse:
    return ''
  specialCharList = ['て']
  #lastWord = parser.convert(verse)[-1]['hira']
  try:
    lastWord = tokenizer(verse)[-1]
    #print(tokenizer(verse))
    if lastWord in specialCharList:
      #return parser.convert(verse)[-2]['orig'] + parser.convert(verse)[-1]['orig']
      return tokenizer(verse)[-2] + tokenizer(verse)[-1]
  except:

    print('Problem in get last word for ', tokenizer(str(verse)))
    lastWord = ''
  return lastWord

def getFirstWord(verse,tokenizer):
  if not verse:
    return ''
  specialCharList = ['て']

  #firstWord = parser.convert(verse)[0]['orig']
  try:
    firstWord = tokenizer(verse)[0]
  except:
    print('Problem in get first word for ', tokenizer(verse))
    firstWord = ''
  return firstWord



def translateToJapanese(word,hiraParser):
  resStr = ''
  
  for kanaStr in hiraParser.convert(word):
    if kanaStr and isEnglish(kanaStr['orig']) and (not kanaStr['orig'].replace(' ','').isdecimal()):
      resStr += translator.translate(kanaStr['orig'].lower(),dest='ja').text
      #print('English: ',kanaStr['orig'],'  ', resStr)
    else:
      resStr += kanaStr['orig']
  

  return resStr
  

def firstWordFromCorpus(corpusTxt,resDict,hiraParser,tokenizer):
  #resList = set()
  resList = []
  
  corpusTxt = lyricProcess(punctPreprocess(corpusTxt)).split('\n')
  
  for verse in corpusTxt:
    verse = translateToJapanese(verse,hiraParser)

    #resList.add(getFirstWord(verse,tokenizer))
    resList.append(getFirstWord(verse,tokenizer))

  for phrase in resList:
    try:
      phrase = str(phrase)
    except:
      #print('ErrorStr detected')
      #print(phrase)
      continue
    if phrase == '<EOS>':
      continue
    soundLen = getLengthJPWord(phrase,hiraParser)
    if soundLen in resDict:
      # We already have sound number in dictionary
       
      if phrase not in resDict[soundLen]:
        # phrase does not exist in dictionary
        #resDict[soundLen].append([phrase,1])
        resDict[soundLen][phrase] = 1
      else:
        phrasefreq = resDict[soundLen][phrase] + 1
        resDict[soundLen][phrase] = phrasefreq
      
      #resDict[soundLen].add(phrase)
      
    else:
      # We don't have sound number in dictionary
      resDict[soundLen] = {phrase:1}
      #resDict[soundLen] = {phrase}
      #resDict[soundLen] = {phrase,1}
  return resDict


def lastWordFromCorpus(corpusTxt,resDict,hiraParser,tokenizer):
  #resList = set()
  resList = []
  
  corpusTxt = lyricProcess(punctPreprocess(corpusTxt))
  
  for verse in corpusTxt.split('\n'):
    verse = translateToJapanese(verse,hiraParser)

    #resList.add(getLastWord(verse,tokenizer))
    lastStr = getLastWord(verse,tokenizer)
    lastStr = str(lastStr)
    resList.append(punctPreprocess(lastStr))
    #print('---',lastStr,'---')
  
  for phrase in resList:
    
    try:
      phrase = str(phrase)
    except:
      #print('ErrorStr detected')
      #print(phrase)
      continue
    #print('[',phrase,']')
    if phrase == '<EOS>':
      continue
    soundLen = getLengthJPWord(phrase,hiraParser)
    if soundLen in resDict:
      # We already have sound number in dictionary
       
      if phrase not in resDict[soundLen]:
        # phrase does not exist in dictionary
        #resDict[soundLen].append([phrase,1])
        resDict[soundLen][phrase] = 1
      else:
        phrasefreq = resDict[soundLen][phrase] + 1
        resDict[soundLen][phrase] = phrasefreq
      
      #resDict[soundLen].add(phrase)
      
    else:
      # We don't have sound number in dictionary
      resDict[soundLen] = {phrase:1}
      #resDict[soundLen] = {phrase}
      #resDict[soundLen] = {phrase,1}

  return resDict

def pykakasiTag(verse,hiraParser):
  resList = []
  hiraList = hiraParser.convert(verse)
  for origStr in hiraList:
    resList.append(str(origStr['orig']))

  return resList

def pykakasiTagDoubleWord(versePara,hiraParser,tokenizer):
  resList = []
  
  for verse in versePara.split('\n'):
    verse = str(verse.split('\n')[0])
    
    hiraList = hiraParser.convert(verse)
    
    if len(list(hiraList)) > 1:
      resList.append(str(getFirstWord(verse,tokenizer)))
      #resList.append(hiraList[0]['orig'] + hiraList[1]['orig'])
    for i,origStr in enumerate(hiraList):
      if (len(list(hiraList)) > 1 and i <= 1) or (len(list(hiraList)) > 2 and  i > len(list(hiraList))-2):
        # Skip the first two 
        continue
      else:
        resList.append(str(origStr['orig']))
    
    #print('hiraList: ',hiraList)
    #print('verse: [',verse,']')
    if len(list(hiraList)) > 2 and  i > len(list(hiraList))-2:
      resList.append(str(getLastWord(verse,tokenizer)))

  return resList


# The problem
def createNoteCluster(note,params,multiNoteList,track):
  
  noteDict = {}
  tokenDict = {}

  tokenSeq = [] # For keeping track of order of tokens

  tokenStr = ""
  if params is None:
    # First note step
    i = 0
    timeOffset = getNoteData(note, 't',i,track) - 5
    prevTime = getNoteData(note, 't',i,track)-timeOffset
    durTime = prevTime
  else:
    prevTime,durTime,timeOffset,noteDict,tokenDict,tokenSeq,i = params
  

  currTime = getNoteData(note, 't',i,track)-timeOffset

  if durTime < currTime:
      # This means there is a gap between ending duration of note and new note
      # So we add a value of 0
      #print('note: ',0,'   time: ',durTime,'  duration: ', currTime-durTime, '  velocity: ',0)
      if 0 in noteDict:
        noteDict[0] += 1
      else:
        noteDict[0] = 1

      multiNoteList.append(0)

      tokenStr = "n"+str(0)+"/d"+str(currTime-durTime)+"|"

      tokenSeq.append(tokenStr)
      if tokenStr in tokenDict:
        tokenDict[tokenStr] += 1
      else:
        tokenDict[tokenStr] = 1


  durTime = currTime + getNoteData(note, 'dur',i,track)
  noteData = getNoteData(note,'n',i,track)
  tokenStr = "n"+str(noteData)+"/d"+str(getNoteData(note, 'dur',i,track))+"|"
  multiNoteList.append(noteData)
  tokenSeq.append(tokenStr)

  if tokenStr in tokenDict:
    tokenDict[tokenStr] += 1
  else:
    tokenDict[tokenStr] = 1
  # Count frequency
  if getNoteData(note,'n',i,track) not in noteDict:
    noteDict[getNoteData(note,'n',i,track)] = 1
  else:
    noteDict[getNoteData(note,'n',i,track)] += 1
  i+= 1
  return (prevTime,durTime,timeOffset,noteDict,tokenDict,tokenSeq,i),multiNoteList

def divideListCluster(noteList):
  # using list comprehension + zip() + slicing + enumerate() 
  # Split list into lists by particular value 
  size = len(noteList) 
  idx_list = [idx + 1 for idx, val in
              enumerate(noteList) if val == 0] 
    
    
  res = [noteList[i: j] for i, j in
          zip([0] + idx_list, idx_list + 
          ([size] if idx_list[-1] != size else []))] 
    
  return res

def getNoteGroupCluster(vsqxPath):
  # Encode info into vector
  df = None
  params = None
  noteClusterList = []

  path = PurePath(vsqxPath)
  vsqx = xml.dom.minidom.parse(str(path))


  TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])


  mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)
  time = 0


  for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
    mf.addTrackName(trackNo, time, "Track {}".format(str(trackNo)))
    #print('track: ',trackNo)
    #i = 0
    timeOffset = 0

    prevTime = 0
    durTime = 0

    for noteCount,note in enumerate(track.getElementsByTagName('note')):
      #print(noteCount)
      params,noteClusterList = createNoteCluster(note,params,noteClusterList,track)
      #mf.addNote(trackNo, 0, getNoteData(note, 'n'), currTime / 480, getNoteData(note, 'dur') / 480, 64)
      
    #mf.addTempo(trackNo, time, TEMPO)
  return (params,noteClusterList)

# To get the 

def printNoteCluster(noteClusterList):
  for noteGroup in noteClusterList:
    lenCluster = len(noteGroup)-1
    print(lenCluster,' : ',noteGroup)

def combineSmallNoteClusterCount(noteClusterList):
  lenMin = 6

  resList = []
  tmpSum = 0
  for noteGroup in noteClusterList:
    lenCluster = len(noteGroup)-1
    if lenCluster > lenMin or tmpSum+lenCluster > lenMin:
      if tmpSum != 0:
        if tmpSum < lenMin or lenCluster < lenMin:
          resList.append(lenCluster+tmpSum)
          tmpSum = 0 
          continue
        else:
          resList.append(tmpSum)
        tmpSum = 0 
      resList.append(lenCluster)
    else:
      tmpSum += lenCluster
  return resList

def generateLyric(syllableCount,lengthWord,initalWord,hiraParser):
  if initalWord is None or initalWord not in word_dict or len(word_dict[initalWord]) == 0:
    word = getRandomWord(firstWordDict,lengthWord,word_dict)
  else:
    word = getNextWord(word_dict[initalWord])
  
  beginWordLen = getLengthJPWord(word,hiraParser)
  syllableTar = syllableCount - beginWordLen
  resultSeq = exploreList(word_dict[word],syllableTar, [[word,beginWordLen]],word_dict)
  resStr = ''.join([row[0] for row in resultSeq])
  sumNum = sum([row[1] for row in resultSeq])
  return resStr,sumNum

pronounceDict = {'ゅ','ゃ','ょ','ぇ','ぃ','ぉ'}

def hiraTokenizer(phrase):
  resStr = []
  lenPhrase = len(phrase)
  skip = False
  for i,letter in enumerate(phrase):
    if skip:
      skip = False
      continue
    if i+1 < lenPhrase and phrase[i+1] in pronounceDict:
      # We found a compound sound
      #print(letter+phrase[i+1])
      resStr.append(letter+phrase[i+1])
      skip = True
    else:
      #print(letter)
      resStr.append(letter)

  return resStr

multiConstant = 2

def getNextWord(wordList):
  probDict = {}
  for k, v in wordList.items():

    probDict[k] = v / sum(wordList.values())
  
  labelList = list(probDict.keys())
  weights = list(probDict.values())

  #print(len(labelList))
  chosenWord = punctPreprocess(random.choices(labelList, weights=weights, k=1)[0])


  return chosenWord

def getNextWordWithList(probDict):
  if probDict is None:
    probDict = {}
    for k, v in wordList.items():
      probDict[k] = v / sum(wordList.values())
    
  labelList = list(probDict.keys())
  weights = list(probDict.values())

  chosenWord = punctPreprocess(random.choices(labelList, weights=weights, k=1)[0])
  #chosenWord = random.choices(labelList, weights=weights, k=1)[0]

  return chosenWord,probDict



def generateNextWordList(wordList):
  resList = []
  wordListTmp = wordList.copy()
  probDict = None

  for i in range(0,len(wordListTmp)):
    word,probDict = getNextWordWithList(wordListTmp)
    resList.append(word)
    del wordListTmp[word]
  return resList

def getRandomWord(firstWordDict,syllableNum,word_dict): 
  probDict = {}
  if syllableNum <= 0:
    return ''

  for k, v in firstWordDict[syllableNum].items():
    #probDict[k] = v / sum(firstWordDict[syllableNum].values())
    v = v*multiConstant
    probDict[k] = v / sum(val*multiConstant for val in firstWordDict[syllableNum].values())
  labelList = list(probDict.keys())
  weights = list(probDict.values())

  while True:
    #first_word = np.random.choice(list(firstWordDict[syllableNum]))
    first_word = random.choices(labelList, weights=weights, k=1)[0]
    #print(first_word)

    chain = [first_word]
    word = chain[-1]
    #print(word_dict.keys())
    #if word in word_dict and not isEnglish(word):
    if word in word_dict and not hasEnglishWords(word,tagger):
      return word
    else:
      #print('Failed ', word)
      pass
  return word

def findAlternativeWord(word,syllable,wordLen,word_dict):
  if syllable - wordLen == 0:
    # Match in terms of syllable count
    if wordLen in lastWordDict and word in lastWordDict[wordLen]:
      # It means it is a perfect match
      return word,syllable,wordLen
    else:
      ### TEMPORARY
      #resWord = random.sample(lastWordDict[wordLen],1)[0]
      resWord = getRandomWord(lastWordDict,syllable,word_dict)

      return resWord,syllable,wordLen
  else:
    # Syllables don't exactly match
    # Temporary
    #resWord = random.sample(lastWordDict[syllable],1)[0]
    resWord = getRandomWord(lastWordDict,syllable,word_dict)

    return  resWord,syllable,syllable


def exploreList(wordList,syllable, resList,fullList):
  if syllable == 0:
    return resList
  #elif syllable < 0:
    
  #for k, v in wordList.items():
  
  #for k, v in sorted(wordList.items(), key=lambda x: random.random()):
  for k in generateNextWordList(wordList):
    #if isEnglish(k):
    if hasEnglishWords(k,tagger):
      continue
    #k = translateToJapanese(k,kks)

    nextWord = k
    lenOfNextWord = getLengthJPWord(nextWord, kks)

    #resList.append(k)
    resList.append([k,lenOfNextWord])

    if syllable <= 4:
      nextWord,syllable,lenOfNextWord = findAlternativeWord(nextWord,syllable,lenOfNextWord,fullList)
      #resList[-1] = nextWord
      resList[-1] = [nextWord,lenOfNextWord]
      return resList
    else:
      if syllable - lenOfNextWord > 0:
        syllableRemain = syllable - lenOfNextWord
      else:
        del resList[-1]
        continue
      #seqList = exploreList(wordList[probDict],syllableRemain, resList)
      return exploreList(fullList[nextWord],syllableRemain, resList,fullList)

  return resList

def getNoteData(note, key,indexSecond,track):
  #print(note.getElementsByTagName(key))
  try:
    return int(note.getElementsByTagName(key)[0].firstChild.data)
  except:
    secondIndDict = {'n':'noteNum','t':'posTick','dur':'durTick'}
    return int(track.getElementsByTagName(secondIndDict[key])[indexSecond].firstChild.data)
    
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]       
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device) 
    src_mask = model.make_src_mask(src_tensor)  
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor) 
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask) 
        pred_token = output.argmax(2)[:,-1].item() 
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention

def appendToMainList(mainList,translation):
  translation[:] = [item for item in translation if item != '<eos>']
  translation[:] = [item for item in translation if item != '']
  mainList += translation
  return mainList

def getAllNotesFromStr(token):
  tokList = token.split('n')
  
  resStr = ''
  for tok in tokList:
    #print(tok)
    resStr += tok.split('/')[0]
  return resStr

def addTokensToDup(tokenList,dupList,timeVal,K=6):
  for token in tokenList:
    if token in dupList:
      # We must increment frequency counter
      dupList[token] = [dupList[token][0]+1  ,  dupList[token][1]]
    else:
      dupList[token] = [1,timeVal]

  return dupList



def isDuplicate(tokenList,dupList,timeVal,pastThresh=3):
  for token in tokenList:
    if token in dupList:
      # We have a duplicate token

      # We check to see if the duplicate was recently
      if abs(timeVal - dupList[token]) > pastThresh:
        # The repeating token is long enough, so we won't label as duplicate
        continue
      #print('Duplicate: ', token)
      return True

  return False


# If a note/duration token appears within 2 seq too many times, we declare as duplicate
def isDuplicateSeq(tokenList,dupList,timeVal,pastThresh=2,freqThresh=4):
#def isDuplicateSeq(tokenList,dupList,timeVal,pastThresh=3,freqThresh=4):
  for token in tokenList:
    if token == '<eos>' or token == '':
      continue
    freq,prevTime = dupList[token]
    if abs(timeVal-prevTime) <= pastThresh:
      # The same token appeared within the past time interval
      # Now we check to see if it appeared too many times
      
      #if token == 'n71/d45':
      #  print('freq: ', freq, '   diffTime: ',abs(timeVal-prevTime))
      
      if freq >= freqThresh:
        # We will declare this as a duplicate

        # We will reset the counter
        dupList[token] = [1,timeVal]
        print(' duplicate token: [',token,']')
        return True, dupList
    
    else:
      # Not on the same thresh interval, we must reset duplist
      dupList[token] = [1,timeVal]
    
  return False, dupList
