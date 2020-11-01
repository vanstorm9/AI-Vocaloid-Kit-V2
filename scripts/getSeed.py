# python3 getSeed.py --filePath Crossing-Fields.vsqx --noteLimit 8 --outputPath testSeed.txt

# This is to convert VSQX to midi
from midiutil import MIDIFile

import xml.dom.minidom
import sys
from pathlib import PurePath
import os

import numpy as np
import argparse

def getNoteData(note, key,indexSecond,track):
  #print(note.getElementsByTagName(key))
  try:
    return int(note.getElementsByTagName(key)[0].firstChild.data)
  except:
    secondIndDict = {'n':'noteNum','t':'posTick','dur':'durTick'}
    return int(track.getElementsByTagName(secondIndDict[key])[indexSecond].firstChild.data)

parser = argparse.ArgumentParser(description='Commands for the vocaloid generator')
parser.add_argument('--filePath', dest="filePath",action="store",default='Crossing-Fields.vsqx',
                   help='Use beginning notes to initalize melody generation.')
parser.add_argument('--noteLimit', dest="noteLimit",action="store",type=int,default=8,
                   help='The number of notes in the seed')
parser.add_argument('--outputPath', dest="outputPath",action="store",default='testSeed.txt',
                   help='Output path of the text file containing the note seed')
args = parser.parse_args()



vsqxPath = args.filePath
noteLimit = args.noteLimit
txtPath = args.outputPath

#vsqxPath = '/content/AiDee-simplified.vsqx'

assert os.path.exists(vsqxPath)


extentStr = vsqxPath.split('.')[-1]

if extentStr == 'vsqx':
	path = PurePath(vsqxPath)
	vsqx = xml.dom.minidom.parse(str(path))
	TEMPO = int(vsqx.getElementsByTagName('tempo')[0].childNodes[1].firstChild.data[:-2])
	mf = MIDIFile(len(vsqx.getElementsByTagName('vsTrack')), removeDuplicates=False)

	time = 0
	f = open(txtPath,"w")

	for trackNo, track in enumerate(vsqx.getElementsByTagName('vsTrack')):
		for i,note in enumerate(track.getElementsByTagName('note')):
			if i >= noteLimit:
				f.close()
				exit()

			n = getNoteData(note,'n',i,trackNo)
			dur = getNoteData(note, 'dur',i,trackNo)
			saveStr = 'n'+str(n)+'/d'+str(dur)+'\n'
			f.write(saveStr)

			print('note: ',n,'   time: ',getNoteData(note,'t',i,trackNo),'  duration: ', dur)
	f.close()
	# Work in progress, as we need a way to extract midi duration
	'''
	elif extentStr == 'mid' or extentStr == 'midi':
		print('Midi')
	'''
else:
	#print('Seed file must be a "vsqx" file or a "mid" or "midi" file')
	print('Seed file must be a "vsqx"')
	raise