# AI-Vocaloid-Kit-V2
A Python kit that uses deep learning to generate vocaloid music. Developers and musicians can now have AI support to aid and inspire ideas for their next vocaloid song.

Can take a folder of vsqx files and use Seq2seq model (encoders and decoders) in order to study pattern of a song and generate it’s own unique music (in the form of a vsqx file or midi file). 

It also uses markov models to generate its own accompanying Japanese lyrics that is in syllable sync with the melody. This means that if a melody verse contains 7 notes, the script will generate a lyric verse of 7 syllables to match it. It can also combine verses if one generated verse is too short So if there are 2 notes in the 1st verse and 5 notes in 2nd verse, the lyric verse combine those two verses and will generate a verse of 7 syllables (2+5=7). The corpus was web-scaped from the Studio48 site, containing lyrics from groups like AKB48, Nogizaka46, Keyakizaka46, etc.


The script will generate a vsqx file that contains the melody and lyrics built into it automatically. This can then be automatically loaded into a Vocaloid software (testedon Vocaloid Editor 4) to have the Vocaloid sing your AI generated song. There is also a manual option in which you can take your generated midi file
 and song and import it into your vocaloid editor yourself. 

For those who don't own a vocaloid editor, you can just listen to the generated midi file and read the text files containing the Japanese lyrics


What this AI vocaloid kit is capable of:
- Converting a directory of VSQX files into a training and validation dataset

- Using a Seq2Seq transformer deep learning architecture model to either randomly generate melodies from scratch or from an inital seed of midi notes

- Use markov models to generate Japanese lyrics and sync them to generated melodies from song. Counts notes and syllables to ensure flow between melody and lyrics.

- Allow users to train their own melody-generation model based on their VSQX dataset. 

# **Dependencies:**
Pip install the required dependencies:
```
	pip3 install -r requirements.txt
```


# **__How to run:__**
```
	python3 main.py --modelPath savedModels/9-22-music.pt --seed inputs/noteSeed.txt --dupThresh 3 --numOfNotes 200
```
The "--modelPath" argument is the path to the trained model

The "--seed" argument is the path to the file that contains the first few notes of a midi file

The "--dupThresh" parameter is a value to serve as a value to balance note harmony and repeating note sequences. Decrease the value decrease chance of duplication, though this can affect the note harmony among verses

The "--numOfNotes" parameter directly controls the length of the generated song.


Optionally, you can also take a vsqx file and generate a file with the first few notes to serve as a seed.
```
	python3 getSeed.py --filePath Crossing-Fields.vsqx --noteLimit 8 --outputPath testSeed.txt
```

# **__Training procedure:__**
If you want to train your own model on your own set of vsqx files, you have the ability to do so with these training scripts in the "scripts/" directory.

First, you would use this command to convert your directory of vsqx files into a csv dataset.
```
	python3 datasetConvert.py --vsqxDir All-song/ --seqLen 7 --stride 2
```
You can retrieve a sample dataset from here:

https://drive.google.com/drive/u/1/folders/1gr5FSq0X8cT--Eat5t61sspcoNRWNUiX

Then to begin the training process, run this:
```
	python3 train.py --modelOutput music-model.pt
```



