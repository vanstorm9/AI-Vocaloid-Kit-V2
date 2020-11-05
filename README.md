# AI-Vocaloid-Kit-V2
A Python kit that uses deep learning to generate vocaloid music. Developers and musicians can now have AI support to aid and inspire ideas for their next vocaloid song.

Can take a folder of vsqx files and use Seq2seq model (encoders and decoders) in order to study pattern of a song and generate it’s own unique music (in the form of a vsqx file or midi file). 

![GitHub Logo](/images/video-gif.gif)

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
	python3 main.py --modelPath savedModels/9-22-music.pt --seed inputs/noteSeed.txt --dupThresh 3 --numOfNotes 50
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

# **__Design notes:__**
Previous version of the AI Vocaloid Kit used midi files as a main source as music data for training and testing. This changed in this version as the current version uses VSQX files (Vocaloid Editor file format) for dataset conversion, training, and testing. While there are a lot more midi files than VSQX files, there were a lot of variables that affeted the quality of the data processing depending on the file. The main purpose of this kit is to generate melodies that would be sung from a singer's voice (aka monophonic melody). This means that midi files that had polyphonic properties (playing multiple melodies on different channels simultanous) were not suitable candidates to be used for training. Also a majority of midi files are based off from a wide variety of instruments, not just singing. Rapid, fluctuating melodies from a piano does not flow the same way as a more steady singing voice, so vsqx files were chosen as the main format for note and data representation instead of pure midi files. As a result, the melodies produced by this version is of a higher quality at a much more consistant rate.

The previous version of this kit had scripts that could convert Youtube videos to midi files, which in turn would be feed into the network to serve as a seed for note generation. However there were problems with this as the Youtube conversion scripts were not accurate (threfore poor quality seed input) along with one of its components from a conversion library being only supported in Python 2.7. Users can choose to use make their own scripts/use 3rd party scripts if they desire though.

# **__Possible improvements that can be made:__**

The number of VSQX files used for training the current model is below 100 files, which is a relatively low number in a machine learning context. It was hard for me to gather a lot of VSQX, so if a much larger number of VSQX files gets collected in the future, retraining can help improve the quality of generated melodoies.

In terms of lyrics, while the markov model does have decent respect for Japanese grammar structure, the choice of nouns and verbs chosen can sometimes lead to a sentence combination that does not make a whole lot of sense. Investing in a deep learning model that generates higher-quality sentences as well as respect syllable count constraints would be the ideal solution in the future.    

There are certain songs in which syllables are stressed and extended across multiple notes, which makes data extraction and lyrics generation more complex. Methods to normalize dataset creation as well as add lyric variety with stressed and unstressed words would be investigated in the future.
