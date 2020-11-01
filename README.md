# AI-Vocaloid-Kit-V2
A Python kit that uses deep learning to generate vocaloid music. Developers and musicians can now have AI support to aid and inspire ideas for their next vocaloid song.

Can take a folder of vsqx files and use Seq2seq model (encoders and decoders) in order to study pattern of a song and generate it’s own unique music (in the form of a vsqx file or midi file). 

It also uses markov models to generate its own accompanying Japanese lyrics that is in syllable sync with the melody. This means that if a melody verse contains 7 notes, the script will generate a lyric verse of 7 syllables to match it. It can also combine verses if one generated verse is too short So if there are 2 notes in the 1st verse and 5 notes in 2nd verse, the lyric verse combine those two verses and will generate a verse of 7 syllables (2+5=7).


The script will generate a vsqx file that contains the melody and lyrics built into it automatically. This can then be automatically loaded into a Vocaloid software (testedon Vocaloid Editor 4) to have the Vocaloid sing your AI generated song. There is also a manual option in which you can take your generated midi file
 and song and import it into your vocaloid editor yourself. 

For those who don't own a vocaloid editor, you can just listen to the generated midi file and read the text files containing the Japanese lyrics


Thing this kit is capable of:



# **Dependencies:**
Pip install the required dependencies:
```
	pip3 install -r requirements.txt
```


# **__How to run:__**
The "--seed" argument is the path to the file that contains the first few notes of a midi file

```
	python3 main.py --seed inputs/noteSeed.txt
```
