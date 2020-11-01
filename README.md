# AI-Vocaloid-Kit-V2
A Python kit that uses deep learning to generate vocaloid music. Developers and musicians can now have AI support to aid and inspire ideas for their next vocaloid song.

Can take in midi song information and use Seq2seq model in order to study pattern of a song and generate it’s own unique music (in the form of midi file) as well as Markov Models to generate its own accompanying lyrics.

The script will generate a vsqx file that contains the melody and 

That midi file and generated lyrics can be manually then fed into a Vocaloid software (tested on Vocaloid Editor 4) to have the Vocaloid sing your AI generated song.

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
