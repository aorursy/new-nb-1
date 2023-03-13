#%reset

#!pip install pyspellchecker

#!pip install googletrans

import pandas as pd

import matplotlib as plt

import re

import numpy as np

#from spellchecker import SpellChecker

from nltk.stem import WordNetLemmatizer

#from googletrans import Translator

from textblob import Word

from gensim.test.utils import datapath, get_tmpfile

from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec

import nltk

import gensim

from nltk.corpus import stopwords

#nltk.download('stopwords')

#nltk.download('punkt')

#nltk.download('wordnet')

#nltk.download('averaged_perceptron_tagger')

import gc

gc.enable()


import emoji

from nltk.corpus import wordnet

import datetime

import time

import operator

from textblob import TextBlob

from tqdm import tqdm, trange

from nltk.tokenize import TweetTokenizer



from IPython.display import clear_output

clear_output()
def replace_contractions(text):

  

  """

  This functions check's whether a text contains contractions or not. 

  In case a contraction is found, the corrected value from the dictionary is 

  returned.

  Example: "I've" towards "I have"

  """

  

  #replace words with "'ve" to "have"

  matches = re.findall(r'\b\w+[\'`´]ve\b', text)

  if len(matches) != 0:

    text = re.sub(r'[\'`´]ve\b', " have", text)

  

  #replace words with "'re" to "are"

  matches = re.findall(r'\b\w+[\'`´]re\b', text)

  if len(matches) != 0:

    text = re.sub(r'[\'`´]re\b', " are", text)

  

  #replace words with "'ll" to "will"

  matches = re.findall(r'\b\w+[\'`´]ll\b', text)

  if len(matches) != 0:

    text = re.sub(r'[\'`´]ll\b', " will", text)

  

  #replace words with "'m" to "am"

  matches = re.findall(r'\b\w+[\'`´]m\b', text)

  if len(matches) != 0:

    text = re.sub(r'[\'`´]m\b', " am", text)

  

  #replace words with "'d" to "would"

  matches = re.findall(r'\b\w+[\'`´]d\b', text)

  if len(matches) != 0:

    text = re.sub(r'[\'`´]d\b', " would", text)

  

  #replace words with contraction according to the contraction_dict

  matches = re.findall(r'\b\w+[\'`´]\w+\b', text)

  for x in matches:

    if x in contraction_dict.keys():

      text = text.replace(x, contraction_dict.get(x))

  

  # replace all "'s" by space

  matches = re.findall(r'\b\w+[\'`´]s\b', text)

  if len(matches) != 0:

    text = re.sub(r'[\'`´]s\b', " ", text)

  return text



# Dictionary of contractions coming out of the pre-investigation in the other kernel

contraction_dict = {"Can't":"can not", "Didn't":"did not", "Doesn't":"does not", 

                    "Isn't":"is not", "Don't":"do not", "Aren't":"are not", "#":"",

                    "ain't": "is not", "aren't": "are not","can't": "cannot",

                    "'cause": "because", "could've": "could have", "couldn't": "could not",

                    "didn't": "did not",  "doesn't": "does not", "don't": "do not",

                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",

                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will",

                    "I'll've": "I will have","I'm": "I am", "I've": "I have",

                    "i'd": "i would", "i'd've": "i would have", "i'll": "i will",

                    "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not",

                    "it'd": "it would", "it'd've": "it would have", "it'll": "it will",

                    "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                    "mayn't": "may not", "might've": "might have","mightn't": "might not",

                    "mightn't've": "might not have", "must've": "must have",

                    "mustn't": "must not", "mustn't've": "must not have",

                    "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",

                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",

                    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                    "she's": "she is", "should've": "should have", "shouldn't": "should not",

                    "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                    "this's": "this is","that'd": "that would", "that'd've": "that would have",

                    "that's": "that is", "there'd": "there would", "there'd've": "there would have",

                    "there's": "there is", "here's": "here is","they'd": "they would",

                    "they'd've": "they would have", "they'll": "they will",

                    "they'll've": "they will have", "they're": "they are", "they've": "they have",

                    "to've": "to have", "wasn't": "was not", "we'd": "we would",

                    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",

                    "we're": "we are", "we've": "we have", "weren't": "were not",

                    "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                    "what's": "what is", "what've": "what have", "when's": "when is",

                    "when've": "when have", "where'd": "where did", "where's": "where is",

                    "where've": "where have", "who'll": "who will", "who'll've": "who will have",

                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",

                    "will've": "will have", "won't": "will not", "won't've": "will not have",

                    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",

                    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",

                    "y'all're": "you all are","y'all've": "you all have","you'd": "you would",

                    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                    "you're": "you are", "you've": "you have", "c'mon":"come on",

                    "Don''t":"do not", "Haden't":"had not", "Grab'em":"grab them", "USA''s":"USA",

                    "Pick'em":"pick them", "I'lll":"I will", "Tell'em":"tell them", "Y'all":"you all",

                    "Wouldn't":"would not", "Shouldn't":"should not", "I'DVE":"I would have",

                    "SHOOT'UM":"shoot them", "CANN'T":"can not", "COUD'VE":"could have", "Yo'ure":"you are",

                    "LOCK'EM":"lock them", "G'night":"goodnight", "W'ell":"we will", "IT'D":"it would",

                    "Couldn't":"could not", "LOCK'UM":"lock them", "WOULD'NT":"would not", "Cant't":"can not",

                    "HADN'T":"had not", "It''s":"it is", "Don'ts":"do not", "Arn't":"are not",

                    "We'll":"we will", "G'Night":"goodnight", "THAT'LL":"that will", "Dpn't":"do not",

                    "Idon'tgetitatall":"I do not get it at all", "THEY'VE":"they have", "Le'ts":"let us",

                    "SEND'EM":"send them", "AIN'T":"is not", "WE'D":"we would", "I'vemade":"I have made",

                    "SHE'LL":"she will", "I'llbe":"I will be", "I'mma":"I am a", "Could'nt":"could not",

                    "You'very":"you are very", "Light'em":"light them", "Con't":"can not", "I'Μ":"I am",

                    "Kick'em":"kick them", "Shoudn't":"should not", "That''s":"that is",

                    "Didn't_work":"did not work", "You'rethinking":"you are thinking", "Dn't":"do not",

                    "CON'T":"can not", "DON'T":"do not", "C'Mon":"come on", "You'res":"you are",

                    "Amn't":"is not", "WE'RE":"we are", "Can't":"can not", "Kouldn't":"could not",

                    "SHouldn't":"should not", "Does't":"does not", "COULD'VE":"could have",

                    "TrumpIDin'tCare":"Trump did not care", "Iv'e":"I have", "Dose't":"does not",

                    "DOESEN'T":"does not", "Give'em":"give them", "Won'tdo":"will not do",

                    "They'l":"they will", "He''s":"he is", "I'veve":"I have", "Wern't":"were not",

                    "Pay'um":"pay them", "She''l":"she will", "Y'know":"you know", "DIdn't":"did not",

                    "O'bamacare":"Obamacare", "I'ma":"I am a", "Ma'am":"madam", "WASN'T":"was not",

                    "Dont't":"do not", "Is't":"is not", "OU'RE":"you are", "YOU'RE":"you are",

                    "Ther'es":"there is", "C'mooooooon":"come on", "They_didn't":"they did not",

                    "Som'thin":"something", "Love'em":"love them", "You''re":"you are", "I'D":"I would",

                    "HASN'T":"has not", "WOULD'VE":"would have", "WAsn't":"was not", "ARE'NT":"are not",

                    "Dowsn't":"does not", "It'also":"it is also", "Geev'um":"give them", "Theyv'e":"they have",

                    "Theyr'e":"they are", "Take'em":"take them", "Book'em":"book them", "Havn't":"have not",

                    "DOES'NT":"does not", "Who''s":"who is", "WON't":"will not", "I'Il":"I will",

                    "I'don":"I do not", "AREN'T":"are not", "Ev'rybody":"everybody", "Hold'um":"hold them",

                    "WE'LL":"we will", "Cab't":"can not", "IJustDon'tThink":"I just do not think",

                    "Wouldn'T":"would not", "U'r":"you are", "I''ve":"I have", "DONT'T":"do not",

                    "G'morning":"good morning", "You'ld":"you would", "We''ll":"we will", "YOUR'E":"you are",

                    "TrumpDoesn'tCare":"Trump does not care", "Wasn't":"was not", "You'all":"you all",

                    "Y'ALL":"you all", "G'bye":"goodbye", "YOU'VE":"you have", "Does'nt":"does not",

                    "Don'TCare":"do not care",  "Weren't":"were not", "Y'All":"you all", "They'lll":"they will",

                    "You'reOnYourOwnCare":"you are on your own care", "I'veposted":"I have posted",

                    "Run'em":"run them", "Vote'em":"vote them", "Would't":"would not", "I'l":"I will",

                    "Ddn't":"did not", "I'mm":"I am", "Sshouldn't":"should not", "Your'e":"you are",

                    "I'v":"I have", "We'really":"we are really", "DOESN'T":"does not", "DiDn't":"did not",

                    "Needn't":"need not", "They'er":"they are", "Look'em":"look them", "I'vÈ":"I have",

                    "Didn`t":"did not", "I'lll":"I will", "Wouldn't":"would not", "It`s":"it is", "What's":"what is",

                    "ISN`T":"is not", "WE'RE":"we are", "Are'nt":"are not", "DOesn't":"does not", "I'M":"I am",

                    "WON'T":"will not", "WEREN'T":"were not", "TrumpDon'tCareAct":"Trump do not care act",

                    "HAVEN'T":"have not", "That''s":"that is", "Do'nt":"do not"}
def replace_symbol_special(text,check_vocab=False, vocab=None): 



    ''' 

    This method can be used to replace dashes ('-') around and within the words using regex.

    It only removes dashes for words which are not known to the vocabluary.

    Next to that, common word separators like underscores ('_') and slashes ('/') are replaced by spaces. 

    '''



        

    # replace all dashes and abostropes at the beginning of a word with a space

    matches = re.findall(r"\s+(?:-|')\w*", text)

    # if there is a match is in text

    if len(matches) != 0:

      # remove the dash from the match or better text

      for match in matches:

        text = re.sub(match, re.sub(r"(?:-|')", ' ', match), text)

    

    # replace all dashes and abostrophes at the end of a word with a space

    # function works as above

    matches = re.findall(r"\w*(?:-|')\s+", text)

    if len(matches) != 0:

      for match in matches:

        text = re.sub(match, re.sub(r"(?:-|')", ' ', match), text)

    

    if check_vocab == True:

      # replace dashes and abostrophes in the middle of the word only in case it is not known to a dictionary

      # function works as above

      matches = re.findall(r"\w*(?:-|')\w*", text)

      if len(matches) != 0:

        for match in matches:

          #check if the word with dash in the middle in in the vocabluary

          if match not in vocab.keys():

            text = re.sub(match, re.sub(r"(?:-|')", ' ', match), text)

    

    #

    text = re.sub(r'(?:_|\/)', ' ', text)

    

    text = re.sub(r' +', ' ', text)#-

    return text

  

# Initially we consideredto remove the dash for words with this beginning. 

# However we found that it had almost no impact. Applying it to the total text, would kill correct spellings.

# pre_suffix_dict = {'bi-':'bi', 	'co-':'co','re-':'re',	'de-':'de','pre-':'pre',	'sub-':'sub', 'un-':'un'}




def find_smilies(text):

  

  '''

  For investigation only: Find most common keyboard typed smilies in the text.

  '''

  

  #define a pattern to find typical keyboard smilies

  pattern = r"((?:3|<)?(?::|;|=|B)(?:-|'|'-)?(?:\)|D|P|\*|\(|o|O|\]|\[|\||\\|\/)\s)"

  # Find the matches n the text

  matches = re.findall(pattern, text)

  # If the text contain matches print the text and the smilies found

  if len(matches) != 0:

    print(text, matches)

    

    



    

def replace_smilies(text):

  

  '''

  Simplyfied method to replace keyboard smilies with its very simple translation.

  '''

  

  #Find and replace all happy smilies

  matches = re.findall(r"((?:<|O|o|@)?(?::|;|=|B)(?:-|'|'-)?(?:\)|\]))", text)

  if len(matches) != 0:

    text = re.sub(r"((?:<|O|o|@)?(?::|;|=|B)(?:-|'|'-)?(?:\)|\]))", " smile ", text)

  

  #Find and replace all laughing smilies

  matches = re.findall(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:d|D|P|p)\b)", text)

  if len(matches) != 0:

    text = re.sub(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:d|D|P|p)\b)", " smile ", text)

  

  #Find and replace all unhappy smilies

  matches = re.findall(r"((?:3|<)?(?::|;|=|8)(?:-|'|'-)?(?:\(|\[|\||\\|\/))", text)

  if len(matches) != 0:

    text = re.sub(r"((?:3|<)?(?::|;|=|8)(?:-|'|'-)?(?:\(|\[|\||\\|\/))", " unhappy ", text)

  

  #Find and replace all kissing smilies

  matches = re.findall(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:\*))", text)

  if len(matches) != 0:

    text = re.sub(r"((?:<)?(?::|;|=)(?:-|'|'-)?(?:\*))", " kiss ", text)

  

  #Find and replace all surprised smilies

  matches = re.findall(r"((?::|;|=)(?:-|'|'-)?(?:o|O)\b)", text)

  if len(matches) != 0:

    text = re.sub(r"((?::|;|=)(?:-|'|'-)?(?:o|O)\b)", " surprised ", text)

    

  #Find and replace all screaming smilies

  matches = re.findall(r"((?::|;|=)(?:-|'|'-)?(?:@)\b)", text)

  if len(matches) != 0:

    text = re.sub(r"((?::|;|=)(?:-|'|'-)?(?:@)\b)", " screaming ", text)

    

  #Find and replace all hearts

  matches = re.findall(r"♥|❤|<3|❥|♡", text)

  if len(matches) != 0:

    text = re.sub(r"(?:♥|❤|<3|❥|♡)", " love ", text)

  

  text = re.sub(' +', ' ',text)

  return text
def remove_stopwords(text, stop_words):

  

  ''' 

  Remove stopwords and multiple whitespaces around words

  '''

  

  #Compile stopwords separated by | and stopped by word boundary 

  stopword_re = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b')

  # Replace the stopwords by space

  text = stopword_re.sub(' ', text)

  #Replace double spaces by a single space

  text = re.sub(' +', ' ',text)

  return text
def clean_text(text, scope='general'):

  

  '''

  This function handles text cleaning from various symbols.

  - it translates special font types into the standard text type of python.

  - it removes all symbols except for dashes and abostrophes being handled by 

    "replace_symbol_special".

  - it handles multi letter appearances like "comiiii" > "comi"

  - typical unknown words like "Trump"

  '''

  

  

  

  #compile all special symbols from the dictionary to one regex function

  translate_regex = re.compile(r'(' + r'|'.join(translate_dictionary.keys()) + r')')

  

  # find all matches of special symbols in the text

  matches = re.findall(translate_regex, text)

  # if there is one or more matches

  if len(matches) != 0:

    for x in matches:

      if x in translate_dictionary.keys():

        #replace the symbol by its replacement item

        text = re.sub(x, translate_dictionary.get(x), text)

  

  # find and remove all "http" links

  matches = re.findall(r'http\S+', text)

  if len(matches) != 0:

    text = re.sub(r'http\S+', '', text)

  

  #remove all backslashes

  matches = re.findall(r'\\', text)

  if len(matches) != 0:

    text = re.sub(r'\\', ' ', text)

  

  # compile all remaining special characters into one translate line and replace them by space

  # the translate function is really fast thus here our preferred choice

  text = text.translate(str.maketrans(''.join(puncts), len(''.join(puncts))*' '))  

  

  #find words where 4 repetitions of a letter goes in a row and reduce them to only one

  #we are not correcting words with 2 or three identical letters in a row as this could destroy correct words

  #first find repeating characters

  matches = re.findall(r'(.)\1{3,}', text)

  # is some are found

  if len(matches) != 0:

    #for each match replace it with its first letter (x[0])

    for x in matches:

      character_re = re.compile(x + '{3,}')

      matchesInside = re.findall(character_re, text)

      if len(matchesInside) != 0:

        for x in matchesInside:

          text = re.sub(x, x[0], text)

          

  # hahaha s by one haha 

  matches = re.findall(r'\b[h,a]{4,}\b', text)

  if len(matches) != 0:

    text = re.sub(r'\b[h,a]{4,}\b', 'haha', text)

  

  # as we found many unknown word variations including 'Trump' we reduce thse  words just to Trump

  # being represented in most word vectors

  matches = re.findall(r'\w*[Tt][Rr][uU][mM][pP]\w*', text)

  if len(matches) != 0:

    for x in matches:

      text = re.sub(x, 'Trump', text)

      

  #remove potential double spaces generated during processing        

  text = re.sub(' +', ' ',text) 

  

  # those symbols are not touched by this function ->see replace_contraction or replace_special_symbols

  #keep = ["'", '-', '´']

  

  

  return text











# The dictionary was generated in the compare and investigation phase in the other notebook

translate_dictionary = {'\t': 't', '0': '0', '1': '1', '2': '2', '3': '3', '5': '5', '6': '6',

                         '8': '8', '9': '9', 'd': 'd', 'e': 'e', 'h': 'h', 'm': 'm', 't': 't',

                         '²': '2', '¹': '1', 'ĝ': 'g', 'œ': 'ae', 'ŝ': 's', 'ǧ': 'g', 'ɑ': 'ɑ',

                         'ɒ': 'a', 'ɔ': 'c', 'ə': 'e', 'ɛ': 'e', 'ɡ': 'g', 'ɢ': 'g', 'ɪ': 'i',

                         'ɴ': 'n', 'ʀ': 'r', 'ʏ': 'y', 'ʙ': 'b', 'ʜ': 'h', 'ʟ': 'l', 'ʰ': 'h',

                         'ʳ': 'r', 'ʷ': 'w', 'ʸ': 'y', 'ˢ': '5', '͞': '-', '͟': '_', 'ͦ': 'o',

                         'Α': 'a', 'Β': 'b', 'Ε': 'e', 'Μ': 'm', 'Ν': 'n', 'Ο': 'o', 'Τ': 't',

                         'έ': 'e', 'ί': 'i', 'α': 'a', 'κ': 'k', 'χ': 'x', 'І': 'i', 'А': 'a',

                         'Б': 'e', 'Е': 'e', 'З': '#', 'И': 'n', 'К': 'k', 'М': 'm', 'Н': 'h',

                         'О': 'o', 'Р': 'p', 'С': 'c', 'У': 'y', 'Х': 'x',  'в': 'b',

                         'к': 'k', 'м': 'm', 'н': 'h', 'ы': 'bi', 'ь': 'b', 'ё': 'e', 'љ': 'jb',

                         'ғ': 'f', 'ү': 'y', 'Ԝ': 'w', 'հ': 'h', 'א': 'n', '௦': '0', '౦': 'o',

                         '൦': 'o', '໐': 'o', 'Ꭵ': 'i', 'Ꭻ': 'j', 'Ꮷ': 'd', 'ᐨ': '-', 'ᐸ': '<',

                         'ᑲ': 'b', 'ᑳ': 'b', 'ᗞ': 'd', 'ᴀ': 'a', 'ᴄ': 'c', 'ᴅ': 'n', 'ᴇ': 'e',

                         'ᴊ': 'j', 'ᴋ': 'k', 'ᴍ': 'm', 'ᴏ': 'o', 'ᴑ': 'o', 'ᴘ': 'p', 'ᴛ': 't',

                         'ᴜ': 'u', 'ᴠ': 'v', 'ᴡ': 'w', 'ᴵ': 'i', 'ᴷ': 'k', 'ᴺ': 'n', 'ᴼ': 'o',

                         'ᵉ': 'e', 'ᵒ': 'o', 'ᵗ': 't', 'ᵘ': 'u', 'ẃ': 'w', 'ἀ': 'a', 'Ἀ': 'a',

                         'Ἄ': 'a', 'ὶ': 'l', 'ὺ': 'u', '‒': '-', '₁': '1', '₃': '3', '₄': '4',

                         'ℋ': 'h', '℠': 'sm', 'ℯ': 'e', 'ℴ': 'c', '╌': '--', 'ⲏ': 'h', 'ⲣ': 'p',

                         '下': 'under', '不': 'Do not', '人': 'people', '伎': 'trick', '会': 'meeting',

                         '作': 'Make', '你': 'you', '克': 'Gram', '关': 'turn off', '别': 'do not',

                         '加': 'plus', '华': 'China', '卖': 'Sell', '去': 'go with', '哥': 'brother',

                         '园': 'garden', '国': 'country', '圆': 'circle', '土': 'soil', '地': 'Ground',

                         '坏': 'Bad', '外': 'outer', '大': 'Big', '失': 'Lost', '子': 'child', '小': 'small',

                         '成': 'to make', '戦': 'War', '所': 'Place', '拿': 'take', '故': 'Therefore',

                         '文': 'Text', '明': 'Bright', '是': 'Yes', '有': 'Have', '歌': 'song', 

                         '殊': 'special', '油': 'oil', '温': 'temperature', '特': 'special', 

                         '獄': 'prison', '的': 'of', '税': 'tax', '系': 'system', '群': 'group',

                         '舞': 'dance', '英': 'English', '蔡': 'Cai', '议': 'Discussion', '谷': 'Valley',

                         '豆': 'beans', '都': 'All', '钱': 'money', '降': 'drop', '障': 'barrier',

                         '骗': 'cheat', '세': 'three', '안': 'within', '영': 'spirit', '요': 'Yo',

                          'ͺ': '', 'Λ': 'L', 'Ξ': 'X', 'ά': 'a', 'ή': 'or', 'ι': 'j',

                         'ξ': 'X', 'ς': 's', 'ψ': 't', 'ό': 'The', 'ύ': 'gt;', 'ώ': 'o',

                         'ϖ': 'e.g.', 'Г': 'R', 'Д': 'D', 'Ж': 'F', 'Л': 'L', 'П': 'P', 

                         'Ф': 'F', 'Ш': 'Sh', 'б': 'b', 'п': 'P', 'ф': 'f', 'ц': 'c', 

                         'ч': 'no', 'ш': 'sh', 'щ': 'u', 'э': 'uh', 'ю': 'Yu', 'ї': 'her',

                         'ћ': 'ht', 'Ձ': 'Winter', 'ա': 'a', 'դ': 'd', 'ե': 'e', 'ի': 's',

                         'ձ': 'h', 'մ': 'm', 'յ': 'y', 'ն': 'h', 'ռ': 'r', 'ս': 'c', 

                         'ր': 'p', 'ւ': '³', 'ב': 'B', 'ד': 'D', 'ה': 'God', 'ו': 'and',

                         'ט': 'ninth', 'י': 'J', 'ך': 'D', 'כ': 'about', 'ל': 'To', 'ם': 'From', 

                         'מ': 'M', 'ן': 'Estate', 'נ': 'N', 'ס': 'S.', 'ע': 'P', 'ף': 'Jeff',

                         'פ': 'F', 'צ': 'C', 'ק': 'K.', 'ר': 'R.', 'ש': 'That', 'ת': 'A',

                         'ء': 'Was', 'آ': 'Ah', 'أ': 'a', 'إ': 'a', 'ا': 'a', 'ة': 'e', 

                         'ت': 'T', 'ج': 'C', 'ح': 'H', 'خ': 'Huh', 'د': 'of the', 'ر': 'T',

                         'ز': 'Z', 'س': 'Q', 'ش': 'Sh', 'ص': 's', 'ط': 'I', 'ع': 'AS', 'غ': 'G',

                         'ف': 'F', 'ق': 'S', 'ك': 'K', 'ل': 'to', 'م': 'M', 'ن': 'N', 'ه': 'e', 

                         'و': 'And', 'ى': 'I', 'ي': 'Y', 'چ': 'What', 'ک': 'K', 'ی': 'Y', 

                         'क': 'A', 'म': 'M', 'र': 'And', 'ગ': 'C', 'જ': 'The same', 

                         'ત': 'I', 'ર': 'I', 'ஜ': 'SAD', 'ლ': 'L', 'ṑ': 'o', 'ἐ': 'e',

                         'ἔ': 'Ë', 'ἡ': 'or', 'ἱ': 'ı', 'ἴ': 'i', 'ὀ': 'The', 'ὁ': 'The',

                         'ὐ': 'ÿ', 'ὰ': 'a', 'ὲ': '.', 'ὸ': 'The', 'ύ': 'gt;', 'ᾶ': 'a', 

                         'ῆ': 'or', 'ῖ': 'ก', 'ῦ': 'I', 'う': 'U', 'さ': 'The', 'っ': 'What',

                         'つ': 'One', 'な': 'The', 'よ': 'The', 'ら': 'Et al', 'エ': 'The', 

                         'ク': 'The', 'サ': 'The', 'シ': 'The', 'ジ': 'The', 'ス': 'The',

                         'チ': 'The', 'ツ': 'The', 'ニ': 'D', 'ハ': 'Ha', 'マ': 'Ma', 

                         'リ': 'The', 'ル': 'Le', 'レ': 'Les', 'ロ': 'The', 'ン': 'The',

                         '一': 'One', '与': 'versus', '且': 'And', '为': 'for', '买': 'buy',

                         '了': 'Up', '些': 'some', '他': 'he', '以': 'Take', '们': 'They',

                         '件': 'Items', '传': 'pass', '伦': 'Lun', '但': 'but', '信': 'letter',

                         '候': 'Waiting', '偽': 'Pseudo', '全': 'all', '公': 'public', '其': 'its',

                         '养': 'support', '冬': 'winter', '凸': 'Convex', '击': 'hit', '判': 'Judge',

                         '到': 'To', '友': 'Friend', '可': 'can', '吗': 'What?', '和': 'with',

                         '唯': 'only', '因': 'because', '圣': 'Holy', '在': 'in', '基': 'base',

                         '堂': 'Hall', '士': 'Shishi', '复': 'complex', '多': 'many', '天': 'day',

                         '好': 'it is good', '如': 'Such as', '婚': 'marriage', '孩': 'child', 

                         '宠': 'Pet', '寓': 'Apartment', '对': 'Correct', '屁': 'fart', 

                         '屈': 'Qu', '巨': 'huge', '己': 'already', '式': 'formula', '当': 'when',

                         '彼': 'he', '徒': 'only', '得': 'Got', '怒': 'angry', '怪': 'strange',

                         '恐': 'fear', '惧': 'fear', '想': 'miss you', '愤': 'anger', '我': 'I',

                         '战': 'war', '批': 'Batch', '把': 'Put', '拉': 'Pull', '拷': 'Copy', 

                         '接': 'Connect', '操': 'Fuck', '收': 'Receive', '政': 'Politics', 

                         '教': 'teach', '斤': 'jin', '斯': 'S', '新': 'new', '时': 'Time', 

                         '普': 'general', '曾': 'Once', '本': 'this', '杀': 'kill', '极': 'pole',

                         '查': 'check', '栗': 'chestnut', '株': 'stock', '样': 'kind', '检': 'Check',

                         '欢': 'Happy', '死': 'dead', '汉': 'Chinese', '没': 'No', '治': 'rule', 

                         '法': 'law', '活': 'live', '点': 'point', '燻': 'Moth', '物': 'object',

                         '猜': 'guess', '猴': 'monkey', '理': 'Rational', '生': 'Health', '用': 'use',

                         '白': 'White', '百': 'hundred', '直': 'straight', '相': 'phase', '看': 'Look',

                         '督': 'Supervisor', '知': 'know', '社': 'Society', '祝': 'wish', '积': 'product',

                         '稣': 'Jesus', '经': 'through', '结': 'Knot', '给': 'give', '美': 'nice', 

                         '耶': 'Yay', '聊': 'chat', '胜': 'Win', '至': 'to', '虚': 'Virtual', '製': 'Made', 

                         '要': 'Want', '认': 'recognize', '讨': 'discuss', '让': 'Let', '识': 'knowledge',

                         '话': 'words', '语': 'language', '说': 'Say', '谊': 'friendship', 

                         '谓': 'Predicate', '象': 'Elephant', '贺': 'He', '赢': 'win', '迎': 'welcome',

                         '还': 'also', '这': 'This', '通': 'through', '鉄': 'iron', '问': 'ask', 

                         '阿': 'A', '题': 'question', '额': 'amount', '鬼': 'ghost', '鸡': 'Chicken',

                         '가': 'end', '갈': 'Go', '게': 'to', '격': 'case', '경': 'circa', '관': 'tube',

                         '국': 'soup', '금': 'gold', '나': 'I', '는': 'The', '니': 'Nee', '다': 'All',

                         '대': 'versus', '도': 'Degree', '된': 'The', '드': 'De', '들': 'field', 

                         '때': 'time', '런': 'Run', '렵': 'Hi', '록': 'rock', '뤼': 'Crown', 

                         '리': 'Lee', '마': 'hemp', '만': 'just', '반': 'half', '분': 'minute', 

                         '사': 'four', '상': 'Prize', '서': 'book', '석': 'three', '성': 'castle',

                         '스': 'The', '시': 'city', '않': 'Not', '야': 'Hey', '약': 'about', 

                         '어': 'uh', '와': 'Wow', '용': 'for', '유': 'U', '을': 'of', '이': 'this',

                         '인': 'sign', '잘': 'well', '제': 'My', '쥐': 'rat', '지': 'G', '초': 'second',

                         '캐': 'Can', '탱': 'Tang', '트': 'The', '티': 'tea', '패': 'tile', '품': 'Width', 

                         '한': 'One', '합': 'synthesis', '해': 'year', '허': 'Huh', '화': 'anger', '황': 'sulfur',

                         '하': 'Ha', 'ﬁ': 'be', '０': '#', '２': '#', '８': '#', 'Ｅ': 'e', 'Ｇ': 'g',

                         'Ｈ': 'h', 'Ｍ': 'm', 'Ｎ': 'n', 'Ｏ': 'O', 'Ｓ': 's', 'Ｕ': 'U', 'Ｗ': 'w',

                         'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g',

                         'ｈ': 'h', 'ｉ': 'i', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o',

                         'ｒ': 'r', 'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｙ': 'y',

                         '𝐀': 'a', '𝐂': 'c', '𝐃': 'd', '𝐅': 'f', '𝐇': 'h', '𝐊': 'k', '𝐍': 'n', 

                         '𝐎': 'o', '𝐑': 'r', '𝐓': 't', '𝐔': 'u', '𝐘': 'y', '𝐙': 'z', '𝐚': 'a',

                         '𝐛': 'b', '𝐜': 'c', '𝐝': 'd', '𝐞': 'e', '𝐟': 'f', '𝐠': 'g', '𝐡': 'h', 

                         '𝐢': 'i', '𝐣': 'j', '𝐥': 'i', '𝐦': 'm', '𝐧': 'n', '𝐨': 'o', '𝐩': 'p',

                         '𝐪': 'q', '𝐫': 'r', '𝐬': 's', '𝐭': 't', '𝐮': 'u', '𝐯': 'v', '𝐰': 'w',

                         '𝐱': 'x', '𝐲': 'y', '𝐳': 'z', '𝑥': 'x', '𝑦': 'y', '𝑧': 'z', '𝑩': 'b',

                         '𝑪': 'c', '𝑫': 'd', '𝑬': 'e', '𝑭': 'f', '𝑮': 'g', '𝑯': 'h', '𝑰': 'i',

                         '𝑱': 'j', '𝑲': 'k', '𝑳': 'l', '𝑴': 'm', '𝑵': 'n', '𝑶': '0', '𝑷': 'p',

                         '𝑹': 'r', '𝑺': 's', '𝑻': 't', '𝑾': 'w', '𝒀': 'y', '𝒁': 'z', '𝒂': 'a',

                         '𝒃': 'b', '𝒄': 'c', '𝒅': 'd', '𝒆': 'e', '𝒇': 'f', '𝒈': 'g', '𝒉': 'h',

                         '𝒊': 'i', '𝒋': 'j', '𝒌': 'k', '𝒍': 'l', '𝒎': 'm', '𝒏': 'n', '𝒐': 'o', 

                         '𝒑': 'p', '𝒒': 'q', '𝒓': 'r', '𝒔': 's', '𝒕': 't', '𝒖': 'u', '𝒗': 'v', 

                         '𝒘': 'w', '𝒙': 'x', '𝒚': 'y', '𝒛': 'z', '𝒩': 'n', '𝒶': 'a', '𝒸': 'c',

                         '𝒽': 'h', '𝒾': 'i', '𝓀': 'k', '𝓁': 'l', '𝓃': 'n', '𝓅': 'p', '𝓇': 'r',

                         '𝓈': 's', '𝓉': 't', '𝓊': 'u', '𝓌': 'w', '𝓎': 'y', '𝓒': 'c', '𝓬': 'c',

                         '𝓮': 'e', '𝓲': 'i', '𝓴': 'k', '𝓵': 'l', '𝓻': 'r', '𝓼': 's', '𝓽': 't',

                         '𝓿': 'v', '𝕴': 'j', '𝕸': 'm', '𝕿': 'i', '𝖂': 'm', '𝖆': 'a', '𝖇': 'b',

                         '𝖈': 'c', '𝖉': 'd', '𝖊': 'e', '𝖋': 'f', '𝖌': 'g', '𝖍': 'h', '𝖎': 'i', 

                         '𝖒': 'm', '𝖓': 'n', '𝖕': 'p', '𝖗': 'r', '𝖘': 's', '𝖙': 't', '𝖚': 'u',

                         '𝖛': 'v', '𝖜': 'w', '𝖞': 'n', '𝖟': 'z', '𝗕': 'b', '𝗘': 'e', '𝗙': 'f',

                         '𝗞': 'k', '𝗟': 'l', '𝗠': 'm', '𝗢': 'o', '𝗤': 'q', '𝗦': 's', '𝗧': 't',

                         '𝗪': 'w', '𝗭': 'z', '𝗮': 'a', '𝗯': 'b', '𝗰': 'c', '𝗱': 'd', '𝗲': 'e',

                         '𝗳': 'f', '𝗴': 'g', '𝗵': 'h', '𝗶': 'i', '𝗷': 'j', '𝗸': 'k', '𝗹': 'i',

                         '𝗺': 'm', '𝗻': 'n', '𝗼': 'o', '𝗽': 'p', '𝗿': 'r', '𝘀': 's', '𝘁': 't',

                         '𝘂': 'u', '𝘃': 'v', '𝘄': 'w', '𝘅': 'x', '𝘆': 'y', '𝘇': 'z', '𝘐': 'l',

                         '𝘓': 'l', '𝘖': 'o', '𝘢': 'a', '𝘣': 'b', '𝘤': 'c', '𝘥': 'd', '𝘦': 'e',

                         '𝘧': 'f', '𝘨': 'g', '𝘩': 'h', '𝘪': 'i', '𝘫': 'j', '𝘬': 'k', '𝘮': 'm',

                         '𝘯': 'n', '𝘰': 'o', '𝘱': 'p', '𝘲': 'q', '𝘳': 'r', '𝘴': 's', '𝘵': 't',

                         '𝘶': 'u', '𝘷': 'v', '𝘸': 'w', '𝘹': 'x', '𝘺': 'y', '𝘼': 'a', '𝘽': 'b',

                         '𝘾': 'c', '𝘿': 'd', '𝙀': 'e', '𝙃': 'h', '𝙅': 'j', '𝙆': 'k', '𝙇': 'l', 

                         '𝙈': 'm', '𝙊': 'o', '𝙋': 'p', '𝙍': 'r', '𝙏': 't', '𝙒': 'w', '𝙔': 'y',

                         '𝙖': 'a', '𝙗': 'b', '𝙘': 'c', '𝙙': 'd', '𝙚': 'e', '𝙛': 'f', '𝙜': 'g',

                         '𝙝': 'h', '𝙞': 'i', '𝙟': 'j', '𝙠': 'k', '𝙢': 'm', '𝙣': 'n', '𝙤': 'o',

                         '𝙥': 'p', '𝙧': 'r', '𝙨': 's', '𝙩': 't', '𝙪': 'u', '𝙫': 'v', '𝙬': 'w',

                         '𝙭': 'x', '𝙮': 'y', '𝟎': '0', '𝟏': '1', '𝟐': '2', '𝟓': '5', '𝟔': '6',

                         '𝟖': '8', '𝟬': '0', '𝟭': '1', '𝟮': '2', '𝟯': '3', '𝟰': '4', '𝟱': '5',

                         '𝟲': '6', '𝟳': '7', '𝟑':'3', '𝟒':'4', '𝟕':'7', '𝟗':'9',

                         '🇦': 'a', '🇩': 'd', '🇪': 'e', '🇬': 'g', '🇮': 'i', 

                         '🇳': 'n', '🇴': 'o', '🇷': 'r', '🇹': 't', '🇼': 'w', '🖒': 'thumps up',

                         'ℏ':'h', 'ʲ':'j', 'Ｃ':'c', 'ĺ':'i', 'Ｊ':'j', 'ĸ':'k', 'Ｐ':'p'}













# List was cerated in separate notebook investigating on word embedding. 

# These dictionary is used to remove unwanted characters from the text

puncts =                 ['_','!', '?','\x08', '\n', '\x0b', '\r', '\x10', '\x13', '\x1f', ' ', ' # ', '"', '#', 

                         '# ', '$', '%', '&',  '(', ')', '*', '+', ',',  '/', '.', ':', ';', '<',

                         '=', '>', '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', '\x7f', '\x80',

                         '\x81', '\x85', '\x91', '\x92', '\x95', '\x96', '\x9c', '\x9d', '\x9f', '\xa0', 

                         '¡', '¢༼', '£', '¤', '¥', '§', '¨', '©', '«', '¬', '\xad', '¯', '°', '±', '³',

                         '¶', '·', '¸', 'º', '»', '¼', '½', '¾', '¿', '×', 'Ø', '÷', 'ø', 'Ƅ', 'ƽ',

                         'ǔ', 'Ȼ', 'ɜ', 'ɩ', 'ʃ', 'ʌ', 'ʻ', 'ʼ', 'ˈ', 'ˌ', 'ː', '˙', '˚', '́', '̄', '̅', 

                         '̇', '̈', '̣', '̨', '̯', '̱', '̲', '̶', '͜', '͝', '͞', '͟', '͡', 'ͦ', '؟', 'َ', 'ِ', 'ڡ', 

                         '۞', '۩', '܁', 'ा', '्', 'ા', 'ી', 'ુ', '๏', '๏̯͡', '༼', '༽', 'ᐃ', 'ᐣ', 'ᐦ', 'ᐧ',

                         'ᑎ', 'ᑭ', 'ᑯ', 'ᒧ', 'ᓀ', 'ᓂ', 'ᓃ', 'ᓇ', 'ᔭ', 'ᴦ', 'ᴨ', 'ᵻ', 'Ἰ', 'Ἱ', 'ὼ', 

                         '᾽', 'ῃ', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', 

                         '\u2007', '\u2008', '\u2009', '\u200a', '\u200b', '\u200c', '\u200d', '\u200e',

                         '\u200f', '‐', '‑', '‒', '–', '—', '―', '‖', '‘', '’', '‚', '‛', '“', '”', '„',

                         '†', '‡', '•', '‣', '…', '\u2028', '\u202a', '\u202c', '\u202d', '\u202f', '‰',

                         '′', '″', '‹', '›', '‿', '⁄', '⁍̴̛\u3000', '⁎', '⁴', '₂', '€', '₵', '₽', '℃', '℅',

                         'ℐ', '™', '℮', '⅓', '←', '↑', '→', '↓', '↳', '↴', '↺', '⇌', '⇒', '⇤', '∆', '∎',

                         '∏', '−', '∕', '∙', '√', '∞', '∩', '∴', '∵', '∼', '≈', '≠', '≤', '≥', '⊂', '⊕',

                         '⊘', '⋅', '⋆', '⌠', '⎌', '⏖', '─', '━', '┃', '┈', '┊', '┗', '┣', '┫', '┳', '╌', '═',

                         '║', '╔', '╗', '╚', '╣', '╦', '╩', '╪', '╭', '╭╮', '╮', '╯', '╰', '╱', '╲', '▀',

                         '▂', '▃', '▄', '▅', '▆', '▇', '█', '▊', '▋', '▏', '░', '▒', '▓', '▔', '▕', 

                         '▙', '■', '▪', '▬', '▰', '▱', '▲', '▷', '▸', '►', '▼', '▾', '◄', '◇', '○',

                         '●', '◐', '◔', '◕', '◝', '◞', '◡', '◦', '★', '☆', '☏', '☐', '☒', '☙', '☛',

                         '☜', '☞', '☭', '☻', '☼', '♦', '♩', '♪', '♫', '♬', '♭', '♲', '⚆', '⚭', '⚲', '✀',

                         '✓', '✘', '✞', '✧', '✬', '✭', '✰', '✾', '❆', '❧', '➤', '➥', '⠀', '⤏', '⦁',

                         '⩛', '⬭', '⬯', '\u3000', '、', '。', '《', '》', '「', '」', '〔', '・', 'ㄸ', 'ㅓ',

                         '锟', 'ꜥ', '\ue014', '\ue600', '\ue602', '\ue607', '\ue608', '\ue613', '\ue807',

                         '\uf005', '\uf020', '\uf04a', '\uf04c', '\uf070',  '\uf202\uf099', '\uf203',

                         '\uf071\uf03d\uf031\uf02f\uf032\uf028\uf070\uf02f\uf032\uf02d\uf061\uf029',

                         '\uf099', '\uf09a', '\uf0a7', '\uf0b7', '\uf0e0', '\uf10a', '\uf202', 

                         '\uf203\uf09a', '\uf222', '\uf222\ue608', '\uf410', '\uf410\ue600', '\uf469', 

                         '\uf469\ue607', '\uf818', '﴾', '﴾͡', '﴿', 'ﷻ', '\ufeff', '！', '％', '＇',

                         '（', '）', '，', '－', '．', '／', '：', '＞', '？', '＼', '｜', '￦', '￼', '�',

                         '𝒻', '𝕾', '𝖄', '𝖐', '𝖑', '𝖔', '𝗜', '𝘊', '𝘭', '𝙄', '𝙡', '𝝈', '🖑', '🖒']



 
def clean_numbers(x):

  

  """

  The following function is used to format the numbers.

  In the beginning "th, st, nd, rd" are removed

  """

  

  #remove "th" after a number

  matches = re.findall(r'\b\d+\s*th\b', x)

  if len(matches) != 0:

    x = re.sub(r'\s*th\b', " ", x)

    

  #remove "rd" after a number 

  matches = re.findall(r'\b\d+\s*rd\b', x)

  if len(matches) != 0:

    x = re.sub(r'\s*rd\b', " ", x)

  

  #remove "st" after a number

  matches = re.findall(r'\b\d+\s*st\b', x)

  if len(matches) != 0:

    x = re.sub(r'\s*st\b', " ", x)

    

  #remove "nd" after a number

  matches = re.findall(r'\b\d+\s*nd\b', x)

  if len(matches) != 0:

    x = re.sub(r'\s*nd\b', " ", x)

  

  # replace standalone numbers higher than 10 by #

  # this function does not touch numbers linked to words like "G-20"

  matches = re.findall(r'^\d+\s+|\s+\d+\s+|\s+\d+$', x)

  if len(matches) != 0:

    x = re.sub('^[0-9]{5,}\s+|\s+[0-9]{5,}\s+|\s+[0-9]{5,}$', ' ##### ', x)

    x = re.sub('^[0-9]{4}\s+|\s+[0-9]{4}\s+|\s+[0-9]{4}$', ' #### ', x)

    x = re.sub('^[0-9]{3}\s+|\s+[0-9]{3}\s+|\s+[0-9]{3}$', ' ### ', x)

    x = re.sub('^[0-9]{2}\s+|\s+[0-9]{2}\s+|\s+[0-9]{2}$', ' ## ', x)

    #we do include the range from 1 to 10 as all word-vectors include them

    #x = re.sub('[0-9]{1}', '#', x)

    

  return x
def year_and_hour(text):

  """

  This function is used to replace "yr,yrs" by year and "hr,hrs" by hour.

  """

  

  # Find matches for "yr", "yrs", "hr", "hrs"

  matches_year = re.findall(r'\b\d+\s*yr\b', text)

  matches_years = re.findall(r'\b\d+\s*yrs\b', text)

  matches_hour = re.findall(r'\b\d+\s*hr\b', text)

  matches_hours = re.findall(r'\b\d+\s*hrs\b', text)

  

  # replace all matches accordingly

  if len(matches_year) != 0:

    text = re.sub(r'\b\d+\s*yr\b', "year", text)

  if len(matches_years) != 0:

    text = re.sub(r'\b\d+\s*yrs\b', "year", text)

  if len(matches_hour) != 0:

    text = re.sub(r'\b\d+\s*hr\b', "hour", text)

  if len(matches_hours) != 0:

    text = re.sub(r'\b\d+\s*hrs\b', "hour", text)

  return text
def textBlobLemmatize(sentence):

  """

  This function uses the Word lemmatizer function of the textBlob package.

  """  

  #for each word in the text, replace the word by its lemmatized version

  for x in sentence.split():

    sentence = sentence.replace(x, Word(x).lemmatize())

  return sentence
def build_vocab(df):

  

  '''Build a dictionary of words and its number of occurences from the data frame'''

  

  #initialize the tokenizer

  tokenizer = TweetTokenizer()

  

  vocab = {}

  for i, row in enumerate(df):

      #tokenize the sentence 

      words = tokenizer.tokenize(row)

      #for each word, check if it is in the dict otherwise add a new entry

      for w in words:

       

        try:

            vocab[w] += 1

        except KeyError:

            vocab[w] = 1

  

  return vocab
#https://www.kaggle.com/christofhenkel/how-to-preprocessing-for-glove-part1-eda

def check_coverage(vocab,embeddings_index, print_oov_num=100):

  '''

  This function checks what part of the vocabluary and the text is covered by the embedding index.

  It returns a list of tuples of unknown words and its occuring frequency.

  '''

  

  a = {}

  oov = {}

  k = 0

  i = 0



  # for every word in vocab

  for word in vocab:

      # check if it can be found in the embedding

      try:

          # store the embedding index to a

          a[word] = embeddings_index[word]

          # count up by #of occurences in df

          k += vocab[word]

      except:

          # if no embedding for word, add to oov

          oov[word] = vocab[word]

          # # count up by #of occurences in df

          i += vocab[word]

          pass

  # calc percentage of #of found words by length of vocab

  print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

  # devide number of found words by number of all words from df

  print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))



  # return unknown words sorted by number of occurences

  sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

  print('Top unknown words are:', sorted_x[:print_oov_num])



  #return dict of unknown words + occurences

  return oov
def  load_embedding_vocab(path):

  '''

  Load the embeddings in the right format and return the vocab dictionary. 

  '''  

  # Print starting info about the pre-processing

  starttime = datetime.datetime.now().replace(microsecond=0)

  print("Starttime: ", starttime)



  def timediff(time):

    return time - starttime

  

  EMBEDDING_FILE = path

  def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

  embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE)) 

    

  time = datetime.datetime.now().replace(microsecond=0)

  print("Embedding model loaded and vocab returned. Time since start: ", timediff(time))

  

  #return the vocab

  return embeddings_index
def preprocessing_NN(df, model_vocab, calc_coverage=True, print_oov_num=100):

  """

  This function is only correcting words which are not out of the box known towards the embedding dictionary.

  It is optimized using the nltk TweetTokenizer.



  Function that combines the whole pre-processing process specifically for neural networks where less pre-processing is required compared to conventional methods.

  This means we will not remove stopwords, lemmatize or remove typical punctuation.

  """

  

  # Set parameters

  tokenizer = TweetTokenizer()

  

  # Print starting info about the pre-processing

  starttime = datetime.datetime.now().replace(microsecond=0)

  print('Dataset Length: ', len(df), "Starttime: ", starttime)



  def timediff(time):

    return time - starttime

  

  # build a vocabulary from the text 

  vocab = build_vocab(df.comment_text)

  print('Embedding vectors are loaded. \n')

  # check the coverage and receive a dictionary of unknown words

  unknown = check_coverage(vocab,model_vocab, print_oov_num=print_oov_num)

  # extract the list of unknown words

  unknown = unknown.keys()

  

  ## Process the unknown words

  # The replace_contractions function is applied on the data frame

  corrected = [replace_contractions(x) for x in unknown]

  time = datetime.datetime.now().replace(microsecond=0)

  print("Contractions have been replaced. Time since start: ", timediff(time))



  # Replace emojis with text

  corrected = [emoji.demojize(x) for x in corrected]

  time = datetime.datetime.now().replace(microsecond=0)

  print("Emojis have been converted to text. Time since start: ", timediff(time))



  # Replace keyboard smilies with text

  corrected = [replace_smilies(x) for x in corrected]

  time = datetime.datetime.now().replace(microsecond=0)

  print("Smilies have been converted to text. Time since start: ", timediff(time))



  # The clean_text function is applied on the data frame

  corrected = [clean_text(x) for x in corrected]

  time = datetime.datetime.now().replace(microsecond=0)

  print("All signs have been removed. Time since start: ", timediff(time))

  

  # The clean_numbers function is applied

  corrected = [clean_numbers(x) for x in corrected]

  time = datetime.datetime.now().replace(microsecond=0)

  print("All numbers have been replaced with ###. Time since start: ", timediff(time))

  

    # Replace or remove special characters like - / _ according to rules

  corrected = [replace_symbol_special(x, check_vocab=True, vocab=model_vocab) for x in corrected]

  time = datetime.datetime.now().replace(microsecond=0)

  print("Special symbols have been processed. Time since start: ", timediff(time))



  # Abbreviations are replaced by year and hour

  corrected = [year_and_hour(x) for x in corrected]

  time = datetime.datetime.now().replace(microsecond=0)

  print("Yr and hr have been replaced by year and hour. Time since start: ", timediff(time))

  

  # *Takes too long

  #Correct spelling mistakes

  #corrected = [TextBlob(x).correct() for x in corrected]

  #time = datetime.datetime.now().replace(microsecond=0)

  #print("Yr and hr have been replaced by year and hour. Time since start: ", timediff(time))

  

  #create a dictionary from word and correction

  dictionary = dict(zip(unknown, corrected))

  keys = dictionary.keys()

  

  #remove all keys where unknown equals correction after processing

  #create a new dict

  dict_mispell = dict()

  for key in dictionary.keys():

    # if the correction differs from the unknown word add it to the new dict

    if key != dictionary.get(key):

      dict_mispell[key] = dictionary.get(key)

  

  time = datetime.datetime.now().replace(microsecond=0)

  print('Correction dictionary of unknown words prepared. Time since start: ', timediff(time))

  #print(dict_mispell, '\n')

  

  def clean_mispell(text, dict_mispell):

    '''Replaces the unknown words in the text by its corrections.'''

    #tokenize the text with TweetTokenizer

    words = tokenizer.tokenize(text)

    for i, word in enumerate(words):

      # if the word is among the misspellings

      if word in dict_mispell.keys():

        #replace it by the corrected word

        words[i] = dict_mispell.get(word)

    #merge text by space

    text = ' '.join(words)

    # remove all double spaces potentially appearing after pre-processing.

    text  = re.sub(r' +', ' ', text)

    return text

      

  

  #tqdm.pandas()

  df.comment_text = df.comment_text.apply(lambda x: clean_mispell(x, dict_mispell))

  time = datetime.datetime.now().replace(microsecond=0)

  print('Unknown words replaced excluding coverage check. Time since start: ', timediff(time))

  

  # print the final result

  if calc_coverage == True: 

    vocab = build_vocab(df.comment_text)

    unknown = check_coverage(vocab,model_vocab, print_oov_num=print_oov_num)

    time = datetime.datetime.now().replace(microsecond=0)

    print('Pre-processing done including coverage check. Time since start: ', timediff(time))

  

  return df


model_vocab_glove = load_embedding_vocab('../input/glove840b300dtxt/glove.840B.300d.txt')
train_data = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

# kill all other columns except comment text

cols_to_keep = ['comment_text','target']

train_data = train_data.drop(train_data.columns.difference(cols_to_keep), axis=1)

gc.collect()
print('\n Glove \n')

df = preprocessing_NN(train_data,model_vocab_glove, calc_coverage=True, print_oov_num=100)  #if you set calc_coverage=False it reaches 14 min for whole train_set,-  however it does not show the coverage after processing
for i, x in enumerate(df.comment_text):

  print(x)

  if i==15:

    break 