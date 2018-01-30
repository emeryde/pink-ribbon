from config_param import config

import pandas as pd
import numpy as np

from nlp_utils import english_stemmer, stopwords, get_text_intensity, get_text_polarity, get_text_subjectivity
#import nltk
#nltk.download()
df = pd.read_csv(config.original_data_path)

print(df.head())
##Step 1: configure dataset so that dataset appears as id, date, sentence instead of id, date, post
print('\nTransforming Dataset...')
from nltk.tokenize import sent_tokenize
res = pd.DataFrame()
for i in range(df.shape[0]):
    for sentence in sent_tokenize(df.Post.values[i]):
        tmp = pd.DataFrame({'ID': df['ID'][i], 'Date': df['Date'][i], 'Sentence': [sentence]})
        res = res.append(tmp)

df = res
##Step 2: determine if the word "pink" is in a sentence
print('\nGenerating binary "pink" variable')
def binaryPink(post):
    if ('pink' in post.lower().split()):
        return 1
    else:
        return 0

df['bin_pink'] = df['Sentence'].apply(lambda x: binaryPink(x))

print('\nTotal percentage of post with "pink" in text: {}%'.format(str(100*np.mean(df.bin_pink))))
##We can see that roughtly 10% of the blog post have the word pink in them
##Step 2: Conduct word stemming::edit - current project does not need word stemming
'''
from nlp_utils import stem_tokens
token_pattern = r"(?u)\b\w\w+\b"
import re
def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=config.cooccurrence_word_exclude_stopword,
                    encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE)
    ## tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed

df["post_unigram"] = list(df.apply(lambda x: preprocess_data(x["Post"]), axis=1))
print(df.head())
'''

##Three components of textblob sentiment analysis
print('\nGenerating Sentiment Features...')
df['polarity'] = list(df.apply(lambda x: get_text_polarity(x["Sentence"]), axis = 1))
df['subjectivity'] = list(df.apply(lambda x: get_text_subjectivity(x["Sentence"]), axis = 1))
df['intensity'] = list(df.apply(lambda x: get_text_intensity(x["Sentence"]), axis = 1))

##Polarity refers to how positive or negeative text is [-1,1]
##Subjectivity refers to how opinionated text is [0,1]
##Intensity handles negation terms such as 'not great' or 'not bad' to make them the correct polarity [x0.5, x2.0]4

##Side note: intensity will be excluded from initial processing
print('\nExporting to csv...')
df.to_csv(config.processed_data_path, index = False, header = True)



#Site of the image: basically the creation of the ribbon
#Site of the audience: What does it mean to people 20+ years later
#Finding a mixed reaction in the literature
#Think before you pink
#Controversy over companies putting pink ribbon over products
#Number of issues over the campaign
#Pinkwashing: Using the color pink to make profits
#Conflict of interest occur in certain products

##The symbolic meaning of the pink ribbon changing
##Some people are against the symbol because they feel people are

##What is the perception of the pink ribbon nowadays
##What language is persuasive to different audiences
##Is it changing their behavior -- number of breast cancers checks done every year

##Establish a baseline for each person -- find out sentiment to pink ribbon normalized to personality
##Each individual blogger: segment into pink groups and non-pink groups
##Establish baseline score of each person
##T-test comparison of person's normalized feelings towards the pink ribbon
##Determine statistical significance of difference, if difference