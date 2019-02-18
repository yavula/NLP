#!/usr/bin/env python
# coding: utf-8

# In[26]:


import nltk
from bs4 import BeautifulSoup 


# In[27]:


puzzle_Words='EGBDAFKJLMORCNST'
required_Word='M'
letters=nltk.FreqDist(puzzle_Words.lower())
word_Corpus=nltk.corpus.words.words()
words_Output=[w for w in word_Corpus if len(w) >=6 and required_Word.lower()
                in w and nltk.FreqDist(w) <= letters]
len(words_Output)


# In[28]:


from nltk.corpus import PlaintextCorpusReader
mycorpus = PlaintextCorpusReader('.', '.*\.txt')


# In[29]:


corpus_String = mycorpus.raw('HW_WikipediaDiscussions.txt')


# In[30]:


soup = BeautifulSoup(corpus_String, 'lxml')


# In[31]:


parsed_text = soup.get_text()
len(parsed_text)


# In[32]:


corpus_Words=nltk.word_tokenize(parsed_text)


# In[33]:


len(corpus_Words)


# In[36]:


word_Corpus = [w.lower( ) for w in corpus_Words]


# In[37]:


len(word_Corpus)


# In[38]:


import re


# In[39]:


def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False


# In[42]:


alphaword_Corpus=[w for w in word_Corpus if not alpha_filter(w)]
alphaword_Corpus[1:30]


# In[50]:


stopwords = nltk.corpus.stopwords.words('english')


# In[57]:


stoppedCorpus=[w for w in alphaword_Corpus if not w in stopwords]
stoppedCorpus[1:10]


# In[58]:


fdist=nltk.FreqDist(stoppedCorpus)


# In[80]:


fdistList=fdist.most_common(50)
print('Below are 50 words by frequency')
for item in fdistList:
    print(item[0],item[1])


# In[61]:


stopwords.extend(("/a","wp","/b","'s","b","n't","p","/i","dd","/dd","br","q=","/li","rs","li","'m","afd"))


# In[63]:


stoppedCorpus=[w for w in alphaword_Corpus if not w in stopwords]
stoppedCorpus[1:20]


# In[64]:


fdist=nltk.FreqDist(stoppedCorpus)
fdistList=fdist.most_common(50)
for item in fdistList:
    print(item[0],item[1])


# In[93]:


from nltk.collocations import *


# In[94]:


bigram_Corpus = nltk.collocations.BigramAssocMeasures()


# In[95]:


finderA = BigramCollocationFinder.from_words(alphaword_Corpus)


# In[96]:


scoredWords = finderA.score_ngrams(bigram_Corpus.raw_freq)
print('Below are 50 bigrams without applying stop word filter')
for b in scoredWords[:50]:
    print(b)


# In[97]:


print('Below are top 50 Bigrams by frequency after applying stop word filter, and less than equal to 3 filter')
finderA.apply_ngram_filter(lambda w1, w2: len(w1) < 4)
scored = finderA.score_ngrams(bigram_Corpus.raw_freq)
for bscore in scored[:50]:
    print (bscore)


# In[98]:


print('Below are 50 bigrams by Mutual Information Scores with minimum frequency 5')
finderB = BigramCollocationFinder.from_words(stoppedCorpus)
finderB.apply_freq_filter(5)
scoredWords = finderB.score_ngrams(bigram_Corpus.pmi)
for b in scoredWords[:50]:
    print (b)


# In[ ]:





# In[ ]:




