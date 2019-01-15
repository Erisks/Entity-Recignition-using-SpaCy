#!/usr/bin/env python
# coding: utf-8

# In[9]:


#entity
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pprint
#nlp(language object)=language model used - english
nlp = en_core_web_sm.load() 


# In[10]:


#we only need to apply the model(nlp)once, the entire background pipeline will return the objects
# returns a processed doc
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print([(X.text, X.label_) for X in doc.ents])
#doc.ents -> named entities recognized


# In[11]:


print([(X, X.ent_iob_, X.ent_type_) for X in doc])
##(text, tag(BILUO tagging), entitytype)
##"B" means the token begins an entity, "I" means it is inside an entity, "O" means it is outside an entity, and "" means no entity tag is set.


# In[12]:


from bs4 import BeautifulSoup
import requests
import re
def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
article = nlp(ny_bb)
len(article.ents)


# In[13]:


#entities found and labels
labels = [x.label_ for x in article.ents]
Counter(labels)


# In[14]:


#frequency of tokens and the most common 
items = [x.text for x in article.ents]
Counter(items).most_common(3)


# In[15]:


#selection of random sentence 
sentences = [x for x in article.sents]
print(sentences[20])


# In[16]:


#generate raw markup with displaCy visualizer
displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')


# In[17]:


# displacy visualizer
displacy.render(nlp(str(sentences[20])), style='dep', jupyter = True, options = {'distance': 120})


# In[18]:


#verbatim, extract part-of-speech and lemmatize this sentence.
[(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in nlp(str(sentences[20])) 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]


# In[19]:


dict([(str(x), x.label_) for x in nlp(str(sentences[20])).ents])


# Named entity extraction are correct except “F.B.I”.

# In[20]:


print([(x, x.ent_iob_, x.ent_type_) for x in sentences[20]])


# In[21]:


#visualize whole article
displacy.render(article, jupyter=True, style='ent')


# In[ ]:





# In[ ]:




