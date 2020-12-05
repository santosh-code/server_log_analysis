# server_log_analysis

# Importing the required Libraries
import pandas as pd 
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.util import ngrams
from wordcloud import WordCloud, STOPWORDS
import re
from sklearn.pipeline import Pipeline
import nltk, re, string, collections
import matplotlib.pyplot as plt
import seaborn as sns


# Load the  dataset

df = pd.read_csv("C:/Users/USER/Desktop/SQL_DATA_1.csv", encoding='cp1252')

# Extracting the required features
logDfObj = pd.DataFrame(df, columns=['Date','Message','Event','Severity'])
# extracting Date
logDfObj['Date'] = logDfObj['Date'].str.extract(r'^(\d{2}/\d{2}/\d{4})', expand=False)

# Removing the entries with nan 
df1 = logDfObj[logDfObj['Event'].notna()]

#df1['Message'].isna().value_counts()
df1.loc[df1.Severity == 'Warning', 'Severity'].count()
df1.loc[df1.Severity == 'Error', 'Severity'].count()
df1.loc[df1.Severity == 'Information', 'Severity'].count()
df1.loc[df1.Severity == 'Unknown', 'Severity'].count()

# Severity Vs Event

sns.catplot(x="Severity",y="Event",data=df1)
plt.xlabel("Severity")
plt.ylabel("Event")
plt.title('Event Vs Severity')

df1_unknown= df1.loc[df1['Severity'] == 'Unknown']
df1_unknown['Event'].unique()
df1_unknown.loc[df1_unknown.Event == 903,"Event"].count()
df1_unknown.loc[df1_unknown.Event == 258,"Event"].count()
df1_unknown.loc[df1_unknown.Event,'Event'].count()

df2_error= df1.loc[df1['Severity'] == 'Error']
df3_warning= df1.loc[df1['Severity'] == 'Warning']
df4_info= df1.loc[df1['Severity'] == 'Information']



# categorizing the errors based on severity/ Event ID
BigramFreq = collections.Counter(df2_error['Event'])
BigramFreq
l = BigramFreq.most_common(10)
l

        
df2_error['Level'] = np.where(
    df2_error['Event'] == 100 ,'Sev 2', np.where(
    df2_error['Event'] == 455,'Sev 3',np.where(
    df2_error['Event'] == 1002,'Sev 2',np.where(
    df2_error['Event'] == 15,'Sev 3',np.where(
    df2_error['Event'] == 10010,'Sev 3',np.where(
    df2_error['Event'] == 4199,'Sev 2',np.where(
    df2_error['Event'] == 1000,'Sev 1',np.where(
    df2_error['Event'] == 7011,'Sev 2',np.where(
    df2_error['Event'] == 8189,'Sev 1','Sev 4')))))))))

df2_error['Solution'] = np.where(
    df2_error['Level'] =='Sev 2','Resart your machine/application', np.where(
    df2_error['Level'] == 'Sev 3','Reach out to system admin',np.where(
    df2_error['Level'] == 'Sev 1','Reach out to Network Engineer',np.where(
    df2_error['Level'] == 'Sev 4','Reach out to system admin','Reach out to system admin'))))




#word_cloud

import matplotlib.pyplot as plt
cate_grp = df2_error.groupby('Level').Event.sum()
cat = cate_grp.reset_index()
cat

plt.pie(cat['Event'],labels=cat['Level'],autopct='%1.1f%%')
plt.title('Event ID vs Severity')

file=df2_error['Message'].values.tolist()
def unigram(file,n):
    text = " ".join(file)
    CleanedText = re.sub(r'[^a-zA-Z]'," ",text)
    CleanedText = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(CleanedText) if word not in stopwords.words("english") and len(word) > 3])
    return CleanedText


CleanedText = unigram(file,1)
CleanedText
tokens = nltk.tokenize.word_tokenize(CleanedText)
cts=nltk.FreqDist(tokens)
cts.plot(20)




wordcloud = WordCloud(random_state=21).generate(CleanedText)
plt.figure(figsize = (30,15))
plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()
#bi-gram wordcloud

def ngrams(file,n):
    text = " ".join(file)
    text1 = text.lower()
    text2 = re.sub("[^A-Za-z" "]+"," ",str(text1))
    text3 = " ".join([WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text2) if word not in stopwords.words("english") and len(word) > 2])
    words = nltk.word_tokenize(text3)
    ngram = list(nltk.ngrams(words,n))
    return ngram

ngram = ngrams(file,2)
ngram[1:10]

for i in range(0,len(ngram)):
    ngram[i] = "_".join(ngram[i])

Bigram_Freq = nltk.FreqDist(ngram)
Bigram_Freq.plot(20)

bigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Bigram_Freq)
plt.figure(figsize = (1800,1400))
plt.imshow(bigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()

#tri-gram wordcloud
ngram = ngrams(file,3)

for i in range(0,len(ngram)):
    ngram[i] = "_".join(ngram[i])
Trigram_Freq = nltk.FreqDist(ngram)

trigram_wordcloud = WordCloud(random_state = 21).generate_from_frequencies(Trigram_Freq)
plt.figure(figsize = (50,25))
plt.imshow(trigram_wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()
#histogram
import seaborn as sns
import matplotlib.pyplot as plt
# Data Exploration
plt.bar(df2_error.Level)
plt.bar(height = df2_error, x = df2_error.Level)

df2_error['Level'].value_counts()
plt.hist(df2_error.Level, color ='green') #histogram
plt.title('Histogram For Severity')
plt.xlabel('Severity')
plt.ylabel('Frequency')
sns.barplot(x='Event', y='Level', data=df2_error)
#apriori alogrithm
from apyori import apriori
 
#Here we need a data in form of list for Apriori Algorithm.
#df1_seq = df1_seq.iloc[:,[2,4]]
records = []
for i in range(1, 7962):
    records.append([str(df1_seq.values[i, j]) for j in range(0, 5)])

#records1 = []
#for i in range(1, 7962):
    #records1.append([str(df1_seq.values[i, j]) for j in range(0, 2)])
    
    
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)   

print("There are {} Relation derived.".format(len(association_results)))

for i in range(0, len(association_results)):
    print(association_results[i][0])
    
for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    # second index of the inner list
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
