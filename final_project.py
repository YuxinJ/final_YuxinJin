#final project 
#course: computation for public policy 
#project topic: from a western perspective: China and the Environment 

#import the NYTarticles API 
from nytimesarticle import articleAPI
api = articleAPI('22bcd777b40f8d77e6ccf6469e7f9f16:11:67754634')

#test api searches 
search = api.search (q= 'environment' #put the more important keyword here. 
                                      #need to filter again how does the work 'environment' is used
                     ,fq = {'headline':'China'
                            #'subline':'environment'
                            ,'body':['China']
                            ,'source':['Reuters','AP', 'The New York Times']}
                     ,sort='oldest'
                     ,begin_date = 19900101
                     ,end_date = 20151231
                     )  


#defining a function to parse the result return from API 
def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    for i in articles['response']['docs']:
        dic = {}
        dic['id'] = i['_id']
        if i['abstract'] is not None:
            dic['abstract'] = i['abstract'].encode("utf8")
        dic['headline'] = i['headline']['main'].encode("utf8")
        dic['desk'] = i['news_desk']
        dic['date'] = i['pub_date'][0:10] # cutting time of day.
        dic['section'] = i['section_name']
        if i['snippet'] is not None:
            dic['snippet'] = i['snippet'].encode("utf8")
        dic['source'] = i['source']
        dic['type'] = i['type_of_material']
        dic['url'] = i['web_url']
        dic['word_count'] = i['word_count']
        # locations
        locations = []
        for x in range(0,len(i['keywords'])):
            if 'glocations' in i['keywords'][x]['name']:
                locations.append(i['keywords'][x]['value'])
        dic['locations'] = locations
        # subject
        subjects = []
        for x in range(0,len(i['keywords'])):
            if 'subject' in i['keywords'][x]['name']:
                subjects.append(i['keywords'][x]['value'])
        dic['subjects'] = subjects   
        news.append(dic)
    return(news)

def get_articles(date,query):
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'environment') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
   all_articles = []
    for i in range(0,100): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        articles = api.search(q = query,
               fq = {'headline':['Beijing','China','Shanghai']
                     ,'source':['Reuters','The New York Times']},
               begin_date = date + '0101',
               end_date = date + '1231',
               sort='oldest',
               page = str(i))
        articles = parse_articles(articles)
        all_articles = all_articles + articles
    return(all_articles)

#query = environment; collecting the result into a dictionary 
q_environment = []
for i in range(1990,2016):
    try:
        print 'Processing' + str(i) + '...'
        China_year =  get_articles(str(i),'environment')
        q_environment = q_environment + China_year
    except:
        pass

#query = pollution; collecting the result into a dictionary 
q_pollution = []
for i in range(1990,2016):
    try:
        print 'Processing' + str(i) + '...'
        China_year =  get_articles(str(i),'pollution')
        q_pollution = q_pollution + China_year
    except:
        pass

#save the result dictionaries into CVS files
import csv
keys = q_environment[48].keys()
with open('q_environment_china_headline.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader
    dict_writer.writerows(China_all)

with open('q_pollution_chinabeijingshanghai_headline.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader
    dict_writer.writerows(q_pollution)

#merge q_environment and q_pollution together, then delete the repetitive items
import pandas as pd
q_environment = pd.read_csv('q_environment_chinabeijingshanghai_headline.csv')
q_all = q_environment.append(q_pollution)
#check if tere are repetitive items in the list 
len(q_all['url']) != len(set(q_all['url']))
q_all_norepeat = q_all.drop_duplicates(cols='url', take_last=True)
q_all_norepeat.to_csv('q_all_norepeat.csv')
q_all = pd.read_csv('q_all_norepeat.csv')
def get_headline(df):
    for i in range(0,len(df)):
        print i #need to return the index as well 
        print df['headline'][i]
get_heandlin(q_all) # mannually filter all the result 

#get the filtered article set
#extracted the filtered article entries (selected by hand)
L1=[1,26,37,38,44,47,58,72,86,92,98,109,111,114,118,119,120,141,159,161,186,190,198,199,207,215,219
      ,220,221,223,224,227,233,236,237,241,242,247,249,251,256,260,261,263,267,270,277,278,286,292,296
      ,301,314,317,323,329,330,334,337,344,347,354,358,364,370,371,382,398,429,433,449,454,475,493,496
      ,497,504,505,509,520,524,525,527,540,548,553,557,564,565,574,620,623,624,627,629,630,631,634,640
      ,657,659,669,675,676,686,689,695,703,717,731,732,737,740,762,778,783,785,800,817,828,833,841,843
      ,849,851,852,853,858,860,861,867,872,874,878,897,881,891,893,906,924,928,944,948,951,954,956,957
      ,958,960,961,962,963,969,970,971,972,973,975,976,977,980,981,982,984,986,987,988,995,996,1000,1003
      ,1004,1007,1008,1009,1011,1012,1014,1017,1019,1021,1023,1024,1027,1029,1030,1031,1038,1039,1040,1041
      ,1042,1043,1044,1045,1047,1048,1059,1061,1064,1065,1069,1071,1072,1074,1076,1079,1081,1082,1083,1084
      ,1085,1088,1098,1102,1103,1104,1105,1107,1109,1112,1116,1117,1118,1126,1127,1131,1132,1134,1135,1137
      ,1138,1139,1143,1144,1148,1150,1151,1152,1153,1154,1155,1157,1158,1159,1160,1162,1163,1164,1167,1169
      ,1170,1171,1177,1178,1179,1180,1182,1183,1187,1188,1194,1196,1197,1198,1204,1208,1209,1210,1211,1213
      ,1214,1216,1218,1219,1221,1224,1225,1226,1229,1234,1235,1236,1237,1242,1243,1249,1250,1251,1252,1253
      ,1256,1257,1258,1261,1262,1263,1264,1265,1267,1268,1270,1271,1273,1275]
q_all_filtered = q_all.iloc[L1]
q_all_filtered.to_csv('q_all_filtered.csv')


#now need to scrape the articles, save as txt. 
#show the environment in context 

#start with the sracping txt part 
import requests 
from bs4 import BeautifulSoup
from urllib2 import urlopen
import re
import pandas as pd
import time 

China_all = pd.read_csv("q_all_filtered_alter.csv")

#need a function to soup all the articles and extract the article body 
#define a function: input url, output article body in text 
def get_soup(url):
	index_html = urlopen(url)
	index = BeautifulSoup(index_html,'lxml')
	return(index) 

def get_articlebody(url):
    length = len(get_soup(url).find_all('p',{"itemprop":"articleBody"}))
    articlebody = ""
    for i in range(length):
        articlesoup = get_soup(url)
        articlebody = articlebody + articlesoup.find_all('p',{"itemprop": "articleBody"})[i].get_text().strip()
    return(articlebody)

#write articlebody into the dataframe
for i in range(0,20): 
    China_all['body'][i] = get_articlebody(China_all['url'][i])

#define a function to get all the context of the word 'environment' 
import nltk 
from nltk import word_tokenize
def get_environment_context(df):
    for i in range(0,20):
        print i #need to return the index as well 
        tokens = word_tokenize(df['body'][i].encode('ascii','ignore'))
        text = nltk.Text(tokens)
        context_environment = text.concordance("environment")
        context_emission = text.concordance("emission")
    return(context_environment, context_emission) 
#the above function failed to work when dealing with cookies, thus defining another function to handle the cookies
import urllib2
def get_soup2(url):
    index_html = urllib2.build_opener(urllib2.HTTPCookieProcessor).open(url)
    index = BeautifulSoup(index_html,'lxml') #index of the opened url 
    return(index)

def get_articlebody2(url):
    length = len(get_soup2(url).find_all('p',{"itemprop":"articleBody"}))
    articlebody = ""
    for i in range(length):
        articlesoup = get_soup2(url)
        articlebody = articlebody + articlesoup.find_all('p',{"itemprop": "articleBody"})[i].get_text().strip()
    return(articlebody)

#get all article bodies 
for i in range(0,len(China_all['url'])+1): 
    print "processing" + "line" + str(i)
    try:
        China_all['body'][i] = get_articlebody2(China_all['url'][i])
    except:
        pass
#encode the article body text 
for i in range(200,306):
    try:
        China_all['body'][i]= China_all['body'][i].encode('ascii','ignore')
    except:
        pass
China_all.to_csv('q_all_filtered_body_all.csv')

#text processing 
import nltk 
from nltk import word_tokenize
import pandas as pd 
import string
bodyfiltered = pd.read_csv("q_all_filtered_body_all.csv")

#define a function that tokenize, lower, nonpunctuation and filters stop words for each article body
def token_clean(articlebody):
    articlebody_lower = articlebody.lower()
    articlebody_nonpunct = articlebody_lower.translate(None,string.punctuation)
    tokens = nltk.word_tokenize(articlebody_nonpunct)
    articlebody_cleantoken = [w for w in tokens if not w in stopwords.words('english')]
    return (articlebody_cleantoken)

import math
from __future__ import division
from collections import Counter
from nltk.corpus import stopwords

#defining a function that returns TF-IDF for each document of a certain term 
#this function can be applied to individual clean_token text 
def get_df(word):
    count = 0
    for i in range(0,306):
        try:
            if str(word) in bodyfiltered['tokenbody'][i]:
                count += 1
        except:
            pass
    return (count)

def tfidf(clean_token_article,word):
    #tf 
    counter = Counter(clean_token_article)
    tf = counter[word]
    #df #of documents containing word 
    df = get_df(word)
    #IDF(t) = log ( #{documents} / #{documents containing t } )
    idf = math.log(305/df)
    #TF-IDF(d, t) = TF(d, t) * IDF(t)
    tfidf = float(tf * idf)
    return (tfidf)

#add tokenized article body in to the dataframe
for i in range(0,len(bodyfiltered['body'])+1):
    try:
        bodyfiltered['tokenbody'][i] = token_clean(bodyfiltered['body'][i])
    except:
        pass

bodyfiltered.to_csv("q_all_filtered_body_all.csv")
bodyfiltered = pd.read_csv("q_all_filtered_body_all.csv")

#have tfidf written into the dataframe
bodyfiltered['tfidf_china'] = ''
for i in range(0,306):
    try:
       bodyfiltered['tfidf_china'][i] = tfidf(bodyfiltered['tokenbody'][i],'china')
    except:
        pass 

bodyfiltered['tfidf_pollution'] = ''
for i in range(0,306):
    try:
       bodyfiltered['tfidf_pollution'][i] = tfidf(bodyfiltered['tokenbody'][i],'pollution')
    except:
        pass  

#plot average tfidf by year
tfidfyear_china = bodyfiltered.groupby(bodyfiltered['date'].map(lambda x: x.year))
tfidfyear_china_sum = tfidfyear_china['tfidf_china'].agg('sum')
tfidfyear_china_mean = tfidfyear_china['tfidf_china'].agg('mean')

#to get a line graph for the accumulated TFIDF score (china) by year
tfidfyear_chinasum = pd.DataFrame({'TFIDF_china': tfidfyear_china_sum})

%matplotlib inline
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))

x=tfidfyear_chinasum.index
y=tfidfyear_chinasum[[0]]
plt.plot(x,y)
fig.suptitle("Total TFIDF Score for 'china' Artical Corpus", fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('TFIDF', fontsize=14)

#draw a plot of two y axes 
fig = plt.figure(figsize=(8,4))
fig, ax1 = plt.subplots()
x=tfidfyear_chinamean.index
y1=tfidfyear_chinamean[[0]]
y2=tfidfyear_chinasum[[0]]
ax1.plot(x, y1, 'b-')
ax1.set_xlabel('Year')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('average tfidf score', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('sum tfidf score', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

#########
#ploting the aqi data 
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt
aqi = pd.read_csv('aqi_data_nocity.csv')

for i in range(len(aqi)):
    aqi['date'][i] = datetime.strptime(aqi['date'][i], '%Y/%m/%d').date()
# extract historical data for Beijing by city_code 
aqi_beijing = aqi.loc[aqi['city_code'] == 110000]
aqi_beijing.to_csv('aqi_beijing_day.csv')
#aggregate the data by month in excel... 
aqi_beijing_month = pd.read_csv('aqi_beijing_month.csv')

#plotting a time series graph for the historical data of Beijing AQI
x1 = aqi_beijing['date']
y1 = aqi_beijing['value']
x2 = aqi_beijing_month['date']
y2 = aqi_beijing_month['value']

plt.figure(figsize=(37,15)) #set size of the plotting area 

plt.title('Historical AQI Trend of Beijing')
plt.xlabel('Date')
plt.ylabel('AQI Index')
plt.grid(alpha=0.4)

plt.ylim([0,520])

plt.plot(x1,y1,label='daily measure',color='lightsalmon')
plt.plot(x2,y2,label='monthly average',color='gray')
plt.legend()
plt.subplots_adjust(left=0.15)
plt.show()