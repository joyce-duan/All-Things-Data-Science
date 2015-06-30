# [All-Things-Data-Science] (http://www.allthingsds.com/): Article Finder

### Joyce Duan

## Overview
~4000 data science related articles were scraped from over 800 websites. NMF was used for topic modeling.  A web app was implemented that allows users to browse the topics and search articles.   

##Motivation

With the rapid advancement of technology, it is pertinent for data scientists to keep
updated with the latest developments in methodology and business applications. Such
information is abundant, but currently fragmented, existing in various blogs, articles from
linkedIn influencer articles, newspapers, and professional organization website. In this
project, I used text analysis to model topics related to analytics and
data science. The collection of articles enabled exploratory analysis on trending of
topics. The web app provides a tool to organize these articles and allows users to search for articles of interst. The workflow developed in this project is also applicable to personalized content recommendations for other professions or personal interest areas.

##Getting the Data

The links to articles were collected from various sources including DataTau, weekly newsletters, collections of blogs as recommended on Quora. Articles published from Dec 2013 and June 2014 were scraped from the corresponding source website.

##Topic Modeling

The pipeline includes the following steps:
1. extract body text using boilerpipe
2. clean text by removing url links 
3. tokenziation converts a document into a list of 1, 2, 3-grams.
4. stemming is applied to reduce inflected or derived words to their base forms. This reduces dimension of the TF-IDF matrix. It also enables search using alternative forms of words.
5. TF-IDF: the collection documents are represented by a matrix of TFIDF features the code term frequency.
6. NMF: the TF-IDF matrix is approximatd by the product of 2 matrices: document-topic weighting, and topic-words weighting.

I tested out differnt tokenizers, different stemmers, n-gram ranging from 1 to 5, and number of topics from 20 to 40. The results were manually reviewed to check if topics were distinct and articles with highest weights under each topic share similar contents. The combination with the most intuitive and sensible results were used in the final model.

##Final Output
[The web app] (http://www.allthingsds.com/), allows users to browse topics, search for articles, and explore visualization of trends by each topic.

##Next Steps
* Add daily feed of newly published articles
* Add feature that allows user to rate articles
* Build personalized recommendation engine using user rating data
