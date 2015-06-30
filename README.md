# [All-Things-Data-Science] (http://www.allthingsds.com/): Article Finder

### Joyce Duan

## Overview
~4000 data science related articles were scraped from over 800 websites. NMF was used for topic modeling.  A web app was implemented that allow users to browse the topics and search articles.   

##Motivation

With the rapid advancement of technology, it is pertinent for data scientists to keep
updated with the latest developments in methodology and business applications. Such
information is abundant, but currently fragmented, existing in various blogs, articles from
linkedIn influencer articles, newspapers, and professional organization website. In this
project, I used text analysis to model topics related to analytics and
data science. The collection of articles enabled exploratory analysis on trending of
topics. The web app provides a tool to organize there articles and allow users to search for articles of interst. The workflow developed in this project is also applicable to personalized content recommendations for other professions or personal interest areas.

##Getting the Data

The links to articles were collected from various sources including DataTau, weekly newsletters, collections of blogs recommended on Quora. Articles published from Dec 2013 and June 2014 were scraped from the corresponding source website.

##Topic Modeling
use TFIDF
vectorization and nonnegative
matrix factorization on a corpus of ~500
articles to discover topics covered in these articles.

##Final Output
[The web app] (http://www.allthingsds.com/), allows users to browse topics, search for articles, and explore visualization of trends by each topic.

##Next Steps
* Add daily feed of articles
* Add feature that allow user to rate articles
* Build personalized recommendation engine using user rating data
