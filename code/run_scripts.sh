work flow:
1.  get links
2. get articles
3. add body_text


------------------------ get links (urls ) of articles from various sources

--- feedly
ipython my_html_links_scraper.py

--- data science weekly
ipython dsw_scraper.py


---- datatau
ipython get_links.py   or get_html.py   (i dont know why the 2 different files )

---- kdnugget monthly newsletter
ipython kdnugget_monthly_scraper.py

----scrape way back
ipython Wayback.py

---process slack messages
cd scraper
ipython process_pdf.py
    fname = '/Users/joyceduan/data/slack/06-04.pdf'

---------  mongo db admin stuff
update_alllinks.py

------------------------  get acutal raw_html contents
cd scraper
ipython get_article.py


------------------------ get body_text 
cd preprocess
ipython add_body_text_mongo.py

------------------------ topic modeling
cd model
topic_modeling.py


------------------------ make recommendations
cd recommender
ipython recommender.py 05-19_2.txt > 05-19_2-out.txt

