from flask import Flask
from flask import render_template
from flask import request
import datetime
#from build_model import read_model, transform_tfidf, model_pkl_fname, vectorizer_pkl_fname 
import pickle
'''
ToDo:
                1. unicode character: copy and paste did not work
2. make the form larger
3. what happens in '\n' was entered in the form??


input:  


'''

app = Flask(__name__)

'''
to start this
python the_app.py
'''

my_title = '''<html>
        <head>
            <meta charset="utf-8">
            <title>Article Recommender </title>
        </head>
        '''

# OUR HOME PAGE
#============================================
@app.route('/')
#  http://localhost:6969/
def index():
    p_acronym = 'AllThingDS'
    pname_full = 'All Things Data Science '
    return render_template('index.html', pname_full = pname_full, p_acronym = p_acronym)


@app.route('/home0')
def home0():
    my_html = \
    '''
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>Article Classifier </title>
        </head>

      <body>
        <!-- page content -->
        <h1>New York Time Article Classifier</h1>
    '''
    timenow = "time is %s <br>" % datetime.datetime.now()
    more_stuff = \
    '''
        <hr>
        <div>
            This is an app to classify article based on its content.<br>
            <a href = 'form'> go to the form here </a> <br>
            good luck
        </div>
      </body>

    </html>
    '''
    return my_html + timenow + more_stuff


# Form page to submit text
#============================================
# create page with a form on it
@app.route('/form')
#  http://localhost:6969/form
def cool_form():
    action = '/a_classifier'
    return render_template('form.html', action = action)


@app.route('/form0')
def submission_page():
    return '''

    Enter text here< br> 

    <form action="/a_classifier" method='POST' >
        <input type="text" name="user_input" />
        <input type="submit" />
    </form>
    '''

'''
MoMA bought another painting by Monet.  It was the most expensive art acquired. 

'''
# My word counter app
#==============================================
# create the page the form goes to
@app.route('/a_classifier', methods=['POST'] )

def a_classifier():
    '''
    article classifier
    '''
    # get data from request form, the key is the name you set in your form
    data = request.form['user_input']

    # convert data from unicode to string
    #data = data.decode('utf-8')

    # run a simple program that counts all the words
    total_words = len(data.split(' '))

    vectorizer2, clf2 = read_model()
    x_vectorized = transform_tfidf(vectorizer2, [data])
    y_pred = clf2.predict(x_vectorized[0])[0]
    output = 'predict: %s ' % (y_pred)

    return render_template('result.html', y_pred = y_pred, data = data)
    '''
    # now return your results
    my_html = my_title + '<b>Total words</b> is %i' % ( total_words)
    my_html = my_html + '<br>' + output 
    my_html = my_html + "<br>  <a href = 'form'> return to form </a> <br>  "
    return my_html
    '''
@app.route('/recommender')

def recommender():
    username = 'Joyce'
    with open('../code/eda/df_recom.pkl','r') as f:
        df_recom = pickle.load(f)
    with open('../code/eda/relevant_all.pkl','r') as f:
        relevant_all = pickle.load(f)
    return render_template('recom.html', username = username, idx = range(df_recom.shape[0]), \
        df_recom = df_recom, relevant_all=relevant_all)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6969, debug=True)
