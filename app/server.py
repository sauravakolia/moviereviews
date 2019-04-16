from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os
import numpy as np

# import HashingVectorizer from local dir

app = Flask(__name__)


from fastai import *
from fastai.vision import *

#export_file_url = 'https://drive.google.com/uc?export=download&id=1Wb46r11xYneUWbDcTPO2zxkK899UATrg'
export_file_name = 'export.pkl'

path = Path(__file__).parent

cur_dir = os.path.dirname(__file__)

path=os.path.join(cur_dir,
                 'model',
                 'export.pkl')
path=Path(path)

 def setup_learner():
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise



######## Preparing the Classifier


async def analyze(document):
    learn=setup_learner()
    prediction = learn.predict(document)
    p=prediction[1]
    p=p.item()
    prob=prediction[2][p].item()
    return prediction[0],prob

######## Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = analyze(review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)



if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)

