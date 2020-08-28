import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from content_review import content_review_func
from flask import send_file
from flask import render_template
import pandas as pd

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            review_results = content_review_func(filename)

            return review_results

    return render_template('frontend.html')

@app.route('/download_template/',methods=["POST"])
def download_template():
    template_file = 'Keywords_DB.xlsx'
    print ('test')
    return send_file('./model/{}'.format(template_file), attachment_filename='{}'.format(template_file),as_attachment=True)

