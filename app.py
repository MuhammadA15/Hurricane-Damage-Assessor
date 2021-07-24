import os
import glob
from os import walk

from flask import Flask, render_template, url_for, redirect, send_from_directory, flash, request, make_response
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask import request
from werkzeug.urls import url_parse
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np

import pdfkit

from .config import Config

from bokeh.embed import components 
from bokeh.models import GMapOptions, ColumnDataSource, HoverTool, ColorBar
from bokeh.plotting import gmap
from bokeh.transform import linear_cmap
from bokeh.palettes import YlOrRd

from .model_predictions import HurricaneLosses

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login = LoginManager(app)
login.login_view = 'login'

# Install wkhtmltopdf via https://wkhtmltopdf.org/downloads.html and set path here for PDF generator
path_wkhtmltopdf = 'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

bootstrap = Bootstrap(app)

from .db_models import User

@app.route('/', methods=['GET', 'POST'])
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():        
    if request.method == 'POST':
        f = request.files['csv_file']
        _dir = 'users'
        _dir = os.path.join(_dir, current_user.username)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        f.save(_dir + '/' + f.filename)
        flash("File Uploaded Successfully")
    return render_template('dashboard.html')


@app.route('/hurricanes', methods=['GET', 'POST'])
@login_required
def hurricanes():
    predictions = []
    script = None
    div = None
    try:
        _, _, filenames = next(walk('users/' + current_user.username))
    except:
        filenames = []
    if request.method == 'POST': 
        selected = request.form.get('csv_select')
        if selected is not None:     
            _dir = 'users/' + current_user.username + '/' + selected
            if os.path.exists(_dir):
                loss = HurricaneLosses(current_user.username)
                predictions = loss.predict_losses(_dir)
                #predictions = pd.read_csv(_dir)
                predictions_dir = 'users/' + current_user.username + '/data'
                if not os.path.exists(predictions_dir):
                    os.makedirs(predictions_dir)
                predictions_file = predictions_dir + '/' + selected.replace('.csv', '') + '_Predictions.csv'
                predictions.to_csv(predictions_file)
                points = predictions.to_dict(orient='records')
                
                x = [p.get('glon') for p in points]
                y = [p.get('glat') for p in points]
                lng = sum(x) / len(points) 
                lat = sum(y) / len(points)
                predictions['radius'] = np.sqrt(predictions['lossprediction']) * 15
                gmap_options = GMapOptions(lat=lat, lng=lng, map_type='roadmap', zoom=7)
                # Set API Key for Google Maps API here
                api_key = None
                with open("api_key.txt", "r") as file:
                    api_key = file.read().strip()
                hover = HoverTool(
                    tooltips = [
                        ('Location', '@streetname'),
                        ('City', '@city'),
                        ('vmax_gust', '@vmax_gust'),
                        ('vmax_sust', '@vmax_sust'),
                        ('Predicted Loss', '@lossprediction')
                    ]
                )
                p = gmap(api_key, gmap_options, title='Loss Map', width=700, height=500, tools=[hover, 'reset', 'wheel_zoom', 'pan'])
                source = ColumnDataSource(predictions)
                palette = YlOrRd[9]
                newPalette = palette[::-1]
                high = predictions['lossprediction'].max()
                low = predictions['lossprediction'].min()
                mapper = linear_cmap('lossprediction', newPalette, low, high)
                center = p.circle('glon', 'glat', size='radius', line_color='black', line_width=1, alpha=0.6, color=mapper, source=source)
                color_bar = ColorBar(color_mapper=mapper['transform'], location=(0,0))
                p.add_layout(color_bar, 'right')
                script, div = components(p)

                return render_template('hurricanes.html', filenames=filenames, script=script, div=div)
    return render_template('hurricanes.html', filenames=filenames, script=script, div=div)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    username = request.form.get('username')
    password = request.form.get('password')
    remember = request.form.get('rememberMe')
    if request.method == 'POST':
        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            flash('Invalid Username or Password', 'failedLogin')
            return redirect(url_for('login'))
        login_user(user, remember)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('dashboard')
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    errors = False
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password2 = request.form.get('password2')
        email = request.form.get('email')
        user = User.query.filter_by(username=username).first()
        if user is not None:
            flash('Username Already Exists', 'username')
            errors = True
        user = User.query.filter_by(email=email).first()
        if user is not None:
            flash('Email Already In Use', 'email')
            errors = True
        if password != password2:
            flash('Password Does Not Match', 'password')
            errors = True
        if errors:
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('You are now registered!', 'registered')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/generatePDF', methods=['POST'])
@login_required
def generatePDF():
    selected = request.form.get('csv_select')
    if selected is not None:
        prediction_file = selected.replace('.csv', '') + '_Predictions.csv'     
        _dir = 'users/' + current_user.username + '/data/' + prediction_file
        if os.path.exists(_dir):
            output = pd.read_csv(_dir)
            output = output.to_dict(orient='records')
            res = render_template('pdf_template.html', output=output)
            responseString = pdfkit.from_string(res, False, configuration=config)
            response = make_response(responseString)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Diposition'] = 'attachment;filename=output.pdf'
            return response
        else:
            flash('You must generate predictions for the selected policy before you can download a report')
            return redirect(url_for('hurricanes'))
