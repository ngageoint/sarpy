from wtforms import StringField
from flask_wtf import FlaskForm


class FrameForm(FlaskForm):
    slider_val = StringField('slider_val')
