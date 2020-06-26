from flask_wtf import FlaskForm
from wtforms import *

from wtforms import validators, ValidationError
from email_validator import validate_email, EmailNotValidError

class ContactForm(FlaskForm):
   name = StringField("Name Of Student", [validators.DataRequired("Please enter your name.")])
   Gender = RadioField('Gender', choices=[('M', 'Male'), ('F', 'Female')])
   Address = TextAreaField("Address")

   email = StringField("Email", [validators.DataRequired("Please enter your email address."), \
                                     validators.Email("Please enter your email address.")])

   Age = IntegerField("age")
   language = SelectField('Languages', choices=[('cpp', 'C'),
                                                 ('py', 'Python')])
   submit = SubmitField("Send")