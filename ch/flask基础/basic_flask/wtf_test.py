from flask import *
from wtf_form import ContactForm
app = Flask(__name__)
app.secret_key = 'asdgfd'


@app.route('/contact', methods=['GET', 'POST'])
def contact():
   form = ContactForm()
   if request.method == 'POST':
      if form.validate() == False:
         flash('All fields are required.')
         return render_template('wtf_contact.html', form=form)
      else:
         return render_template('wtf_success.html')
   elif request.method == 'GET':
      return render_template('wtf_contact.html', form=form)


if __name__ == '__main__':
   app.run(debug=True)