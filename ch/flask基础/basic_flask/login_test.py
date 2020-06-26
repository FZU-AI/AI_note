from flask import *
app = Flask(__name__)
app.secret_key = "sadsfsa"

@app.route('/')
def index():
    return render_template('log_in.html')

@app.route('/login',methods = ['POST', 'GET'])
def login():
    #print(request.form)
    if request.method == 'POST':
        if request.form['username'] == 'admin' :
            return redirect(url_for('success'))
        else:
            abort(401)
    elif request.method == 'GET':
        if request.args.get('username')== 'admin' :
            return redirect(url_for('success'))
        else:
            abort(401)
    return redirect(url_for('index'))

@app.route('/success')
def success():
   return 'logged in successfully'

if __name__ == '__main__':
   app.run(debug = True)