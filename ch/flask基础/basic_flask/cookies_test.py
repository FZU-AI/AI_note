from flask import *
app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def results():
    return render_template("index.html")

@app.route('/set_cookies',methods = ['POST', 'GET'])
def set_cookies():
    if request.method == 'POST':
        name = request.form.get('nm')
    elif request.method == 'GET':
        name = request.args.get('nm')
    resp = make_response('<h1>set '+str(name)+'</h1>')
    resp.set_cookie('userID', name)
    return resp

@app.route('/get_cookies')
def get_cookies():
   name = request.cookies.get('userID')
   return '<h1>welcome '+str(name)+'</h1>'

if __name__ == '__main__':
   app.run(debug = True)