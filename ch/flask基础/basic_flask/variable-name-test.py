from flask import Flask

app = Flask(__name__)

@app.route('/hello/<name>/kkk')
def hello_benben(name):
    #传参里的name和装饰器里的name对应
    return 'hello {0}'.format(str(name))

@app.route('/blog/<int:postID>/')
def show_blog(postID):
   return 'Blog Number {0}'.format(int(postID))

@app.route('/rev/<float:revNo>/')
def revision(revNo):
   return 'Revision Number {0}'.format(revNo)

@app.route('/kk/<path:uu>/')
def path(uu):
   return 'path: {0}'.format(str(uu))


if __name__ == '__main__':
    app.run(debug=True)