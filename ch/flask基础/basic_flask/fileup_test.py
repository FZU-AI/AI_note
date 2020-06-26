from flask import *
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)


@app.route('/upload', methods = ['GET', 'POST'])
def upload_index():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      UPLOAD_FOLDER = "./img"
      file_dir = os.path.join(os.getcwd(), UPLOAD_FOLDER)
      if not os.path.exists(file_dir):
         os.makedirs(file_dir)
      #从url获得文件,设置对应的路径
      file = request.files['file']
      print(file.filename)
      file_path = os.path.join(file_dir, file.filename)
      file.save(file_path)
      return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(debug = True)