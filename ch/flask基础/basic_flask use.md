Flask：Web框架，基于Werkzeug WSGI工具包和Jinja2模板引擎。

## Flask 基本使用

```
from flask import Flask
#项目必须导入Flask模块。Flask类的一个对象是WSGI应用程序。Flask构造函数使用当前模块（__name __）的名称作为参数。
app = Flask(__name__)

#指hello函数与初始url绑定。
@app.route('/')
def hello_world():
   return "hello"
#运行程序
if __name__ == '__main__':
   app.run()
```

> Flask类的**run()**方法在本地开发服务器上运行应用程序。
>
> ```
> app.run(host, port, debug, options)
> ```
>
> host：要监听的主机名，port指端口，debug：指是否提供调试信息，options指传参。



## Flask 路由

### 绑定url

Flask绑定路由常见的包括两种方式：

一种通过route装饰器绑定路由

> Flask类的**route()**函数告诉应用程序哪个URL应该调用相关的函数。
>
> ```
> app.route(rule, options)
> ```
>
> rule指与函数绑定的路由，options指传参。

一种是通过application对象的add_url_rule()函数实现。

> ```
> app.add_url_rule(url,endpoint,view_func)
> ```
>
> url指函数绑定的路由地址；endpoint指站点，使用url_for()进行反转传入的第一个参数是endpoint的值，可不指定；view_func指对应的函数，这个只需要写函数名字，不需要加括号。

### 构建url

**url_for()**函数对于动态构建特定函数的URL

url_for的操作对象是函数，而不是路径。想要完整的URL，可以设置`_external`为Ture。

### 提取url信息

服务器处理客户端请求数据，在Flask模块中通过导入Request对象进行处理。Request对象的重要属性如下所列：

- **Form** - 它是一个字典对象，包含表单参数及其值的键和值对。
- **args** - 解析查询字符串的内容，它是问号（？）之后的URL的一部分。
- **Cookies** - 保存Cookie名称和值的字典对象。
- **files** - 与上传文件有关的数据。
- **method** - 当前请求方法。

#### url传递数据的方法

- get方法：以未加密的形式将数据发送到服务器。最常见的方法。
- head方法：和GET方法相同，但没有响应体。
- post方法：将HTML表单数据发送到服务器。POST方法接收的数据不由服务器缓存。
- put方法：用上传的内容替换目标资源的所有当前表示。
- delete方法：删除由URL给出的目标资源的所有当前表示。

Flask默认响应GET请求，可以通过route()装饰器里的methods添加新的响应方法。

前端网页可以通过form传递参数

```
<form action = "http://localhost:5000/login" method = "post">
	<p>Enter Name:</p>
	<p><input type = "text" name = "nm" /></p>
    <p><input type = "submit" value = "submit" /></p>
</form>
```

#### 服务器提取数据的方法

##### 对于**POST**方法

```
user = request.form['nm']
```

##### 对于**GET**方法

```
user = request.args.get('nm')
```

**args**是包含表单参数对及其对应值对的列表的字典对象。



## Flask 变量规则

指向规则参数中添加变量部分，目的是实现**动态构建url**。此变量部分标记为<variable-name>。它作为关键字参数传递给与规则**相关联的函数**。

简单来说：就是将路由中一部分内容作为参数传给相关联函数。

```
@app.route('/hello/<name>/kkk/')
def hello_benben(name):
    #传参里的name和装饰器里的name对应
    return 'hello {0}'.format(str(name))
```

除了默认的字符串变量部分，还可以使用以下转换器构建规则：

- int:接受整数
- float:接受浮点数
- path:接受用作目录分隔符的斜杠

```
@app.route('/blog/<int:postID>/')
def show_blog(postID):
   return 'Blog Number {0}'.format(int(postID))

@app.route('/rev/<float:revNo>/')
def revision(revNo):
   return 'Revision Number {0}'.format(revNo)

@app.route('/kk/<path:uu>/')
def path(uu):
   return 'path: {0}'.format(str(uu))
```

## Flask 模板

可以以HTML的形式返回绑定到某个url的函数的输出。可以通过**render_template()**函数呈现HTML文件。

```
return render_template(‘hello.html’)
```

使用了**render_template()**函数后会在项目路径里的templates文件夹里寻找html文件

### web templating system

web模板系统，指设计一个html脚本，其中可以动态的插入变量数据。web模板系统包括模板引擎，某种数据源和模板处理器。

Flask使用**jinga2**模板引擎。模板处理采用HTML语法散布占位符。

**Jinja2**模板引擎使用以下分隔符从HTML转义。

- {% ... %}用于语句
- {{ ... }}用于表达式可以打印到模板输出
- {# ... #}用于未包含在模板输出中的注释
- \# ... ##用于行语句

## Flask 重定向

函数调用的另一种方式，可以使用**redirect()**函数

```
Flask.redirect(location, statuscode, response)
```

- **location**参数是应该重定向响应的URL。
- **statuscode**发送到浏览器标头，默认为302。
- **response**参数用于实例化响应。



## Flask 静态文件

Web应用程序通常需要静态文件，Flask通过**/static**路径进行存储静态文件。包括一些js,css文件。

## Flask Cookies

Cookie：用于记住和跟踪与客户使用相关的数据。一般是以文本的形式存储在客户端处。

**Request对象**包含Cookie的属性。cookie还存储其网站的到期时间，路径和域名。

### 设置cookies

在Flask中，对响应对象设置cookie。使用**make_response()**函数从视图函数的返回值**获取响应对象**。之后使用响应对象的**set_cookie()**函数来存储cookie。

### 读取cookies

request.cookies属性的**get()**方法用于读取cookie。

```
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
```

## Flask Sessions

**Session（会话）**数据存储在服务器上。会话是客户端登录到服务器并注销服务器的时间间隔。

需要在该会话中保存的数据会存储在服务器上的**临时目录**中。

具体需要做的事：分配**会话ID**，会话数据存储在cookie的顶部，服务器以加密方式对其进行签名。

### Session设置变量

```
Session[‘username’] = ’admin’
#设置一个'username'会话变量
```

### Session释放变量

```
session.pop('username', None)
```

## Flask 消息闪现

用于与用户进行交互，传递消息。Flask框架的闪现系统可以在一个视图中**创建消息**，并在名为**next**的视图函数中呈现它。

Flask模块包含**flash()**方法。它将**消息传递给下一个请求**，该请求通常是一个**模板**。

```
flash(message, category)
```

- **message**参数是要闪现的实际消息。
- **category**参数是可选的。它可以是“error”，“info”或“warning”。

### 删除信息

从会话中删除消息，模板调用**get_flashed_messages()**。

```
get_flashed_messages(with_categories, category_filter)
```

如果接收到的消息具有类别，则第一个参数是元组。第二个参数仅用于显示特定消息。

```
 if request.method == 'POST':
        if request.form['username'] != 'admin' or \
            request.form['password'] != 'admin':
            error = 'Invalid username or password. Please try again!'
        else:
            flash('You were successfully logged in')
            return redirect(url_for('index'))
```

## Flask File上传

1. HTML表单：**enctype**属性设置为“multipart / form-data”
2. URL处理程序从**request.files[]**对象中提取文件，保存到指定位置。

Flask对象的配置设置中定义**默认**上传文件夹的**路径**和上传文件的**最大大小**。

| name                           |                    mean                    |
| ------------------------------ | :----------------------------------------: |
| app.config[‘MAX_CONTENT_PATH’] |            定义上传文件夹的路径            |
| app.config[‘MAX_CONTENT_PATH’] | 指定要上传的文件的最大大小（以字节为单位） |

```
#html form
<form action = "http://localhost:5000/uploader" method = "POST" 
         enctype = "multipart/form-data">
         <input type = "file" name = "file" />
         <input type = "submit"/>
</form>

#python
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      UPLOAD_FOLDER = "./img"
      file_dir = os.path.join(os.getcwd(), UPLOAD_FOLDER)
      if not os.path.exists(file_dir):
         os.makedirs(file_dir)
      file = request.files['file']
      print(file.filename)
      file_path = os.path.join(file_dir, file.filename)
      file.save(file_path)
      return 'file uploaded successfully'
```

## Flask 扩展

Flask通常被称为微框架，因为核心功能包括基于**Werkzeug**的WSGI和路由以及基于**Jinja2**的模板引擎。Flask扩展为Flask框架提供了可扩展性。常见扩展包括： 	

- **Flask Mail** - 为Flask应用程序提供SMTP接口
- **Flask WTF** - 添加WTForms的渲染和验证
- **Flask SQLAlchemy** - 为Flask应用程序添加SQLAlchemy支持
- **Flask Sijax** - Sijax的接口 - Python/jQuery库，使AJAX易于在Web应用程序中使用

Flask扩展名通常命名为flask-foo。导入的操作如下：

```
from flask_foo import [class, function]
```

对于0.7以后的Flask版本，您还可以使用语法：

```
from flask.ext import foo
```

对于此用法，需要激活兼容性模块。它可以通过运行flaskext_compat.py来安装：

```
import flaskext_compat
flaskext_compat.activate()
from flask.ext import foo
```

### Flask Mail

Flask Mail是用于与电子邮件服务器建立简单的接口。需要通过设置以下应用程序参数的值来配置Flask-Mail。

| 序号 |                          参数与描述                          |
| ---- | :----------------------------------------------------------: |
| 1    |          **MAIL_SERVER**电子邮件服务器的名称/IP地址          |
| 2    |              **MAIL_PORT**使用的服务器的端口号               |
| 3    |           **MAIL_USE_TLS**启用/禁用传输安全层加密            |
| 4    |          **MAIL_USE_SSL**启用/禁用安全套接字层加密           |
| 5    |   **MAIL_DEBUG**调试支持。默认值是Flask应用程序的调试状态    |
| 6    |               **MAIL_USERNAME**发件人的用户名                |
| 7    |                **MAIL_PASSWORD**发件人的密码                 |
| 8    |            **MAIL_DEFAULT_SENDER**设置默认发件人             |
| 9    |          **MAIL_MAX_EMAILS**设置要发送的最大邮件数           |
| 10   | **MAIL_SUPPRESS_SEND**如果app.testing设置为true，则发送被抑制 |
| 11   | **MAIL_ASCII_ATTACHMENTS**如果设置为true，则附加的文件名将转换为ASCII |

#### Mail类

管理电子邮件消息传递需求。

构造

```
flask-mail.Mail(app = None)
```

具体操作

- **connect()**：打开与邮件主机的连接
- **send()**：发送Message类对象的内容
- **send_message()**：发送消息对象

#### Message类

封装了一封电子邮件。

构造

```
flask-mail.Message(subject, recipients, body, html, sender, cc, bcc, reply-to, date, charset, extra_headers, mail_options, rcpt_options)
```

具体操作

- **attach()** - 为邮件添加附件。此方法采用以下参数：

  > **filename** - 要附加的文件的名称
  >
  > **content_type** - MIME类型的文件
  >
  > **data** - 原始文件数据
  >
  > **处置** - 内容处置（如果有的话）。

- **add_recipient()** - 向邮件添加另一个收件人

### Flask WTF

用户输入的数据以Http请求消息的形式通过GET或POST方法提交给服务器端脚本。

- 服务器端脚本必须从http请求数据重新创建表单元素。因此，实际上，表单元素必须定义两次 - 一次在HTML中，另一次在服务器端脚本中。
- 使用HTML表单的另一个缺点是很难（如果不是不可能的话）动态呈现表单元素。HTML本身无法验证用户的输入。

**WTForms**：一个灵活的表单，渲染和验证库，能够方便使用。

安装WTF:

```
pip install flask-WTF
```

WTF包含一个**Form**类，该类必须用作用户定义表单的父级。

**WTforms**包中包含各种表单字段的定义。下面列出了一些**标准表单字段**。

#### **标准表单字段**

| 序号 | 标准表单字段与描述                                          |
| ---- | ----------------------------------------------------------- |
| 1    | **TextField**表示<input type ='text'> HTML表单元素          |
| 2    | **BooleanField**表示<input type ='checkbox'> HTML表单元素   |
| 3    | **DecimalField**用于显示带小数的数字的文本字段              |
| 4    | **IntegerField**用于显示整数的文本字段                      |
| 5    | **RadioField**表示<input type = 'radio'> HTML表单元素       |
| 6    | **SelectField**表示选择表单元素                             |
| 7    | **TextAreaField**表示<testarea> HTML表单元素                |
| 8    | **PasswordField**表示<input type = 'password'> HTML表单元素 |
| 9    | **SubmitField**表示<input type = 'submit'>表单元素          |

WTForms包也包含验证器类。它对表单字段应用验证很有用。以下列表显示了常用的验证器。

#### 常用的验证器

| 序号 | 验证器类与描述                                          |
| ---- | ------------------------------------------------------- |
| 1    | **DataRequired**检查输入字段是否为空                    |
| 2    | **Email **检查字段中的文本是否遵循电子邮件ID约定        |
| 3    | **IPAddress** 在输入字段中验证IP地址                    |
| 4    | **Length** 验证输入字段中的字符串的长度是否在给定范围内 |
| 5    | **NumberRange**验证给定范围内输入字段中的数字           |
| 6    | **URL**验证在输入字段中输入的URL                        |

### Flask SQLite

实质是连接sqlite3，对数据集进行增删改查

### Flask SQLAlchemy

Flask-SQLAlchemy是Flask扩展，它将对SQLAlchemy的支持添加到Flask应用程序中。

安装Flask-SQLAlchemy扩展

```
pip install flask-sqlalchemy
```

您需要从此模块导入SQLAlchemy类。

```
from flask_sqlalchemy import SQLAlchemy
```

创建一个Flask应用程序对象并为要使用的数据库设置URI。

```
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'
```

使用应用程序对象作为参数创建SQLAlchemy类的对象。该对象包含用于ORM操作的辅助函数。它还提供了一个父Model类，使用它来声明用户定义的模型。

```
db = SQLAlchemy(app)
class students(db.Model):
    id = db.Column('student_id', db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    city = db.Column(db.String(50))  
    addr = db.Column(db.String(200))
    pin = db.Column(db.String(10))
    def __init__(self, name, city, addr,pin):
       self.name = name
       self.city = city
       self.addr = addr
       self.pin = pin
```

 要创建/使用URI中提及的数据库，请运行**create_all()**方法。

```
db.create_all()
```

#### CRUD

**SQLAlchemy**的**Session**对象管理**ORM**对象的所有持久性操作。

以下session方法执行CRUD操作：

- **db.session.add** (模型对象) - 将记录插入到映射表中
- **db.session.delete** (模型对象) - 从表中删除记录
- **db.session.commit()** - 提交记录
- **model.query.all()** - 从表中检索所有记录（对应于SELECT查询）。

您可以通过使用filter属性将过滤器应用于检索到的记录集。

### Flask Sijax

**Sijax**代表**'Simple Ajax'**，它是一个**Python/jQuery**库，它使用**jQuery.ajax**来发出AJAX请求。

#### 安装

```
pip install flask-sijax
```

#### 组态

- **SIJAX_STATIC_PATH** - 要被镜像的Sijax javascript文件的静态路径。默认位置是**static/js/sijax**。在此文件夹中，保留**sijax.js**和**json2.js**文件。
- **SIJAX_JSON_URI** - 从中加载json2.js静态文件的URI

## Flask 部署

### 外部可见服务器

禁用了**debug**，则可以通过将主机名设置为**'0.0.0.0'**，使本地计算机上的开发服务器可供网络上的用户使用。

### 部署——mod_wsgi

**mod_wsgi**是一个Apache模块，它提供了一个WSGI兼容接口，用于在Apache服务器上托管基于Python的Web应用程序。

#### 安装mod_wsgi

```
pip install mod_wsgi
```

