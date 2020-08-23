# 内部类

**内部类：定义在另一个类中的类**

**优点**

> - 内部类方法可以访问外部定义所在的作用域中的数据，包括私有的数据，而不破坏封装性
> - 内部类可以对同一个包中的其他类隐藏起来
> - 当想要定义一个回调函数的时候，使用匿名内部类比较便捷

**内部类的分类**

> - 成员内部类
> - 静态内部类
> - 局部内部类
> - 匿名内部类

**成员内部类**

> - 在类的内部定义，与实例变量和实例方法同级别的类
>
> - 外部类的一个实例部分，创建内部类对象时，必须依赖外部类
>
>   - ```java
>     Outer outer = new Outer();
>     // 内部类的创建
>     Outer.Inner inner = outer.new Inner();
>     ```
>
>   - ```java
>     Outer.Inner inner = new Outer().new Inner();
>     ```
>
> - 当外部类，内部类存在重名的属性时，优先访问内部类的属性，如果要访问外部类的属性
>
> ​        Outer.this.属性   即可访问外部类的属性
>
> - 成员内部类不能定义静态成员，但是可以包含静态常量（static final）

**静态内部类**（在成员内部类上添加static关键字）

> - 不依赖外部类对象，可以直接或通过类名访问或创建对象
>
> ```java
> Outer.Inner inner = new Outer.Inner();
> ```
>
> - 可以声明（包含）静态成员
> - 静态内部类与外部类是同级别的，内部类中调用外部类的属性和方法时，需要在内部类中创建外部类对象，通过对象进行调用
> - 静态内部类可以直接调用内部类的对象和方法

**局部内部类**

> - 定义在外部类的方法中，作用范围和创建对象的范围仅限于当前方法
> - 局部内部类与局部变量同级别，创建时不能添加任何访问修饰符
> - 局部内部类可以直接访问外部类，内部类的属性
> - 局部内部类访问局部变量时，局部变量前添加final关键字
> - 局部内部类需要在定义的外部类的方法中完成对象的创建

**匿名内部类**

> - 没有类名的局部内部类（一切特征与局部内部类相同）
> - 必须继承一个父类或者一个接口
> - 只能创建一个该类的对象
> - 可读性较差

# Object类

> - 是超类，基类，他是所有类的直接或者间接父类，位于继承树的最顶端
> - 任何类，如果没有书写继承哪个类，都默认继承Object类，否则为间接继承
> - Object类中定义的方法是所有对象都具备的方法
> - Object类可以存储任何对象
>   - 作为参数，可以接收任何对象
>   - 作为返回值，可以返回任何对象

## Object方法

- getClass() 方法

public final Class<?> getClass():返回对象的类，在反射机制中有用

- hashCode()方法

public int hashCode() ：返回一个对象的哈希代码值，相同对象返回相同的哈希码

- toString() 方法

public String toString() ：返回对象的字符串表示形式。总的来说，这 toString方法返回一个字符串，“以文本方式表示”这个对象， **建议所有子类都重写此方法，通常为展示对象的属性**

- equals（）方法

public boolean equals(Object obj) ：比较对象是否相同（通过比较地址来判断）

- finalize() 方法

当对象被判定为垃圾对象时，由JVM自动调用该方法，用以标记为垃圾对象，进入回收队列

# 包装类

基本数据类型所对应的引用类型，就有了属性和方法可以调用

| 基本数据类型 | 包装类型  |
| ------------ | --------- |
| byte         | Byte      |
| short        | Short     |
| int          | Integer   |
| long         | Long      |
| float        | Float     |
| double       | Double    |
| boolean      | Boolean   |
| char         | Character |

---

## 类型转换与装箱，拆箱

装箱：将基本类型转换成引用类型

拆箱：将引用类型转换成基本类型

```java
ublic class demo01 {
    public static void main(String[] args) {
        int num = 10;
        // 装箱
        // 使用 Integer 对象来创建
        Integer integer1 = new Integer(num);
        Integer integer2 = Integer.valueOf(num);

        // 拆箱
        Integer integer3 = new Integer(100);
        int num2 = integer3.intValue();

        // JDK1.5之后,提供自动装箱拆箱，本质上是Integer.valueOf()方法
        int a = 10;
        Integer i = a;

    }
}
```

----

## Integer

| 方法                                 | 描述                                                    |
| ------------------------------------ | ------------------------------------------------------- |
| public String toString()             | 返回表示这 `Integer`价值的 `String`对象，即转换为字符串 |
| public static int max(int a,  int b) | 返回其中的较大的值，min同理                             |
| public static int parseInt(String s) | 将字符串参数作为带符号的十进制整数                      |
| public int intValue()                | 作为一个 int返回该 Integer的值。                        |

---

**整数缓冲区**：在于当传入的数值在-128与127之间时，会被缓存在一个对象数组中，这样设计的原因是出于高并发时需要处理大量的数据，提前创建一个数组储存常用整数以备使用，可以用以缓冲，而当传入的数值不在这个范围之内，程序便会创建一个新的对象

# String类

- 字符串是常量，创建之后不可改变，变量内容修改的话，是**变量指向发生了改变**（重新开辟空间）
- 字符串字面值存储在字符串池中，可以共享
- 字符串的创建
  - String str = "hello";        //产生一个对象，字符串池中存储
  - String str = new String("hello")；   // 产生两个对象，堆和池中各一个，此时栈中的变量是指向堆中的

```java
 public static void main(String[] args) {
        String name = "hello";     //"hello"这个常量存储在字符串池中
        String name2 = "hello";     //变量name2指向池中的"hello","hello"为同一个

        String s1 = new String("java");   //堆和池都创建了一个对象"java"，变量s1指向堆中的
        String s2 = new String("java");   //堆和池重新都创建了一个对象"java"，变量s2指向堆中的，两个"java"不是同一个

        System.out.println(s1 == s2);    //false，==比较的是地址
        System.out.println(s1.equals(s2));   //true，equals比较的是数据
        System.out.println(name == name2);     // true
    }
```

---

## String常用方法

| 方法                                             | 描述                                    |
| ------------------------------------------------ | --------------------------------------- |
| public String[] split(String str)                | 以str作为分隔符拆分字符串               |
| public String trim()                             | 去掉字符串前后的空格                    |
| public char charAt(int index)                    | 返回指定索引的 `char`值                 |
| public boolean contains(String str)              | 判断当前字符串是否包含str               |
| public char[] toCharArray()                      | 将此字符串转换为一个新的字符数组。      |
| public int indexOf(String str)                   | 查找str首先出现的下标，若不存在则返回-1 |
| public int lastIndexOf(tring str)                | 查找str最后一次出现的下标               |
| public String toUpperCase()                      | 将小写转为大写                          |
| public boolean endsWith(String str)              | 判断该字符串是否以str结尾               |
| public String replace(char oldChar,char newChar) | 将旧字符串替换为新字符串                |

---

## 可变字符串

- StringBuffer:可变字符串，速度慢，线程安全
- StrungBuilder:可变字符串，速度快，线程不安全

可变字符串比String效率高，更节省空间

**StrungBuilder常用方法**

| 方法                                                         | 描述                                    |
| ------------------------------------------------------------ | --------------------------------------- |
| public StringBuilder append(char c)                          | 在字符串后追加字符或者字符串            |
| public StringBuilder insert(int offset,char[] str)           | 在指定位置插入字符或者字符串            |
| public StringBuilder replace(int start, int end,  String str) | 将指定位置的字符串进行替换，[start,end) |
| public StringBuilder delete(int start, int end)              | 删除指定位置的字符串                    |
| public StringBuilder reverse()                               | 将字符串进行反转                        |



****

# BigDecimal类

```java
public static void main(String[] args) {
    double d1 = 1.0;
    double d2 = 0.9;
    System.out.println(d1-d2);
}
```

输出结果：0.09999999999999998

double是近似值存储，需要使用BigDecimal类提高精度

BigDecimal

> - java.math包中
> - 作用：精确计算浮点数
> - 创建方式  BigDecimal bd1 = new BigDecimal(1.0);
> - 运算：add,substract,multiply,divide

---

# Calendar类

提供了设置和操作日历的方式

```java
public static void main(String[] args) {
    // 创建Calendar对象，不可以通过new 创建
    Calendar calendar = Calendar.getInstance();
    // 获取时间信息
    // 获取年信息
    int year = calendar.get(Calendar.YEAR);
    System.out.println(year);
    // 获取月  0-11
    int month = calendar.get(Calendar.MONTH);
    // 获取日
    int day = calendar.get(Calendar.DAY_OF_MONTH);
    // 获取星期几
    int day2 = calendar.get(Calendar.DAY_OF_WEEK);
    // 获取小时
    int hour = calendar.get(Calendar.HOUR);
    // 获取分钟
    int minutes = calendar.get(Calendar.MINUTE);
    // 获取秒
    int sec = calendar.get(Calendar.SECOND);

    System.out.println("年："+year+"月："+(month+1)+"日"+day+"小时"+hour+"分："+minutes+"秒："+sec);
}
```

