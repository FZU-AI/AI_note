## 面向对象

#### 1.类和对象

```
类:
	对于一类事物的统称，对当前事物的一些描述，属性描述和行为描述
对象:
	独立，唯一，特殊的个体
```

#### 2.Java中定义类的格式

```java
/*
class ClassName {
	// 属性描述
	// 行为描述
}
要求:
	1. ClassName要求符合大驼峰命名法，并且要做到见名知意！！！
	2. 属性描述，是用于描述当前类拥有的一些特征，这里可以使用变量
	   该变量有一个特定的名字【成员变量】 Field
	3. 行为描述，是用于描述当前类可以做到一些事情，这里使用方法。
	   该方法有一个特定的名字【成员方法】Method
*/
```

##### 2.1 Java中定义类【成员变量】

```java
/**
 * 这里定义的是一个Person类
 * 
 * @author Anonymous
 */
class Person {
	// 属性描述，这里使用成员变量 Field
	// 使用String类型来描述Person类的姓名属性
	String name;
	
	// 使用int类型数据描述Person类的年龄属性
	int age;
	
	// 使用char类型数据描述Person类的性别属性
	char gender;
}
```

##### 2.2 Java中定义类【成员方法】

```java
// 使用方法描述Perosn类的吃饭行为                            
public void eat(String food) {                   
	System.out.println("吃" + food);              
}                                                
                                                 
// 使用方法描述Person类的睡觉行为                            
public void sleep() {                            
	System.out.println("说出来你可能不行，是床先动的手~~~");    
}                                                
                                                 
// 使用方法描述Person类的打豆豆行为                           
public void daDouDou() {                         
	System.out.println("吃饭，睡觉，打豆豆");             
}                 

/*
特征:
	1. 目前没有使用static关键字
	2. 方法和之前所学函数形式一致，同样拥有三大元素 返回值，方法名和形式参数列表
*/
```

#### 3. 类对象使用

##### 3.1 Java中创建类对象的形式

```java
/*
new 对象!!!
new关键字 :
	申请内存的【堆区】空间，并且清理整个空间中的所有数据。
	代码中，只有有new关键字，一定会使用到内存的堆区空间，并且是新的内存空间。
总结:
	类名 对象名 = new 构造方法(所需参数);
*/

// 这里创建了一个Person类对象，对象名person                               
Person person = new Person();                               
System.out.println(person);                                 
/*                                                          
 * QFNZ_Day08.Person@5fd0d5ae                       
 * QFNZ_Day08 完整的包名                                 
 * Person 数据类型，这里创建的对象是一个Person类对象                      
 * @15db9742 当前类对象在内存空间的中的首地址！！！十六进制展示方式           
 */                                                         
                                                            
Person person1 = new Person();                              
System.out.println(person1);                                
/*                                                          
 * QFNZ_Day08.Person@2d98a335                       
 * 发现第二个Person类对象 person1空间首地址2d98a335 和第一个对象不一致     
 * 这里就意味着两个对象的空间首地址不一致，不是同一个对象！！！            
 */                                                         
```

##### 3.2 Java中使用类对象调用成员变量

```java
/*
int[] arr = new int[10];
获取数组的容量:
	arr.length
	获取一个数组中【的】length属性

格式:
	对象名.属性名/成员变量名;
	. ==> 的
	可以操作取值或者赋值操作。
*/

// 通过Person类对象 person调用类内的成员变量                      
// 【赋值】Person类对象 person中对应的成员变量                     
person.name = "骚磊";                                 
person.age = 16;                                    
person.gender = '男';                                
                                                    
// 【取值】展示Person类对象 person中保存的成员变量数据                 
System.out.println("Name:" + person.name);          
System.out.println("Age:" + person.age);            
System.out.println("Gender:" + person.gender);      
```

##### 3.3 Java中使用类对象调用成员方法

```java
/*
得到了一个Scanner类对象sc
Scanner sc = new Scanner(System.in);
使用过以下方法：
	sc.nextInt();
	sc.nextFloat();
	sc.nextLine().charAt(0);

格式:
	类对象.方法名(必要的参数);
	. ==> 的
*/
```

#### 4. 类对象内存分析图

![](.\img\类对象内存分析图.png)

#### 5. 类对象内存转移问题

![](.\img\类对象内存转移问题分析图.png)

#### 6. 构造方法 Constructor

##### 6.1 Java编译器提供的默认的构造方法

```
	通过反编译工具，发现了一些不存在与代码中的内容，而这段内容就是Java编译器为了方便程序开发，提供的一个必要的无参数构造方法。
	Java编译器如果发现当前class没有显式自定义构造方法，会默认提供一个无参数构造方法给予使用。
	如果Java编译器发现在代码中出现了任何一个构造方法，就不会再提供对应的无参数构造方法。
```

##### 6.2 自定义使用构造方法

```
构造方法功能:
	初始化当前类对象中保存的成员变量数据。
目前创建对象的方式;
	new 构造方法(有可能需要的参数);

new：
	1. 根据构造方法提供的数据类型申请对应的堆区内存空间。
	2. 擦除整个空间中所有的数据。零值
构造方法:
	初始化在当前内存堆区空间的成员变量对应的数据
	
格式:
	public 类名(初始化形式参数列表) {
		初始化赋值语句;
	}

要求:
	1. 无论什么时候一定要留有一个无参数构造方法备用
	2. 根据所需情况完成构造方法参数选择
	3. 一个class可以有多个构造方法【方法的重载】
```

#### 7. 方法的重载

![](.\img\多个构造方法.png)

```
总结:
	1. 所有的方法名字都是一致的！！！
	2. 所有的方法参数都是不一样的，参数类型，参数个数，参数顺序不同。返回值不同不构成重载
	3. 同一个类内！！！

这就是方法的重载！！！
优点:
	1. 简化了开发压力
	2. 简化了记忆压力
	3. 更快捷的调用方法，同时又满足了不同的情况！！！

规范：
	重载情况下，在同一个类内，不可以出现相同方法名和相同参数数据类型的方法！！！
	
基本原理:
	方法名一致的情况下，通过形式参数列表数据类型的不同来选择不同的方法执行。
```

#### 8. this关键字

##### 8.1 this关键字特征

```java
package com.qfedu.a_object;

class SingleDog {
	public SingleDog() {
		System.out.println("Constructor : " + this);
	}
	
	public void test() {
		System.out.println("Method Called : " + this);
	}
}
/*
 * this关键字特征:
 * this关键字表示调用当前方法的类对象，
 * 或者是当前构造方法中初始化的类对象
 */
public class Demo4 {
	public static void main(String[] args) {
		SingleDog singleDog = new SingleDog();
		
		System.out.println("Instance : " + singleDog);
		singleDog.test();
	}
}
```

##### 8.2 解决就近原则问题

```java
/**                                          
 * 使用String类型参数和int类型参数初始化类对象成员变量数据          
 *                                           
 * @param name String类型数据 用于初始化name属性        
 * @param age int类型数据 用于初始化age属性             
 */                                          
public Cat(String name, int age) {           
	name = name;                             
	age = age;                               
	System.out.println("带有两个参数的构造方法");       
}                                            
/*
我们期望使用比较直观的参数名方式，告知调用者这里需要的数据到底是什么？
但是会导致【就近原则】问题
	在构造方法中所有的name，age都会被看作是一个局部变量，而不是成员变量
期望:
	可以有一种参数方式告知编译器这里操作的是一个成员变量，而不是一个局部变量！！！
*/

--------------------------修改方式----------------------------------
 /**                                             
 * 使用String类型参数和int类型参数初始化类对象成员变量数据             
 *                                              
 * @param name String类型数据 用于初始化name属性           
 * @param age int类型数据 用于初始化age属性                
 */                                             
public Cat(String name, int age) {              
	/*                                          
	 * 使用this关键字明确告知编译器这里使用的是一个成员变量，而不是         
	 * 局部变量，解决就近原则问题                            
	 */                                         
	this.name = name;                           
	this.age = age;                                       
}                                               
```

#### 9. 成员变量和局部变量的对比

|  区别  |                           成员变量                           |               局部变量               |
| :----: | :----------------------------------------------------------: | :----------------------------------: |
|  作用  |               属性描述，描述当前类拥有哪些属性               |   在方法运行的过程中保存必要的数据   |
|  位置  |        成员变量定义在class大括号以内，其他大括号之外         |   在方法大括号或者代码块大括号以内   |
| 初始值 | 成员变量在没有被构造方法赋值的情况下，是对应数据类型的"零"值 | 未赋值不能参与除赋值之外的其他运算。 |
| 作用域 | 成员变量存储于类对象中，在内存的堆区，哪里持有当前类对象的空间首地址，作用域就在哪里 |        有且只在当前大括号以内        |
| 生存期 | 成员变量的生存期是随着类对象的创建而开始，当对象被JVM的GC销毁时，成员变量的生存期终止 |   从定义位置开始，到当前大括号结束   |

##### 【 "零"值】

```
new关键字申请内存空间，并且擦除的一干二净
对应每一个数据类型的"零"值
基本数据类型
	byte short int   0
	long             0L
	float            0.0F
	double           0.0
	char             '\0' ==> nul    即空字符
	boolean          false
引用数据类型
	全部为null
	Person person  null
	String str     null
	int[] arr      null
```

##### 【补充知识点 JVM的GC机制 简述】

![](.\img\图书管理员简述回收机制.png)

```
Java中内存管理制度GC就类似于图书管理员身份
	1. 在单位时间内，检查当前Java程序使用的内存中是否存在无主内存。
	2. 标记无主内存，多次标记的情况下，会将无主内存释放，归还内存空间。
好处:
	1. 让程序员管理内存更加方便。
	2. 内存管理是完全自动化
劣势:
	1. 内存回收效率不高。
	2. 内存管理出现泄漏问题。            
```

## 面向对象三大特征

#### 1. 面向对象的三大特征

```
封装，继承，多态
```

#### 2. 封装

##### 2.1 不局限于面对对象的封装

```
方法的封装
工具类的封装
框架的封装

需要拥有封装的思想！！！可以用于整合的知识点！！！

一段代码，你写了三遍 ==> 封装成方法
一堆方法，你用了三遍 ==> 封装成工具类
一个工具类，你使用了三遍 ==> 写好对应的注释，完成对应的API
一个类注释自己修改了三遍，==> 写成博客
```

##### 2.2 符合JavaBean规范的类封装过程

```
在Java中定义符合JavaBean规范的类有什么要求
	1. 所有的成员变量全部私有化 ==> private
	2. 必须提供一个无参数构造方法
	3. 要求使用private修饰的成员变量提供对应的操作方法 ==> Setter Getter
```

###### 2.2.1 private关键字

```
private关键字是一个权限修饰符
	private修饰的成员变量，成员方法，【构造方法】都是私有化内容，有且只能在类内使用，类外没有任何的操作权限！！！
```

```java
package com.qfedu.a_private;
/*
 * Private关键字使用
 */
class Dog {
	private String name;
	int age;
	char gender;
	
	public void testField() {
		// 类内可以直接使用私有化private修饰的成员变量
		name = "Bobo";
		test();
	}
	
	private void test() {
		System.out.println("烤羊排！！！");
	}
}


public class Demo1 {
	public static void main(String[] args) {
		Dog dog = new Dog();
		
		// 没有使用private约束的情况下，类外可以使用
		// 当成员变量使用private修饰之后，当前成员变量类外没有操作权限
		// The field Dog.name is not visible
		// dog.name = "王可可";
		dog.age = 5;
		dog.gender = '雌';
		
		// The method test() from the type Dog is not visible
		// 使用private修饰的方法类外不能使用
		// dog.test();
	}
}
```

###### 2.2.2 Setter和Getter方法

```
private修饰的成员变量类外是没有任何操作权限，这里需要提供对应的操作方法，setter和getter方法

Setter方法格式:
	public void set成员变量名(对应成员变量的数据类型 成员变量的形式参数) {
		this.成员变量名 = 成员变量的形式参数;
	}
	
	例如:
	public void setName(String name) {
		this.name = name;
	}

Getter方法格式:
	public 对应成员变量的数据类型 get成员变量名() {
		return 成员变量;
	}
	
	例如:
	public String getName() {
		return name;
	}

如果成员变量是一个boolean类型，Getter方法有所不同
	boolean flag;
	格式:
		public boolean isFlag() {
			return flag;
		}
```

```java
package com.qfedu.a_private;

/*
 * 按照JavaBean规范完成自定义类
 */
class Cat {
	// 所有的成员变量全部私有化
	private String name;
	private int age;
	private char gender;
	private boolean married;
	
	// 根据实际需要完成对应Constructor，Setter， Getter
	
	
	// 【重点】要有对应的无参构造方法
	public Cat() {
		super();
	}
    
    // 对应的构造方法
	public Cat(String name, int age, char gender, boolean married) {
		super();
		this.name = name;
		this.age = age;
		this.gender = gender;
		this.married = married;
	}

	// Setter和Getter方法
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getAge() {
		return age;
	}

	public void setAge(int age) {
		this.age = age;
	}

	public char getGender() {
		return gender;
	}

	public void setGender(char gender) {
		this.gender = gender;
	}

	public boolean isMarried() {
		return married;
	}

	public void setMarried(boolean married) {
		this.married = married;
	}
	
}

public class Demo2 {
	public static void main(String[] args) {
		Cat cat = new Cat();
		
		cat.setName("八九");
		cat.setAge(1);
		cat.setGender('雌');
		cat.setMarried(false);
		
		System.out.println("Name:" + cat.getName());
		System.out.println("Age:" + cat.getAge());
		System.out.println("Gender:" + cat.getGender());
		System.out.println("Married:" + cat.isMarried());
	}
}
```

#### 3. 多类合作

**什么是多类合作**

```
在开发中，除了基本数据类型，大多数情况下，都是类对象操作数据，作为
	1. 方法的参数是类对象
	2. 类定义时成员变量数据类型
```

 **方法的参数【电脑和维修店案例】**

```
需求:
	电脑类
		属性：
			屏幕是否OK boolean ok;
		方法：
			电脑屏幕如果是OK的，可以使用，如果不OK，无法观看
			
	维修店类
		属性：
			店址
			电话
			店名
		方法：
			修电脑（参数是一个电脑类对象）
```

 **成员变量的数据类型为自定义类型**

```
汽车
	发动机
	轮胎
class Car 
	这里需要的数据类型是我们的自定义复合数据类型，自定义类！！！
	Engine engine
	Tyre tyre

发动机也需要一个类
class Engine 
	型号
	排量

轮胎也需要一个类
class Tyre
	型号
	尺寸
```

#### 4. 匿名对象

```java
匿名对象:
	new 构造方法(必要的参数);
匿名对象的用途:
	1. 使用匿名对象直接调用类内的成员方法
	2. 匿名对象直接作为方法的参数
注意:
	使用匿名对象不要操作成员变量，有可能是有去无回
优势：
	1. 阅后即焚，匿名对象在使用之后 立即被JVM GC收回
	2. 解决内存空间，提高效率，简化代码书写
```

#### 5. 继承

##### 5.1 Java中实现继承的方式

```
继承使用的关键字
	extends
格式:
	class A extends B {
	
	}
	A类是B类的一个子类
	B类是A类的唯一父类
	Java中的继承是一个单继承模式

基本要求:
	1. 子类继承父类之后，可以使用父类的非私有化成员变量和成员方法
	2. 子类不能继承得到父类的【私有化成员】。
```

##### 52 继承的问题

###### 5.3.1 父类的构造方法被调用

![](C:/Users/%E4%BD%95%E4%BD%B3%E4%BC%9F/Desktop/java/%E5%8D%83%E9%94%8Bjava%E9%80%86%E6%88%98%E7%8F%AD/Day09-%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E4%B8%89%E5%A4%A7%E7%89%B9%E5%BE%81/img/%E7%BB%A7%E6%89%BF%E7%88%B6%E7%B1%BB%E5%92%8C%E5%AD%90%E7%B1%BB%E5%86%85%E5%AD%98%E5%88%86%E6%9E%90%E5%9B%BE.png)

###### 5.3.2 为什么会自动执行父类的无参数构造方法

```
这里有一个super关键字
	1. 调用父类成员方法和成员变量的关键字。
		[解决就近原则问题]
	2. 用于显式调用父类的构造方法。

super关键字调用父类的构造方法
	super(实际参数);
	Java编译器会根据实际参数的数据类型，参数顺序，选择对应的父类构造方法执行，初始化父类的成员空间，方法重载机制。
	
特征：
	1. 如果没有显式调用父类的构造方法，默认Java编译器会调用无参父类构造方法使用
	2. 根据数据类型选择对应方法
	3. super调用构造方法，必须在当前构造方法的第一行
```

#### 6. 继承

##### 6.1 继承带来的问题

```
	子类可以通过继承获取到父类中非私有化方法，但是父类的方法，不一定满足子类的情况。
	这里不可能通过重新定义类似的方法名，完成子类特定的符合要求的方法。
```

##### 6.2 方法的重写

```
	解决的问题就是在开发中，父类的方法不适用于子类情况，子类可以重写父类的方法，完成自定义的方法使用
    重写之后，在没有增加新的方法名的情况下，重写方法体内容，让方法满足子类，降低了开发压力，提高了效率。

@Override
	严格格式检查
	要求重写方法的和父类中的方法，声明完成一致，包括【返回值类型，方法名和形式参数列表】
```

##### 6.3 重写和继承带来的问题

```
	子类继承父类可以直接使用父类的方法，但是在这种情况下我们可以发现父类的方法是一定不能在子类中使用的，但是又没有一个强制要求。                                                    
需求：                             
	强制要求子类重写父类的方法，从语法角度约束      
```

##### 6.4 abstract关键字

```
abstract修饰的方法
	要求子类强制重写！！！	                                   
	
abstract使用总结:
	1. abstract修饰的方法没有方法体
	2. abstract修饰的方法必须定义在abstract修饰的类内或者interface接口内
	3. 一个普通类【非abstract】修饰的类，继承了一个abstract类，那么必须实现（重写）在abstract类内的所有abstract方法，强制要求
	4. 如果一个abstract A类继承另一个abstract B类，A类可以选择性的实现B类中abstract方法。
	5. abstract修饰的类内允许普通方法存在
	6. abstract修饰的类不能创建自己的类对象！！！
	【原因】
	abstract修饰的类内有可能存在abstract修饰的方法，而abstract修饰的方法是没有方法体的，如果说创建	     了abstract修饰类对应的对象，不能执行没有方法体的abstract方法
	7. 一个类内没有abstract修饰的方法，那么这个类定义成abstract类有意义吗？
		【没有必要的！！！无意义的！！！】
```

#### 7. final关键字

> final修饰的成员变量
> ​	**final修饰的成员变量定义时必须初始化，并且赋值之后无法修改**，一般用于类内带有名字的常量使用
> final修饰的成员方法
> ​	**final修饰的成员方法不能被子类重写，为最终方法**，可以用于一些安全性方法的定义
> final修饰的局部变量
> ​	**final修饰的局部变量一旦被赋值，不能修改！**
> final修饰的类
> ​	**final修饰的类没有子类，不能被继承。**
> ​	**abstract修饰的类不能被final修饰。**



#### 8. static关键字【重点】

##### 8.1 static修饰成员变量

###### 8.1.1 static修饰成员变量的需求

![](C:/Users/%E4%BD%95%E4%BD%B3%E4%BC%9F/Desktop/java/%E5%8D%83%E9%94%8Bjava%E9%80%86%E6%88%98%E7%8F%AD/Day10-%E9%9D%A2%E5%90%91%E5%AF%B9%E8%B1%A1%E4%B8%89%E5%A4%A7%E7%89%B9%E5%BE%81%E7%AC%AC%E4%BA%8C%E8%AE%B2/img/static%E4%BF%AE%E9%A5%B0%E6%88%90%E5%91%98%E5%8F%98%E9%87%8F%E7%9A%84%E9%9C%80%E6%B1%82.png)



```
总结:
	1. 公共区域存放
	2. 共享性使用
	3. 和对象无关
	4. 一处修改，处处受到影响。
```

###### 8.1.2 静态成员变量使用注意事项

```
1. 静态成员变量是使用static修饰的成员变量，定义在内存的【数据区】
2. 静态成员变量不推荐使用类对象调用，会提示警告
	The static field SingleDog.info should be accessed in a static way
	使用static修饰的SingleDog类内的info成员变量，应该通过静态方式访问
3. 静态成员变量使用类名调用是没有任何的问题。【墙裂推荐方式】
4. 在代码中没有创建对象时，可以通过类名直接使用静态成员变量，和【对象无关】
5. 代码中对象已经被JVM的GC销毁时，依然可以通过类名调用静态成员变量，和【对象无关】
6. 不管通过哪一种方式调用静态成员变量，修改对应的静态成员变量数据，所有使用到当前静态成员变量的位置，都会受到影响。
```

###### 8.1.3 为什么静态成员变量和对象无关

```
1. 从内存角度出发分析
	静态成员变量是保存在内存的数据区
	类对象占用的实际内存空间是在内存的堆区
	这两个区域是完全不同的，所有可以说静态成员变量和对象没有关系 【没有对象】

2. 从静态成员变量以及类对象生命周期来分析
	静态成员变量是随着类文件(.class) 字节码文件的加载过程中，直接定义在内存的数据区。静态成员变量从程序运行开始就已经存在。
	类对象是在代码的运行过程中，有可能被创建的。程序的运行过中，有可能会被JVM的CG垃圾回收机制销毁，程序在退出之前一定会销毁掉当前Java程序使用到的所有内存。
	静态成员变量在程序退出之后，才会销毁
	
	静态成员变量的生命周期是从程序开始，到程序结束
	类对象只是从创建开始，而且随时有可能被JVM的GC销毁
	生命周期不在同一个时间线上，所以静态成员变量和类对象无关，【没有对象】
```

##### 8.2 static修饰成员方法

######  静态成员方法的格式

```
异常熟悉的格式
	public static 返回值类型 方法名(形式参数列表) {
	
	}
```

###### 静态成员方法注意事项

```
1. 静态成员方法推荐使用静态方式调用，通过类名调用【墙裂推荐的】
	不推荐使用类对象调用，因为【没有对象】
2. 静态成员方法中不能使用非静态成员 ==> (非静态成员方法和非静态成员变量)	
	因为【没有对象】
3. 静态成员方法中不能使用this关键字
	因为【没有对象】
4. 静态成员方法中可以使用类内的其他静态成员【难兄难弟】
5. 静态成员方法中可以通过new 构造方法创建对象
	单身狗可以找对象
	不能挖墙脚但是能自己找
```

######  静态成员方法特征解释

```
1. 静态成员方法加载时间问题
	静态成员方法是随着.class字节码文件的加载而直接定义在内存的【方法区】，而且此时的静态成员方法已经可以直接运行。可以通过类名直接调用，而此时没有对象存在。【没有对象】
	
2. 为什么静态成员方法不能使用非静态成员
	非静态成员变量和非静态成员方法时需要类对象调用的，在静态成员方法中，是可以通过类名直接执行的，而此时是【没有对象】的。

3. 为什么静态成员方法不能使用this关键字
	this关键字表示的是调用当前方法的类对象，但是静态成员方法可以通过类名调用，this不能代表类名，同时也是【没有对象】

4. 静态成员方法可以使用其他静态成员
	生命周期一致，调用方式一致
```

###### static修饰静态成员方法用途

> 特征：
> ​	1. 摆脱类对象，效率高，节约内存空间，提高开发效率
> ​	2. 类内成员变量和成员方法都不可以使用，但是不影响使用外来数据 ( 函数参数 )。
> ​	3. 静态成员方法通常用于工具类的封装使用。



##### 8.3 类变量和类方法

```
类变量 ==> 静态成员变量
类方法 ==> 静态成员方法
类成员 ==> 静态成员变量和静态成员方法

面试题
	类方法中是否可以使用成员变量?
	类方法可以使用当前类内的静态成员变量，但是不允许使用非静态成员变量
```

##### 8.4 静态代码块

###### 补充知识点 代码块

```
构造代码块
	初始化当前类的所有类对象，只要调用构造方法，一定会执行对应的构造代码块
	
静态代码块
	初始化程序,只要类文件加载,静态代码块中所有内容全部执行
	
局部代码块
	提高效率，解决内存，让JVM回收内存的效率提升。
	
	for () {
		{
			int num
		}
	}
```

###### static修饰静态代码块

```
特征:
	1. static修饰的静态代码块，不能使用this关键字，不能使用类内的非静态成员
	2. static修饰的静态代码块，可以使用类内的其他静态成员‘
	3. static修饰的静态代码块中，定义的变量都是局部变量，静态代码块。首先是是一个代码块，拥有代码块的特征，其次才是通过static修饰之后，可以随着类文件的加载直接运行，有且只运行一次
	static {
		int num = 10;
		num = 20;
	}
```

#### 9. 接口

##### 9.1 Java中接口使用

```
格式：
	interface 接口名 {
		成员变量
		成员方法
	}

类【遵从】接口
	implements
	class 类名 implements 接口 {
	
	}

成员变量 缺省属性是public static final
缺省属性 public abstract
```

##### 9.2 接口使用总结

```
1. 接口中的
	成员变量缺省属性 public static final
	成员方法缺省属性 public abstract

2. 一个非abstract类遵从interface接口，需要强制完成接口中所有缺省属性为public abstract的成员方法

3. 接口和接口之间，允许使用extends关键字继承，并且允许一个接口，继承多个接口
	interface A extends B, C
	生活中: 协议直接的向下兼容问题

4. 接口中可以使用default关键字修饰方法，default方法拥有方法体，可以认为是非强制实现方法，不要求遵从接口的非abstract强制实现，JDK1.8新特征
```

#### 10. 多态

```
多态存在的条件:
1、有继承关系　　
2、子类重写父类方法　　
3、父类引用指向子类对象
如：Parent p = new Child();
多态定义
	父类的引用指向子类的对象或者说接口的引用指向遵从接口的类对象

作用:
	1. 拓宽方法的参数范围
		例如:
			方法参数为Parent类型，可以传入Parent类型本身，或者去子类对象Child()都可以
			方法参数为接口类型,只要是直接或者间接遵从接口的类对象可以作为方法的参数传入
	2. 拓宽方法的返回值范围
	3. 简化代码开发，提高开发效率，整合数据类型
```













































