*集合是java中数据结构的具体实现**

- collection接口
  - list接口
    - Arraylist
    - LinkedList
    - ArrayDeque
  - Set接口
    - HashSet
    - TreeSet
    - EnumSet
    - LinkedHashSet
- Map接口
  - HashMap
  - TreeMap
  - LinkedHashMap

---

---

​                                                             **Java库中的具体集合**

| 集合类型      | 描述                                           |
| ------------- | ---------------------------------------------- |
| ArrayList     | 可以动态增长缩减的索引序列                     |
| LinkedList    | 可以在任何位置进行搞笑插入和删除操作的有序序列 |
| ArrayDeque    | 用循环数组实现的双端队列                       |
| HashSet       | 没有重复元素的无序集合                         |
| TreeSet       | 有序集                                         |
| EnumSet       | 包含枚举类型值的集                             |
| LinkedHashSet | 可以记住元素插入次序的集                       |
| PriorityQueue | 可以高效删除最小元素的集合                     |
| HashMap       | 存储键/值关联的数据结构                        |
| LinkedHashMap | 可以记住键/值项添加次序的映射表                |
| WeakHashMap   | 其值无用后可以被垃圾回收器回收的映射表         |
| IdentityMap   | 用 == ，而不是用equal比较键值的映射表          |

---

# Collection 接口

Collection 是所有单列集合的父接口，定义了单列集合（List和Set）通用的方法，可以操作所有单列集合

- public boolean add (E  e) : 把给定的对象添加到当前集合中
- public void clear () : 清空集合中所有的元素
- public boolean remove (E  e) : 把给定的对象从当前集合中删除
- public boolean contains(E  e) : 判断当前集合中是否包含给定的对象
- public boolean isEmoty() ：判断当前集合是否为空
- public int size() : 返回集合中元素的个数
- public boolean addAll(Collection<? extends E> c)  : 向集合中添加多个元素

---

---

# List接口

List接口继承与Collection接口

- List集合中允许出现重复元素，
- 所有元素以线性方式进行存储，
- 可以通过索引访问指定元素
- 元素有序，即元素存入顺序和删除顺序保持一致

List接口常用方法

- public boolean add (E  e) : 把给定的对象添加到当前列表中的末尾
- public boolean add (int index ,E  e) : 把给定的对象添加到当前列表中的指定位置
- public E get(int index) : 返回指定位置的元素
- public E set(int index ,E  e) : 用指定元素替换掉列表中指定位置的元素

---

List集合的遍历方法

- 普通的for循环
- 使用迭代器进行遍历
- 使用增强for循环

---

## ArrayLsit

- ArrayList是List接口的可调整大小的数组实现
- 每个ArrayList实例有一个容量。容量是用于存储列表中元素的数组的大小。它总是至少和列表大小一样大。为元素添加到ArrayList，它的容量是自动增加。
- 底层是通过数组实现的
- 此实现不同步，意味着多线程，速度较快（vector是同步的，即单线程，速度较慢，被ArrayList逐渐取代）
- 查询快，增删元素慢

底层的动态的实现是通过复制原数组，得到新的数组，再修改新的数组的长度。

---

## LinkedList

- 底层为链表结构（查询慢，增删快）
- 此实现不同步，意味着多线程，速度较快
- 含大量对首尾元素的操作

**常用接口方法**

| 方法               | 描述                                                         |
| ------------------ | :----------------------------------------------------------- |
| void addFirst(E e) | 在此列表的**开始**处插入指定的元素                           |
| void push(E e)     | 在此列表的**开始**处插入指定的元素,等价于addFirst()          |
| void addLast(E e)  | 在此列表的**结尾**处插入指定的元素。等价于add(E e)           |
| E getFirst()       | 返回此列表中的第一个元素。                                   |
| E getLast()        | 返回此列表中的最后一个元素。                                 |
| E pop()            | 从这个列表所表示的堆栈中（开始处）弹出一个元素,并返回        |
| E removeLast()     | 移除并返回此列表中的最后一个元素。                           |
| E removeFirst()    | 移除并返回此列表中的第一个元素,等价于pop（）                 |
| E peekFirst()      | 检索，但不删除该列表的第一个元素，或返回 null如果这个列表是空的。 |
| E peekLast()       | 检索，但不删除该列表的最后一个元素，或返回 null如果这个列表是空的。 |
| object clone()     | 返回该 LinkedList浅拷贝                                      |



# Set接口

继承Collection接口

- 无索引，意味着不能使用含索引的方法，不能使用普通的for循环
- 元素不允许重复出现



## HashSet

- 继承于Set
- 底层是哈希表结构，查询速度非常快
- 无序集合，存储速度和取出元素的顺序可能不同
- 不允许重复元素出现
- 实现不同步，多线程

---

**哈希值**：十进制的整数，由系统随机给出，是一种逻辑地址。hashcode() 返回该对象的哈希码值

> 1：Object类的hashCode.返回对象的内存地址经过处理后的结构，由于每个对象的内存地址都不一样，所以哈希码也不一样。
>
> 2：String类的hashCode.根据String类包含的字符串的内容，根据一种特殊算法返回哈希码，只要字符串内容相同，返回的哈希码也相同。**如果其哈希码相等，则这两个字符串是相等的**
> 3：Integer类，返回的哈希码就是Integer对象里所包含的那个整数的数值，例如Integer i1=new Integer(100),i1.hashCode的值就是100 。由此可见，2个一样大小的Integer对象，返回的哈希码也一样。

---

Set集合不允许元素重复的原理

> Set集合在调用add方法时的时候，add方法会调用元素的hashcode（）方法和equals方法判断元素是否重复。当向HashSet中添加元素的时候，首先计算元素的hashcode值，然后用这个（元素的hashcode）%（HashMap集合的大小）+1计算出这个元素的存储位置，如果这个位置位空，就将元素添加进去；如果不为空，则用equals方法比较元素是否相等，相等就不添加，否则找一个空位添加。

所以HashSet在存储自定义数据类型时，必须重写hashcode() 和 equals() 方法



----

---

## LinkedHashSet

- 继承于HashSet
- 底层是一个哈希表+链表，链表为了记录元素存入的顺序
- 所以存入顺序和存储顺序保持一致
- 实现不同步

---

# Map接口

- 映射键到值的对象。一张Map不能包含重复的键，每个键最多只能映射一个值，一一对应的关系
- 双列集合，一个元素包含两个值（键值对）

Entry对象：Map接口中的一个内部接口，用于记录键与值，是一个键值对对象

Entry的常用方法：



| 方法                | 描述                                   |
| ------------------- | -------------------------------------- |
| K getKey()          | 返回键值对的的键。                     |
| V getValue()        | 返回键值对的的值                       |
| V setValue(V value) | 用指定的值替换原来的值，返回被替换的值 |
| int hashCode()      | 返回此Map项的哈希代码值                |



---



Map接口的常用方法

| 方法                                           | 描述                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| public V put(K key,  V value)                  | j将映射存入Map中，如果映射以前包含一个键的映射，旧值将被指定的值替换，返回值为被替换的值 |
| public V get(Object key)                       | 返回指定的键映射的值,不存在则返回null                        |
| public boolean isEmpty()                       | 如果Map集合为空，则返回true                                  |
| public V remove(Object key)                    | 如果存在（可选操作），则从该Map中移除一个键的映射            |
| public Set<K> keySet()                         | 将map中的所有K取出来放到一个Set集合中                        |
| public abstract Set<Map.Entry<K,V>> entrySet() | 将map中的所有Entry对象取出来放到一个Set集合中                |

Map集合遍历方式

- 通过键找值得方式，先使用keySet()方法，得到所有的K，再通过get() 方法得到Map集合中所有的值
- 通过Entry对象完成遍历，先使用entrySet(),将多个Entry对象存入一个Set中，再遍历Set,获取没一个Entry对象，再对Entry逐一操作

## HashMap

- 继承于Map,底层是一个哈希表，查询速度快
- 无序集合，存储元素和取出元素顺序可能不同

HashMap在存储自定义数据类型时，必须重写hashcode() 和 equals() 方法,用于保证Key唯一

## LinkedHashMap

- 继承于HashMap
- 有序集合，存储元素和取出元素顺序相同，具有可预测的迭代顺序

## Hashtable

- 不允许存储null,即键与值都不能为空
- 单线程，速度慢
- 逐渐被HashMap所替代
- 其子类Properties 较为常用

