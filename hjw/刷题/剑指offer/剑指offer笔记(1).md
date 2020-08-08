# 数组

## 二维数组中查找

**题目描述**

> 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**示例**

二维数组如下：每行每列都是递增排序序列，如果查找数字7，则返回true，如果查找数字5，则返回false

|  1   | 2    | 8    | 9    |
| :--: | ---- | :--- | ---- |
|  2   | 4    | 9    | 12   |
|  4   | 7    | 10   | 13   |
|  6   | 8    | 11   | 15   |

**解题思路**

方法一：暴力解法，循环遍历整个二维数组，直到找到目标整数或者遍历到最后一个元素。时间复杂度高，没有利用题目所给的每行每列是递增序列的条件。

方法二：从右上角开始进行元素比较。如果该数字等于要查找的数字，查找过程结束；如果该数字大于要查找的数字，则剔除这个数字所在的列；如果该数字小于要查找的数字，剔除这个数字所在的行。

![二维数组的查找](C:\Users\何佳伟\Desktop\刷题\剑指offer\image\二维数组的查找.png)

**题解代码**

从右上角开始查找

```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        bool result = false;
        int rows = array.size();            //二维数组的行数
        int columns = array[0].size();        //二维数组的列数
        int i=0;
        int j = columns-1;
        //从矩阵的右上角开始查找
        while(!array.empty() && i<rows && j>=0){
            if(array[i][j] == target){     
                result = true;
                break;
            }else if(array[i][j]>target){    //缩小查找范围，剔除所在列
                j--;
            }else{                           //缩小查找范围，剔除所在行
                i++;
            }
        }
        return result;
    }
};
```

从左下角开始查找（同理）

```c++
class Solution {
public:
    bool Find(int target, vector<vector<int> > array) {
        bool result = false;
        int rows = array.size();            //二维数组的行数
        int columns = array[0].size();        //二维数组的列数
        int i= rows-1;
        int j = 0;
        //从矩阵的左下角开始查找
        while(!array.empty() && j<columns && i>=0){
            if(array[i][j] == target){     
                result = true;
                break;
            }else if(array[i][j]>target){    
                i--;
            }else{                           
                j++;
            }
        }
        return result;
    }
};
```

---

---

## 和为S的两个数字

**题目描述**

> 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

**示例**

> 递增序列为：1，2，4，5，7，9，12             S为6，则返回2，4

**解题思路**

方法一：暴力循环，通过两个循环遍历整个数组

方法二：双指针，由于整个数组是递增排序的，所以设置两个指针head，tail 分别指向有序数组的头和尾。并通过头尾指针所指元素之和sum与S进行比较，如果相等则说明已找到，直接返回下标即可。如果sum>S,则tail--，使tail指向一个较小的元素；如果sum<S,则head++，使head指向一个较大的元素。循环下去直到sum =S。

![和为S的两个数](C:\Users\何佳伟\Desktop\刷题\剑指offer\image\和为S的两个数.png)

---

**解题代码**

```c++

```

---

---

## 数组中重复的数字

**题目描述**

> 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

**题目分析**

> 根据题意，给定的数组是有数字是重复的，并且有多个数字是重复的，要求是找出数组中任意一个重复的数字，根据给定示例，我默认为找到第一个重复的数字。

**解题思路**

> 设计到重复的问题，要联想到集合这种数据结构，c++中的set即为集合。在set中每个元素的值都唯一，而且系统能根据元素的值自动进行排序，所以我们的整理思路是遍历数组元素，先判断这个元素是否在集合中存在，如果存在该元素即为重复元素，如果不存在，则将该元素存入set中。

**解题代码**

```c++
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        set<int> set_int;                         //创建一个空集合
        bool res = false;                        
        for(int i=0 ; i < length ; i++){          //遍历数组
            if(set_int.count(numbers[i]) == 1){   //判断元素是否存在于集合中
                *duplication = numbers[i];   
                res = true;
                break;
            }else{                                //如不存在则将元素存入集合中
                set_int.insert(numbers[i]);       
            }
        }
        return res;
    }
};
```

---

---

## 数字在排序数组中出现的次数

**题目描述**

> 统计一个数字k在排序数组data中出现的次数。默认升序

**解题思路**

方法一：由于是统计元素个数，所以把数组的元素都依次存入multiset中，multiset与set的不同在于允许存在重复元素，最后通过count方法返回要查询的元素个数即可

方法二：一开始的思路是先使用二分法找到k，然后从k开始向两边统计k的个数，但统计的这个时间复杂度达到了O(n)，导致整个算法的复杂度O(nlogn)，由于数组是有序的，所有值相同的元素肯定是相邻的，而通过两次二分查找，分别找到第一个k和最后一个k，可以使时间复杂度减少为O(logn)  分别找到第一个值为 k 和最后一个值为 k  的位置，那么就找到了所有值为 k 的元素了，下标相减即为出现次数。

**解题代码**

方法一：利用multiset求解

```c++
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        multiset<int> data_set;
        for(int i=0 ; i<data.size() ; i++){
            data_set.insert(data[i]);
        }
        return data_set.count(k);
    }
};
```

方法二：通过找第一个值为 k 和最后一个值为 k  的位置来求解(使用STL)

```c++
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        vector<int>::iterator it = lower_bound(data.begin(), data.end(), k); 
        vector<int>::iterator it2 = upper_bound(data.begin(), data.end(), k);
        return it2 - it;
    }
};
```

方法二：手写二分查找

```c++
class Solution {
private:
    int findFirstK(const vector<int> &data, int k){  //二分法查找第一个k
        int l = 0, r = data.size() - 1;
        int mid = (l + r) / 2;
        while(l <= r){
            if(data[mid] > k){
                r = mid - 1;
            } else if(data[mid] < k){
                l = mid + 1;
            } else if(mid - 1 >= 0 && data[mid - 1] == k){ //不是第一个，继续往前找
                r = mid - 1;
            } else{
                return mid;
            }
            mid = (l + r) /2;
        }
        return -1;
    }
    int findLastK(const vector<int> &data, int k){    //二分法查找最后一个k
        int l = 0, r = data.size() - 1;
        int mid = (l + r) / 2;
        while(l <= r){
            if(data[mid] > k){
                r = mid - 1;
            } else if(data[mid] < k){
                l = mid + 1;
            } else if(mid + 1 < data.size() && data[mid + 1] == k){//不是最后一个，继续往后找
                l = mid + 1;
            } else{
                return mid;
            }
            mid = (l + r) / 2;
        }
        return -1;
    }
public:
    int GetNumberOfK(vector<int> data ,int k) {
        int first = findFirstK(data, k);
        int last = findLastK(data, k);
        if(first == -1 || last == -1){
            return 0;
        }
        return last - first + 1;
    }
};

```

---

---

## 数组中出现次数超过一半的元素

**题目描述**

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

**题目分析**

**方法一：** 先将数组元素存入multiset中，遍历整个集合元素，通过count方法返回每个元素的个数，如果存在个数大于一半的元素则说明找到该元素，并返回该元素，不存在则返回0

**方法二：**将整个数组排序后，如存在出现次数超过一半的元素，那么这个元素一定出现在排序数组后的中间位置。先排序后，找到中间位置的元素，再遍历数组统计中间位置元素出现的次数即可，如果大于数组长度一般则输出，否则输出0。

**方法三：**  采用阵地攻守的思想：第一个数字作为第一个士兵，守阵地；count = 1；遇到相同元素，count++;
遇到不相同元素，即为敌人，同归于尽,count--；当遇到count为0的情况，又以新的i值作为守阵地的士兵，继续下去，到最后还留在阵地上的士兵，有可能是主元素。再遍历数组统计中间位置元素出现的次数即可如果大于数组长度一般则输出，否则输出0。  

**解题代码**

**方法一：**利用multiset实现

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        multiset<int> set;
        int res = 0;                                    //初始值设为0
        for(int i=0 ; i < numbers.size() ; i++){        //将元素依次存入multiset中
            set.insert(numbers[i]);
        }
        for(int i=0 ; i < numbers.size() ; i++){
            if(set.count(numbers[i]) > numbers.size()/2){//判断是否存在出现次数超过一半的数字
                res = numbers[i];
                break;
            }
        }
        return res;
    }
};
```

**方法二：**先排序，再判断中间元素出现的次数

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        sort(numbers.begin(),numbers.end());    //将numbers排序
        int length = numbers.size();            //获取数组长度
        int temp = numbers[length / 2];         //获取排序后中间位置的元素
        int count = 0;
        for(int i=0 ; i<length ; i++){          //统计中间位置的元素出现的次数
            if(numbers[i] == temp){
                count++;
            }
        }
        if(count > length / 2){                //出现次数与数组长度的一半进行比较
            return temp;
        }else{
            return 0;
        }
    }
};
```

**方法三：**阵地攻守法

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        int length = numbers.size();         //获取数组长度
        int count = 1;                       //第一个元素作为士兵守阵地
        int k = numbers[0];
        for(int i=1 ; i<length ; i++){       //从第二个元素开始判断
            if(count == 0){                //当遇到count为0的情况，又以新的i值作为守阵地的士兵
                k = numbers[i];
            }else{
                if(numbers[i] == k){        //相同元素，士兵人数加一
                    count++;
                }else{                      //不同元素，互为敌人，士兵人数-1
                    count--;
                }
            }
        }
        count = 0;                         //此时k是留守阵地的人，可能为所求元素
        for(int i=0 ; i<length ; i++){     //统计k在数组中出现的次数
            if(numbers[i] == k){
                count++;
            }
        }
        if(count > length / 2){            //出现次数与数组长度的一半进行比较
            return k;
        }else{
            return 0;
        }
    }
};
```

---

---

---

## 数组中只出现一次的数字

**题目描述**

> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

**解题思路**





**解题代码**

---

---

---



## 构建乘积数组

**题目描述**

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）

**解题思路**

> 找规律题，根据题意，数组中B的元素B[i]等于数组A的所有元素相乘（A[i]换成1），也直接规定了B数组中的第一个元素和最后一个元素的定义。根据题意，B中的元素就等于下列矩阵中每一行的A元素的成绩。由于不可以使用除法，**下三角用连乘可以很容求得，上三角，从下向上也是连乘**。所以先算下三角中的连乘，即我们先算出B[i]中的一部分，然后倒过来按上三角中的分布规律，把另一部分也乘进去。

![构建乘积数组](C:\Users\何佳伟\Desktop\刷题\剑指offer\image\构建乘积数组.png)



**解题代码**

```c++
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int length = A.size();
        vector<int> B(length);
        B[0] = 1;
        for(int i=1;i<length;i++){     //计算下三角
            B[i] = B[i-1] * A[i-1];
        }
        int temp = 1;                  //计算上三角
        for(int j=length-2;j>=0;j--){
            temp *= A[j+1];
            B[j] *= temp;
        }
        return B;
    }
};
```

---

---

---

## 滑动窗口的最大值

**题目描述**

> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

**题目分析**

> 根据题意，如输入的数组长度为 s , 给定的滑动窗口大为 k ,那么一共就存在 (s-k+1) 个滑动窗口，分别求这些窗口的最大值即可。

**解题代码**

```c++
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        vector<int> res;
        if(num.empty() || size == 0){        //判断数组是否为空或者窗口是否为空
            return res;
        }
        int length = num.size();
        int i = 0;                          //滑动窗口的起始位置
        int j = size - 1;                   //滑动窗口的终止位置
        
        while(j<length){                    //滑动窗口一直向前移动
            res.push_back(max(num, i, j));   //存入每个窗口的最大值
            i++;
            j++;
        }
        return res;
    }
private:
    int max(const vector<int>& num,int i,int j){    //返回num数组中[i,j]范围内的最大值
        int max = num[i];
        for(int k=i;k<=j;k++){
            if(num[k] > max){
                max = num[k];
            }
        }
        return max;
    }
};
```

---

---

---

## 调整数组顺序使奇数位于偶数前面

**题目描述**

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**题目分析**

方法一：根据题目要求，前半部分为奇数，后半部分为偶数，并保持相对顺序不变，遍历两次数组，第一次遍历找出所有的奇数，并存放到一个数组中，第二次遍历找出所有的偶数即可。时间复杂度O(n) , 但是使用了新的数组，空间复杂度较大。

方法二：冒泡排序的思想。遍历整个数组，将奇数元素往前进行交换，直到前面都是奇数元素位置，用一个变量记录 k 已经摆好的奇数的位置，并实时更新。

![奇数位于偶数前](C:\Users\何佳伟\Desktop\刷题\剑指offer\image\奇数位于偶数前.png)



**解题代码**

方法一：空间换时间

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int length = array.size();
        vector<int> res;                
        for(int i=0 ; i<length ; i++){        //找出所有的奇数存入数组
            if( array[i] % 2 == 1){
                res.push_back(array[i]);
            }
        }
        for(int i=0 ; i<length ; i++){		//找出所有的偶数存入数组
            if( array[i] % 2 == 0){
                res.push_back(array[i]);
            }
        }
        array.swap(res);                  
    }
};
```

方法二：冒泡排序的思维

```c++
class Solution {
public:
    void reOrderArray(vector<int> &array) {
        int length = array.size();
        int k = 0;                 //用于记录前半部分已经摆好的奇数的位置
        int temp = 0;              //临时变量，用于元素交换
        for(int i=0 ; i<length ; i++){
            if( array[i] % 2 == 1){
                int j = i;
                while( j > k){     //一直向前交换
                    temp = array[j];
                    array[j] = array[j-1];
                    array[j-1] = temp;
                    j--;
                }
                k++;             //新的奇数已经到达了指定位置了
            }
        }
    }
};
```

---

---

---

## 连续子数组的最大和

**题目描述**

> HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

**题目分析**

方法一：动态规划。用dp[i]表示以元素array[i]**结尾**的最大连续子数组和. 

状态转移方程为dp[i] = max{dp[i-1]+array[i],array[i]}. 

以{6,-3,-2,7,-15,1,2,2}为例。

dp[0] = 6;

dp[1] = max{ 6-3 , -3 } = 3

dp[2] = max{ 3-2 , -2 } = 1

dp[3] = max{1+7, 7} = 8

以此类推，最后返回dp数组中的最大值即可

---

**解题代码**

```c++
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        //状态转移方程：dp[i] = max{dp[i-1]+array[i],array[i]}.
        int length = array.size();
        vector<int> dp(length);
        dp[0] = array[0];
        int res = dp[0];            //用于记录整个dp数组中的最大值
        for(int i=1 ; i<length ; i++){
            dp[i] = max(dp[i-1]+array[i],array[i]);
            res = max(res, dp[i]);            //更新最大值
        }
        return res;
    }
};
```

----

---

---

