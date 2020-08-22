## quick_sort 模板
```cpp
void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);
}
```
### 快速排序
给定你一个长度为n的整数数列。  

请你使用快速排序对这个数列按照从小到大进行排序。  

并将排好序的数列按顺序输出。  

输入格式  
输入共两行，第一行包含整数 n。  

第二行包含 n 个整数（所有整数均在1~109范围内），表示整个数列。  
 
输出格式  
输出共一行，包含 n 个整数，表示排好序的数列。  
 
数据范围  
1≤n≤100000  
输入样例：  
5  
3 1 2 4 5  
输出样例：  
1 2 3 4 5  
### 题解
```cpp
//时间复杂度O(nlogn),普通快排
#include <iostream>

using namespace std;

const int N = 100010;

int a[N];

void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        while (q[++ i] < x);
        while (q[-- j] > x);
        if (i < j) swap(q[i], q[j]);
    }

    quick_sort(q, l, j);
    quick_sort(q, j + 1, r);
}

int main()
{
    int n;
    scanf("%d", &n);

    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);

    quick_sort(a, 0, n - 1);

    for (int i = 0; i < n; i ++ ) printf("%d ", a[i]);

    return 0;
}
```
### 第K小数
给定一个长度为n的整数数列，以及一个整数k，请用快速选择算法求出数列的第k小的数是多少。  

输入格式  
第一行包含两个整数 n 和 k。  

第二行包含 n 个整数（所有整数均在1~109范围内），表示整数数列。  

输出格式  
输出一个整数，表示数列的第k小数。  
 
数据范围  
1≤n≤100000,  
1≤k≤n  
输入样例：  
5 3  
2 4 1 5 3  
输出样例：  
3  
```cpp
//时间复杂度O(n)，快排修改
#include<iostream>

using namespace std;

const int N = 1e6 + 10;

int q[N];

int quick_sort(int l,int r ,int k)
{
    if(l == r) return q[l];
    
    int i = l - 1,j = r + 1,x = q[l + r >> 1];
    while(i < j)
    {
        while(q[++ i] < x);
        while(q[-- j] > x);
        if(i < j) swap(q[i],q[j]);
    }
    int el = j - l + 1;
    if(k <= el) return quick_sort(l, j, k); //判断第K小数，是否在左半区间内，在的话递归左区间；
    else return quick_sort(j + 1, r, k - el);
}

int main()
{
    int n, k;
    cin>> n >> k;
    
    for(int i = 0;i < n;i ++) scanf("%d",&q[i]);
    
    cout << quick_sort(0, n - 1,k) << endl;
    return 0;
}
```
