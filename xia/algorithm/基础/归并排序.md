## 归并排序
## 模板
```cpp
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];

    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}
```
## 归并排序
给定你一个长度为n的整数数列。  

请你使用归并排序对这个数列按照从小到大进行排序。  

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
## 题解
```cpp
//归并排序，时间复杂度O(nlogn)
#include <iostream>

using namespace std;

const int N = 1e6 + 10;

int a[N], tmp[N];

void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;

    merge_sort(q, l, mid), merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];
    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}

int main()
{
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);

    merge_sort(a, 0, n - 1);

    for (int i = 0; i < n; i ++ ) printf("%d ", a[i]);

    return 0;
}
```
## 逆序对的数量
给定一个长度为n的整数数列，请你计算数列中的逆序对的数量。  

逆序对的定义如下：对于数列的第 i 个和第 j 个元素，如果满足 i < j 且 a[i] > a[j]，则其为一个逆序对；否则不是。  

输入格式  
第一行包含整数n，表示数列的长度。  

第二行包含 n 个整数，表示整个数列。  

输出格式  
输出一个整数，表示逆序对的个数。  

数据范围  
1≤n≤100000  
输入样例：  
6  
2 3 4 5 6 1  
输出样例：  
5  
## 题解
```cpp
/*
归并排序，排序过程中，给出结果  
数字分布有三种情况，1.都在左区间 2.都在右区间 3.一左一右
*/
#include <iostream>

using namespace std;

const int N = 100010;

int n;
int q[N],tmp[N];

long long merge_sort(int l,int r)
{
    if (l >= r) return 0;
    int mid = l + r >> 1;
    long long res = merge_sort(l, mid) + merge_sort(mid + 1,r);
    //位于左区间和右区间，直接递归
    int k = 0,i = l ,j = mid + 1;
    while(i <= mid && j <= r)
    {
        if(q[i] <= q[j]) tmp[k ++] = q[i ++];
        else//位于左右两侧,且左侧数字大与右侧，该数字及至左区间末尾，均与右区间同一数字构成逆序数
        {
            tmp[k ++] = q[j ++];
            res += mid - i + 1;//左侧区间剩余数字
        }
    }
    while (i <= mid) tmp[k ++] = q[i ++];
    while (j <= r) tmp[k ++] = q[j ++];
    
    for(int i =l,j = 0;i <= r;i ++,j ++) q[i] = tmp[j];
    return res;
}

int main()
{
    scanf("%d",&n);
    for(int i = 0;i < n;i ++) scanf("%d",&q[i]);
    
    cout << merge_sort(0, n - 1) << endl;
    return 0;
}
```
