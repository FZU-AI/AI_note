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
