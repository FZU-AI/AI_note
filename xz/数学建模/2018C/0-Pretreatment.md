```python
import pandas as pd
import numpy as np
```


```python
excel = pd.read_excel('附件1.xlsx', 'Data', index_col=None, na_values=['NA'])
```


```python
excel.columns
```




    Index(['eventid', 'iyear', 'imonth', 'iday', 'approxdate', 'extended',
           'resolution', 'country', 'country_txt', 'region',
           ...
           'addnotes', 'scite1', 'scite2', 'scite3', 'dbsource', 'INT_LOG',
           'INT_IDEO', 'INT_MISC', 'INT_ANY', 'related'],
          dtype='object', length=135)




```python
pd.DataFrame(excel[['eventid','nkill','nwound','property','propextent','propvalue','targtype1','targtype2','targtype3','country','region','resolution','iyear','imonth','iday']])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eventid</th>
      <th>nkill</th>
      <th>nwound</th>
      <th>property</th>
      <th>propextent</th>
      <th>propvalue</th>
      <th>targtype1</th>
      <th>targtype2</th>
      <th>targtype3</th>
      <th>country</th>
      <th>region</th>
      <th>resolution</th>
      <th>iyear</th>
      <th>imonth</th>
      <th>iday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>199801010001</td>
      <td>104.0</td>
      <td>6.0</td>
      <td>-9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34</td>
      <td>11</td>
      <td>NaT</td>
      <td>1998</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>199801010002</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>167</td>
      <td>9</td>
      <td>NaT</td>
      <td>1998</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>199801010003</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>603</td>
      <td>8</td>
      <td>NaT</td>
      <td>1998</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>199801020001</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>95</td>
      <td>10</td>
      <td>NaT</td>
      <td>1998</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>199801020002</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>155</td>
      <td>10</td>
      <td>NaT</td>
      <td>1998</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114178</th>
      <td>201712310022</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>-9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>182</td>
      <td>11</td>
      <td>NaT</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>114179</th>
      <td>201712310029</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1</td>
      <td>4.0</td>
      <td>-99.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>200</td>
      <td>10</td>
      <td>NaT</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>114180</th>
      <td>201712310030</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>4.0</td>
      <td>-99.0</td>
      <td>14</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>160</td>
      <td>5</td>
      <td>NaT</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>114181</th>
      <td>201712310031</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>92</td>
      <td>6</td>
      <td>NaT</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>114182</th>
      <td>201712310032</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>160</td>
      <td>5</td>
      <td>NaT</td>
      <td>2017</td>
      <td>12</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
<p>114183 rows × 15 columns</p>
</div>




```python
import time
from datetime import datetime,date
# 方法一,利用python内置转换
tmp_time = date(excel['iyear'][0], excel['imonth'][0], excel['iday'][0]) # 生成datetime格式的时间
time.mktime(tmp_time.timetuple()) # 转化为时间戳
```




    883584000.0




```python
result = []
# 方法二,转换位pd.Timestamp
for i in range(excel.shape[0]):
#     print(excel['iyear'][i], excel['imonth'][i], excel['iday'][i])
    if excel['iday'][i]==0:
        excel['iday'][i] = 1
    result.append(pd.Timestamp(datetime(excel['iyear'][0], excel['imonth'][0], excel['iday'][0])))
result = pd.Series(result) # 存储所有的起始时间
result.head()
```

    <ipython-input-6-a8da161efb8a>:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      excel['iday'][i] = 1





    0   1998-01-01
    1   1998-01-01
    2   1998-01-01
    3   1998-01-01
    4   1998-01-01
    dtype: datetime64[ns]




```python
end = []
for i in range(excel.shape[0]):
    if pd.notnull(excel['resolution'][i]): # 相反的是isnull(time)
        excel['resolution'][i]=excel['resolution'][i]-result[i]
```

    <ipython-input-7-4bb3bc5e54ae>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      excel['resolution'][i]=excel['resolution'][i]-result[i]
    /home/fxz/anaconda3/envs/notebook/lib/python3.8/site-packages/pandas/core/indexing.py:670: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      iloc._setitem_with_indexer(indexer, value)



```python
# help(excel['resolution'][828])
for i in range(excel.shape[0]):
    if pd.notnull(excel['resolution'][i]): 
        if excel['resolution'][i].value < 0: # 设置timedelta为负的值为NaT
            excel['resolution'][i] = pd.NaT
```

    <ipython-input-8-bcbd64a0a015>:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      excel['resolution'][i] = pd.NaT



```python
excel.loc[69,'resolution'] # 使用这种方法可以避开SettingWithCopyWarning
```




    Timedelta('26 days 00:00:00')




```python
result = pd.DataFrame(excel[['nkill','nwound','propextent','propvalue','resolution']])
```


```python
# resolution 表示持续时间,如果当天解决,设置为86400/2
for i in range(result.shape[0]):
    if pd.notnull(result['resolution'][i]): 
        result.iloc[i,4] = result.iloc[i,4].value # 将持续时间,换成时间戳
    else:
        result.iloc[i,4] = 43200
```


```python
from sklearn.impute import SimpleImputer # 新建填充器来处理NaN
# 1. 创建Imputer器
imp = SimpleImputer(missing_values=np.nan, strategy='mean') # 使用平均值填充的方法

# 2. 使用fit_transform()函数即可完成缺失值填充了
result1=imp.fit_transform(result)
```




    array([[ 1.04000000e+02,  6.00000000e+00,  3.26518063e+00,
             3.75636192e+04,  4.32000000e+04],
           [ 0.00000000e+00,  3.00000000e+00,  3.26518063e+00,
             3.75636192e+04,  4.32000000e+04],
           [ 1.00000000e+00,  0.00000000e+00,  3.26518063e+00,
             3.75636192e+04,  4.32000000e+04],
           ...,
           [ 0.00000000e+00,  0.00000000e+00,  4.00000000e+00,
            -9.90000000e+01,  4.32000000e+04],
           [ 0.00000000e+00,  0.00000000e+00,  3.26518063e+00,
             3.75636192e+04,  4.32000000e+04],
           [ 0.00000000e+00,  0.00000000e+00,  3.26518063e+00,
             3.75636192e+04,  4.32000000e+04]])




```python
result1 = pd.DataFrame(result1) # 转换回DataFrame
result1.to_excel('result1.xlsx', sheet_name='Sheet1',index=False,header=False) # 去掉行和列的索引
```


```python
result1 = pd.read_excel('result1.xlsx', sheet_name='Sheet1', header = None )
result_norm = (result1 - result1.min()) / (result1.max() - result1.min()) # 归一化
result_norm.to_excel('result2.xlsx', sheet_name='Sheet1',index=False,header=False) # 写入结果
```


```python

```
