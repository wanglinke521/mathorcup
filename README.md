# mathorcup 大规模指纹检索
## 介绍
在生物特征识别领域，指纹作为最具独特性与持久性的生物特征之一，被广泛应用于身份识别,该解决方案是mathorcup数学建模A题方案。

```
.
│  code.py         #包含整个检索的过程
│  
└─数据
        TZ_同指.txt         #500人的相同手指的数据，每根手指采集两次
        TZ_同指200_乱序后_Data.txt   #200人的相同手指的数据，每根手指采集两次，但是没有标签
        TZ_异指.txt                 #10000人的手指的数据，乱序且没有标签
```

## Example to Run the Codes
```python
python code.py
```
