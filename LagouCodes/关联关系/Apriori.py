'''记得安装包
pip install efficient-apriori
'''
from efficient_apriori import apriori

# 设置数据集
data = [('尿布', '啤酒', '奶粉','洋葱'),
        ('尿布', '啤酒', '奶粉','洋葱'),
        ('尿布', '啤酒', '苹果','洋葱'),
        ('尿布', '啤酒', '苹果'),
        ('尿布', '啤酒', '奶粉'),
        ('尿布', '啤酒', '奶粉'),
        ('尿布', '啤酒', '苹果'),
        ('尿布', '啤酒', '苹果'),
        ('尿布', '奶粉', '洋葱'),
        ('奶粉', '洋葱')
        ]
# 挖掘频繁项集和规则
itemsets, rules = apriori(data, min_support=0.5, min_confidence=1)
print(itemsets)
print(rules)