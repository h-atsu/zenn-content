---
title: "辞書式最適化について"
emoji: "📖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: []
published: false
---


# はじめに
この記事は数理最適化Advent Calendar5日目の記事です。

# 多目的最適化概要
https://x.com/MickeyKubo/status/1704273840732623046?s=20

# 辞書式最適化概要

# 問題設定
まずはじめに以下のような最適化問題を考えます

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

# 簡単な実験

https://github.com/coin-or/pulp/blob/master/pulp/pulp.py#L1936-L1938

以下ではナップサック問題を例に多目的最適化について

```python
from mip import *

p = [10, 13, 18, 31, 7, 15]
w = [11, 15, 20, 35, 10, 33]
c = 47 
I = range(len(w))

m = Model("knapsack")
x = [m.add_var(var_type=BINARY) for i in I]

m.objective = maximize(xsum(p[i] * x[i] for i in I))

m += xsum(w[i] * x[i] for i in I) <= c

m.optimize()

```

# その他
個人的に思っている雑多なことについていくつか書きます。