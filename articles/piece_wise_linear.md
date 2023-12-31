---
title: "いい感じに区分線形近似する"
emoji: "📏"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [数理最適化,機械学習]
published: true
---

こんにちは，新卒1年目でブレインパッドでデータサイエンティストをしています，[H_A_ust](https://twitter.com/H_A_ust)です．
この記事は[数理最適化アドベントカレンダー](https://qiita.com/advent-calendar/2023/mathematical-optimization)の20日目の記事です．

# はじめに
機械学習を行った後にその予測器を混合整数計画問題として定式化する際に，区分線形近似を使う状況はよくあるかと思います．
一方で非線形関数を愚直に一定間隔で離散化してしまうと間数値の変化が緩やかな区間においても過剰に点をとってしまい無駄が生じてしまうことがしばしばあります．
例えば$f(x) = \log x,\ x \in [1,10]$を等間隔に5点離散化を行うと以下のような区分線形近似となりますが，$x\in[0,1]$の領域において真の関数$\ \log(x)$と区分線形近似した関数との誤差が大きくなってしまっています．
![](/images/piece_wise_linear/log_const_disc.png)
今度は離散点を対数的に等間隔にとってみると以下のように「いい感じ」に区分線形近似できていることが分かります．
![](/images/piece_wise_linear/log_log_disc.png)
本記事ではある程度性質の良い非線形関数に対して，この様な効率的な離散近似点を選ぶ方法について説明します．
また，本記事はGurobiの[Function Constraints](https://www.gurobi.com/documentation/current/refman/general_constraints.html)の解説を参考にして裏側のアルゴリズムを想像して記事を書いています．より良い方法などをご存知でしたら教えていただけると幸いです．

# 問題設定
前述ではゆるく「いい感じ」に区分線形近似すると言いましたが，よりフォーマルにこの「いい感じ」を定義します．
まず，区分線形近似したい対象の関数を$f:\mathbb{R}\to\mathbb{R}$とします．ここで$f$は十分滑らかな関数を仮定します．
また区分線形近似する際の区間の最小値，最大値を$L,R\in\mathbb{R}$と表し，近似する際の$N$個の離散点$L=x_1<x_2<...,<x_N=R$と表すこととします．
さらに近似点$x_1,...,x_N$を与えた際の区分線形関数$\ \bar f:[L,R]\to\mathbb{R}$を

$$
\bar f(x;x_1,...,x_n) = \frac{f(x_{i+1})-f(x_{i})}{x_{i+1} - x_{i}}(x-x_i) + f(x_{i})\ \ (\text{if}\ \  x\in[x_i,x_{i+1}])
$$

で表されるものとします．
この時，真の関数$f$と区分線形関数$\bar f$との近さを

$$
d(f, \bar f) = \max_{x\in [L,R]} \left | f(x) - \bar f(x) \right|
$$
と関数の差の最大値ノルムで表すことにします．

ここで最大許容誤差$\ \text{tol}\in\mathbb{R}_{> 0}$が与えられた際に$d(f, \bar f) \le \epsilon$を満たす離散点$x_1,...,x_N$を見つける問題を考えます．
この他にも$N$を与えた元で$\varepsilon$をなるべく小さくする問題も考えられます．

# アルゴリズム
アルゴリズムを適用する上で$f$に対していくつかの仮定をおきます．
まず近似対象区間における$f$の変曲点$\{c_1,...,c_m\}\subset [L,R]$を全て知っているものとします．
例として，近似対象区間で変曲点を1個持つような非線形関数に対する初期関数$\bar f$は以下の様になります．
![](/images/piece_wise_linear/pwl_init.png)
さらに$f$に対する仮定として，各区間$[L,c_1], [c_1,c_2],...,[c_{m-1}, c_m], [c_m,R]$における微分値$\ \nabla f(x)$と$\ \nabla f^{-1}(y)$を計算できるものとします．
ここで，各区間に対して$\nabla f$が単調関数となっているので$\nabla f^{-1}$も存在が保証されます．

これらの仮定がなくても数値計算でこれらの値を求めることができる気もしますが，とりあえず簡単のために以下では仮定した上で話を進めます．

以上の設定のもとで初期の区分線形関数を$\bar f(\cdot\ ; L,c_1,...,c_m,R)$とします．
ここで各区間$A=[l,r] \in U=\{[L,c_1], [c_1,c_2],...,[c_{m-1}, c_m], [c_m,R]\}$に対して$\max_{x\in A} \left | f(x) - \bar f(x) \right| \le \varepsilon$を満たせばその区間$A$を分割することを終了して，そうでないならば$A$の中でさらに細かく離散点をとることとします．
その際の離散点の取り方として$\ p^* = \argmax_{x\in A} \left | f(x) - \bar f(x) \right|$として区間の分割を行います．

ここでどの様にして$p^*$を求めるかが問題となりますが，これは

$$
p^* = \nabla^{-1} f \left ( \frac{f(r) - f(l)}{r-l} \right )
$$

として与えられます．
以下に導出の概要を示します．
$f$は$\ x\in A$において凹関数か凸関数のいずれかになっています．
以下では凹関数を仮定します．(凸関数だとしても符号を入れ替えて同様の議論ができます)
すると考えるべき問題は

$$
\max_{x\in A}\ \varepsilon(x) = f(x) - \bar f(x)
$$

となります．
$f$を変曲点で区間分割しているので$\nabla^2 f(x) \le 0$もしくは$\nabla^2 f(x) \ge 0$であり，$\bar f$は区分線形関数であるので$\nabla^2 \bar f(x) = 0$であることに注意すると区間$A$において$\varepsilon(x)$も凸関数となっています．
従って

$$
\begin{aligned}
\nabla \varepsilon (x) &= 0 \\
\nabla f(x) - \frac{f(r) - f(l)}{r-l} &= 0
\end{aligned}
$$

を満たす$x$が求めるべき$p^*$になっていることがわかります．

以上のアルゴリズムをまとめると以下のようになります．

1. $U_0 = \{[L,c_1], [c_1,c_2],...,[c_{m-1}, c_m], [c_m,R]\}$, $i=0$とする
2. $U_i$の分割に対して$d(f, \bar f) \le \text{tol}$かどうかを確認する，満たすならば終了
3. 許容誤差を満たさない区間$A=[r,l] \in U_i$に対して$p^*$を計算して$U_{i+1}  = U_i \setminus A \cup [l,p^*] \cup [p^*,r]$
$i\leftarrow i+1$として2.に戻る


# 実装と数値実験
はじめにPythonによる実装について説明します．
まず，近似対象の関数$f$の情報を保持するクラスを定義しておきます．
```python
class AbstractFunction(ABC):

    @abstractmethod
    def __call__(self, x):
        """
        Apply the function to the input x.

        :param x: The input to the function.
        :return: The result of applying the function to x.
        """
        pass

    @abstractmethod
    def df(self, x):
        """
        Compute the derivative of the function at x.

        :param x: The point at which to compute the derivative.
        :return: The derivative of the function at x.
        """
        pass

    @abstractmethod
    def df_inv(self, g, x):
        """
        Compute the inverse derivative of the function at x.

        :param g: The gradient at which to compute the inverse derivative.
        :param x: The position at which to compute the inverse derivative.
        :return: The inverse derivative of the function at x.
        """
        pass
```

この関数と最大許容誤差$\text{tol}$を受け取って前述のアルゴリズムに基づく近似点を返す関数を以下の様に定義します．

```python
def make_pwl_pts(function: AbstractFunction, tol=1e-2):
    f = function
    x_init = list(set([f.lb, *f.infrection_pts, f.ub]))  # 最初は境界と変曲点から探索を始める

    def split_pts(lb, ub):
        xs = []
        grad = (f(ub) - f(lb)) / (ub - lb)
        def f_pwl(x): return grad*(x-lb) + f(lb)
        c = f.df_inv(grad, (lb+ub)/2)
        err = np.abs(f(c) - f_pwl(c))

        if err > tol:
            xs.append(c)
            xs.extend(split_pts(lb, c))
            xs.extend(split_pts(c, ub))
        return xs

    inner_pts = []
    for i in range(len(x_init)-1):
        inner_pts.extend(split_pts(x_init[i], x_init[i+1]))  # tolを満たすまで分割する

    ret = np.array(sorted(x_init + inner_pts))

    return ret
```

関数では区間の分割を再帰関数を用いて行っています．
もし近似する点$N$が与えられたもとで$d(f,\bar f)$をなるべく小さくしたい場合は誤差の大きい区間をヒープで管理して上から分割を行っていけば良いです．

上の関数をもとに冒頭の$f(x) = \log(x)$の分割を行います．
$\text{tol} = 0.1,0.01,0.001$として分割した結果を以下に示します．

![](/images/piece_wise_linear/log_tol0.1.png)
![](/images/piece_wise_linear/log_tol0.01.png)
![](/images/piece_wise_linear/log_tol0.001.png)

期待通り関数値の変化が激しい左端で離散点が多く取られていることが見て取れます．

# おわりに
本記事ではGurobiに実装されているFunction制約のアルゴリズムについて考察しました．
Gurobiのドキュメントを参考に考えたアルゴリズムなのでもっと効率的な方法をご存知でしたら教えていただけると幸いです．