---
title: "機械学習モデルを混合整数計画に埋め込む"
emoji: "🌮"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [数理最適化,機械学習]
published: false
---

こんにちは，新卒1年目でブレインパッドでデータサイエンティストをしています，[H_A_ust](https://twitter.com/H_A_ust)です．
この記事は[数理最適化アドベントカレンダー](https://qiita.com/advent-calendar/2023/mathematical-optimization)の20日目の記事です．

# はじめに
実務において機械学習での予測をもとに意思決定を行うために数理最適化を用いるという場面は多いかと思います．
広告予算配分，価格最適化や適応的実験計画などは機械学習による予測に基づく数理最適化の典型例かと思います．
例えば広告予算配分であればモデル(予算や広告に関する特徴量から利益やCVなどの関数)の学習を行った後に，与えられた予算のもとで利益などの最大化問題を考えるような状況を想定しています．
本記事ではそのような機械学習による予測器の学習を行なった後に，その予測モデルをどう最適化問題(今回は特に混合整数線形計画問題)の中に組み込むかという話をしようと思います[^1]．

非線形な予測器を混合整数線形計画問題の中に組み込む方法として以前から[Gurobi Machine Learning](https://gurobi-machinelearning.readthedocs.io/en/stable/userguide.html)と呼ばれる有償ソルバーの[Gurobi](https://www.gurobi.com/)を用いたライブラリが存在していましたが[^3]，今月に新しく無償ソルバーの[SCIP](https://www.scipopt.org/)ベースで同じような機能を提供する[PySCIOPOpt-ML](https://pyscipopt-ml.readthedocs.io/en/latest/)と呼ばれるライブラリが公開されました．
本記事では以下のPySCIPOpt-MLの元論文を参考にしながらライブラリの紹介と中身の数理モデルについての解説を行いたいと思います．


https://arxiv.org/abs/2312.08074


# 機械学習に基づく混合整数線形計画問題
ひとくちに機械学習に基づく混合整数線形計画問題と言っても定式化を考える上で，最適化問題の決定変数が予測器の入力変数になる場合とならない場合で大きく状況が異なる様に思います．

決定変数が予測器の入力変数とならない例としては，各ユーザーに対するクリック率の予測結果をもとに予算制約を満たすようにクーポン配布の計画を立てる問題や，需要予測を行った後に予測需要量をもとにして在庫発注の計画を立てる問題などが挙げられます．これらの問題は機械学習モデルの入力変数に決定変数が現れず，最適化の入力データを作成する際に前処理として予測器による推論を行うことができるので，一応は予測部分と最適化部分を切り離して考えることができます[^4]．

一方で前述の広告予算配分や価格最適化問題などでは，モデルの入力変数として決定変数(予算配分，価格)が含まれる形となっているので最適化問題の中で予測部分を考慮してあげる必要があります[^5][^6]．
ここでの難しさは入力に対して非線形な機械学習モデルをどうやって混合整数線形計画問題の中に入れ込むかという点になっています．
仮に考えている機械学習モデルが線形回帰のように入力変数に対して線形なモデル，そうでなくても決定変数に対してだけでも線形なモデルであれば自然に混合整数線形計画問題に組み込むことができますが，そうでない場合は工夫が必要となります[^7]．

Gurobi-MLやPySCIPOpt-MLではこれらの問題点に対して非線形の予測器の推論を混合整数線形計画で表現することで全体としての最適化問題も混合整数線形計画問題で定式化することを可能にしています．
ただSCIPは混合整数非線形計画問題も扱えるので必ずしも混合整数線形計画に定式化しているとは限らないので注意が必要です．


[久保先生解説動画](https://youtu.be/RoRwF6Gudzk?si=-urrR961dnyKfDQL)

# PySCIPOpt
PySCIPOpt-MLの紹介をする前に，準備としてSCIPのモデラーであるPySCIPOptについて説明を行いたいと思います．

```python
x = model.addVar("x")
y = model.addVar("y", vtype="INTEGER")
model.setObjective(x + y)
model.addCons(2*x - y*y >= 0)
model.optimize()
sol = model.getBestSol()
print("x: {}".format(sol[x]))
print("y: {}".format(sol[y]))
```

# PySCIPOpt-ML


# 数値実験


# おわわりに



[^1]:[久保先生の動画](https://www.youtube.com/watch?v=BAqRNNkyor0&list=PLz8sHu_CzBwP6JgUUOM265kMCNQU2u-r2)で言うところのMachineLearning→MathematicalOptimizationの内容になるかと思います．
[^2]: 他にも最適決定木を使って学習も推論を含んだ最適化問題も混合整数線形計画問題で行うといったアプローチも存在します
[^3]: 久保先生の動画で詳しく解説されているのでぜひご覧ください
[^4]: この様な問題に対しても[Smart "Predict then Optimize"](https://arxiv.org/abs/1710.08005)といったような考えるべきことがあります
[^5]: 典型的な広告予算配分に関しては連続最適化で解くこともあるかと思いますが，ここでは複雑な論理制約が入る場合を考えて混合整数線形計画で定式化する状況を考えます，この話題についてはOptimization Nightで私の上司の魚井さんが話してくれた[資料](https://speakerdeck.com/hidenari/optimizationnight-ji-jie-xue-xi-toshu-li-zui-shi-hua-norong-he)が参考になります．
[^6]: ここでの話は魚井さんの資料で言うところの「疎な融合問題」と「密な融合問題」の話です．
[^7]: 機械学習モデルの入力変数として現れる決定変数の次元が1次元であれば区分線形近似を行って混合整数線形計画に定式化することができますが，多次元になると辛くなります，多次元の非線形関数に対しても区分線形近似する手法は[Model Building in Mathematical Programing](https://www.wiley.com/en-us/Model+Building+in+Mathematical+Programming%2C+5th+Edition-p-9781118443330)の教科書に載っています．こちらの本は[Research Gate](https://www.researchgate.net/file.PostFileLoader.html?id=546b5d0bd685cc9e2b8b45d4&assetKey=AS%3A273636437495809%401442251418466)にてpdfをダウンロードすることができます．