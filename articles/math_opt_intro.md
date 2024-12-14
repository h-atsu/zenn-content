---
title: "OR-Toolsの新しいモデラーMathOptの紹介"
emoji: "💭"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: [数理最適化, Python]
published: false
---

こんにちは，ブレインパッドでデータサイエンティストをしている[だーはー](https://twitter.com/H_A_ust)です．
この記事は[数理最適化 Advent Calendar 2024](https://qiita.com/advent-calendar/2024/mathematical-optimization)の 8 日目の記事です．

# はじめに

本記事では[Google OR-Tools](https://developers.google.com/optimization)の新しいモデラーである[math_opt](https://developers.google.com/optimization/math_opt)について紹介します．

mathopt の公式ドキュメント：https://developers.google.com/optimization/math_opt

pywraplp の公式ドキュメント：https://developers.google.com/optimization/reference/python/linear_solver/pywraplp

次の OR-Tools のメジャーアップデートとなる ver10.0 で公式ローンチ予定らしい
参照：https://github.com/google/or-tools/discussions/3992

mathopt の doxygen のドキュメント：https://or-tools.github.io/docs/python/classortools_1_1math__opt_1_1python_1_1model_1_1Model.html

CO@Work2024：https://co-at-work.zib.de/

CO@Work での OR-Tools の紹介スライド：https://co-at-work.zib.de/slides/COatWork2024_Lichocki_Google.pdf

CO@Work での OR-Tools の動画：https://youtu.be/qqQ-D63X4LM
