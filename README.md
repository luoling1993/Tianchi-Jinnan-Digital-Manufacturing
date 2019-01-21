# **津南数字制造算法挑战赛**

- [赛题介绍](https://tianchi.aliyun.com/competition/entrance/231695/introduction)
- 线上成绩
  - 由于数据量太少，异常值的波动影响较大，基于该框架下分数在0.00009-0.0001之间波动
  - automl，tree_stacking，selector和blending之间的各种组合没有细测(因为评分次数过少)
  - 各种参数没有调优

- 整体项目框架
  - processing: 特征预处理，主要参考[鱼佬的baseline](https://tianchi.aliyun.com/notebook-ai/detail?postId=41822)，在那基础上加了自己的理解，并做了代码的整理封装
  - tree_models：通过stacking方式进行模型融合，主要参考[鱼佬的baseline](https://tianchi.aliyun.com/notebook-ai/detail?postId=41822)，同样做了代码的整理封装
  - selector：通过[蛇佬的思路](https://github.com/luoda888/tianchi-diabetes-top12)，用贪婪算法的方式求得局部最优的特征组合，做了代码的封装整理，并修改了一些部分
  - automl：h20的automl，主要参考[蛇佬的baseline](https://tianchi.aliyun.com/notebook-ai/detail?postId=43185)，发现模型数量越多并不一定效果更好(反正我测了320个模型分数低于20个模型的分数)
  - blending：tree_models+automl的简单加权融合，其实blending是用boostrap方法的，此处没用boostrap。ps. tree_models之前测试过boostrap，线下效果很好，线上就是渣渣，所以没用
  - main：整个项目的整合
- 题外话
  - 题目真的不太好做，数据量少特征逻辑性差
  - 很多神仙特征搞不明白，但是就是能上分，特征解释性太差了
  - 真的佩服前排的那些大佬的
- 扩展
  - 之前打算用构造好的特征做cnn或者用AutoEncoder进行特征的抽取的，但是后面实现了selector和automl，感觉应该是差不多的作用
  - 本来想打算将stacking封装一下的，写到一半没有继续写下去了，还有带配置的fit、predict和非sklearn的模型怎么集成进去还需要扩展，现在只实现了sklearn的api的集成
  - 因为有解释性的特征比较难找，所以可以尝试暴力生成特征，然后通过大量模型自己实现模型的赛选学习，从而达到上分