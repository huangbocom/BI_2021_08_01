## Thinking1	Wide & Deep的模型结构是怎样的，为什么能通过具备记忆和泛化能力（memorization and generalization）
** Wide & Deep算法由wide推荐和deep推荐两部分组成，模型结合了LR和DNN。wide推荐采用LR，输入特征是原始的离散特征（实际中往往还有叉乘特征）；Deep推荐部分是前馈神经网络，网络会对一些离散特征学习一个低维的dense embeddings，然后和一些原始的连续特征一起作为网络的输入。这两个部分通过ensemble或joint training的方式融合。
** wide推荐使用的特征是Cross Product Transformation生成的组合特征，但无法学习到训练集中没有出现的组合特征，捕捉sparse特征之间的高阶相关性，即“记忆” 历史数据中曾共同出现过的特征对，因此具备记忆能力；deep推荐通过深度学习为sparse特征学习低维的dense embeddings来捕获特征相关性，可以学习到训练集中没有出现的组合特征，因此具备泛化能力。
## Thinking2	在CTR预估中，使用FM与DNN结合的方式，有哪些结合的方式，代表模型有哪些？
** 并行结构：deepfm
** 串行结构：NFM
## Thinking3	为什么YouTube采用期望观看时间作为评估指标
** 	与YouTube商业收益有关，观看时长越长收益越多，而CTR指标对于视频搜索具有一定的欺骗性，所以采用期望观看时间作为评估指标。
