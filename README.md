# MNIST_numberRecognition_neuralNetworks
基于一个 2-layer 「神经网络」对 MNIST 数据集的一个数字识别，IDE 为 Octave<br/>
MINST 数据集地址：http://yann.lecun.com/exdb/mnist/<br/>
项目采用了「矢量化编程」，循环相对较少<br/>
<br/>
　　ex_nn.m 是项目的执行文件<br/>
　　loadData.m 是加载数据的函数，参数为 0 时加载 test 数据，否则加载 training 数据；只要修改此函数里的文件名变量至 MINST 数据集，即可运行<br/>
　　lrCostFunction_nn.m 是代价函数，返回 J 和 J 对 θ 的导数<br/>
　　fmin_nn.m 是求取 θ 的函数，通过 Mini-batch GD 和 BGD 迭代，得到使 J 最小的 θ<br/>
　　predict_nn.m 是预测函数，传入 test 样本集和 θ，得到 m*1 预测结果<br/>
　　sigmoid.m 和 sigmoidGradient.m 是激活函数和激活函数的导数<br/>
　　transformDataToImage.m 把 data 转化为 image，方便查看，单独使用<br/>
<br/>
最后打印出程序用时，程序用时与 n_2 （隐藏层神经元数目）变量大小成正相关<br/>
n_2 等于 30 时，Macbook Air (1.6GHz 双核 Intel Core i5)耗时约 80 s， test 准确率为 91.23%<br/>
