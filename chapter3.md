当一个高尔夫球员刚开始学习打高尔夫时，他们通常会在挥杆的练习上花费大多数时间。慢慢地他们才会在基本的挥杆上通过变化发展其他的击球方式，学习低飞球、左曲球和右曲球。类似的，我们现在仍然聚焦在反向传播算法的理解上。这就是我们的“基本挥杆”——神经网络中大部分工作学习和研究的基础。本章，我会解释若干技术能够用来提升我们关于反向传播的初级的实现，最终改进网络学习的方式。

本章涉及的技术包括：更好的代价函数的选择——[交叉熵](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function) 代价函数；四中规范化方法（L1 和 L2 规范化，dropout 和训练数据的人工扩展），这会让我们的网络在训练集之外的数据上更好地泛化；更好的[权重初始化方法](http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization)；还有[帮助选择好的超参数的启发式想法](http://neuralnetworksanddeeplearning.com/chap3.html#how_to_choose_a_neural_network's_hyper-parameters)。同样我也会再给出一些简要的[其他技术介绍](http://neuralnetworksanddeeplearning.com/chap3.html#other_techniques)。这些讨论之间的独立性比较大，所有你们可以随自己的意愿挑着看。另外我还会在代码中实现这些技术，使用他们来提高在第一章中的分类问题上的性能。

当然，我们仅仅覆盖了大量已经在神经网络中研究发展出的技术的一点点内容。此处我们学习深度学习的观点是想要在一些已有的技术上入门的最佳策略其实是深入研究一小部分最重要那些的技术点。掌握了这些关键技术不仅仅对这些技术本身的理解很有用，而且会深化你对使用神经网络时会遇到哪些问题的理解。这会让你们做好在需要时快速掌握其他技术的充分准备。

# 交叉熵代价函数
---
我们大多数人觉得错了就很不爽。在开始学习弹奏钢琴不久后，我在一个听众前做了处女秀。我很紧张，开始时将八度音阶的曲段演奏得很低。我很困惑，因为不能继续演奏下去了，直到有个人指出了其中的错误。当时，我非常尴尬。不过，尽管不开心，我们却能够因为明显的犯错快速地学习到正确的东西。你应该相信下次我再演奏肯定会是正确的！相反，在我们的错误不是很好的定义的时候，学习的过程会变得更加缓慢。

理想地，我们希望和期待神经网络可以从错误中快速地学习。在实践中，这种情况经常出现么？为了回答这个问题，让我们看看一个小例子。这个例子包含一个只有一个输入的神经元：

![](http://upload-images.jianshu.io/upload_images/42741-54bcf1d440d6de82.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们会训练这个神经元来做一件非常简单的事：让输入 $$1$$ 转化为 $$0$$。当然，这很简单了，手工找到合适的权重和偏差就可以了，不需要什么学习算法。然而，看起来使用梯度下降的方式来学习权重和偏差是很有启发的。所以，我们来看看神经元如何学习。

为了让事情确定化，我会首先将权重和偏差初始化为 $$0.6$$ 和 $$0.9$$。这些就是一般的开始学习的选择，并没有任何刻意的想法。一开始的神经元的输出是 $$0.82$$，所以这离我们的目标输出 $$0.0$$ 还差得很远。点击右下角的“运行”按钮来看看神经元如何学习到让输出接近 $$0.0$$ 的。注意到，这并不是一个已经录好的动画，你的浏览器实际上是正在进行梯度的计算，然后使用梯度更新来对权重和偏差进行更新，并且展示结果。设置学习率  $$\eta=0.15$$ 进行学习一方面足够慢的让我们跟随学习的过程，另一方面也保证了学习的时间不会太久，几秒钟应该就足够了。代价函数是我们前面用到的二次函数，$$C$$。这里我也会给出准确的形式，所以不需要翻到前面查看定义了。注意，你可以通过点击 “Run” 按钮执行训练若干次。

![1](http://upload-images.jianshu.io/upload_images/42741-752fcc2176bf9b8c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![2](http://upload-images.jianshu.io/upload_images/42741-99457432d0a54542.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![3](http://upload-images.jianshu.io/upload_images/42741-1eedcdca670ee566.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![4](http://upload-images.jianshu.io/upload_images/42741-b2e1647ff9c88555.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 我们这里是静态的例子，在原书中，使用的动态示例，所以为了更好的效果，请参考[原书的此处动态示例](http://neuralnetworksanddeeplearning.com/chap3.html)。

正如你所见，神经元快速地学到了使得代价函数下降的权重和偏差，给出了最终的输出为 $$0.09$$。这虽然不是我们的目标输出 $$0.0$$，但是已经挺好了。假设我们现在将初始权重和偏差都设置为 $$2.0$$。此时初始输出为 $$0.98$$，这是和目标值的差距相当大的。现在看看神经元学习的过程。点击“Run” 按钮：

![1](http://upload-images.jianshu.io/upload_images/42741-95ceb66b3451dbb7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![2](http://upload-images.jianshu.io/upload_images/42741-68d123092f32aed6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![3](http://upload-images.jianshu.io/upload_images/42741-bff381a28137c4b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![4](http://upload-images.jianshu.io/upload_images/42741-83d1aa344e862ab9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![5](http://upload-images.jianshu.io/upload_images/42741-d16017b9a247c13f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

虽然这个例子使用的了同样的学习率（$$\eta=0.15$$），我们可以看到刚开始的学习速度是比较缓慢的。对前 $$150$$ 左右的学习次数，权重和偏差并没有发生太大的变化。随后学习速度加快，与上一个例子中类似了，神经网络的输出也迅速接近 $$0.0$$。

> 强烈建议参考[原书的此处动态示例](http://neuralnetworksanddeeplearning.com/chap3.html)感受学习过程的差异。

这种行为看起来和人类学习行为差异很大。正如我在此节开头所说，我们通常是在犯错比较明显的时候学习的速度最快。但是我们已经看到了人工神经元在其犯错较大的情况下其实学习很有难度。而且，这种现象不仅仅是在这个小例子中出现，也会再更加一般的神经网络中出现。为何学习如此缓慢？我们能够找到缓解这种情况的方法么？

为了理解这个问题的源头，想想神经元是按照偏导数（$$\partial C/\partial w$$ 和 $$\partial C/\partial b$$）和学习率（$$\eta$$）的乘积来改变权重和偏差的。所以，我们在说“学习缓慢”时，实际上就是说这些偏导数很小。理解他们为何这么小就是我们面临的挑战。为了理解这些，让我们计算偏导数看看。我们一直在用的是二次代价函数，定义如下

![](http://upload-images.jianshu.io/upload_images/42741-689311744b65e8ee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 $$a$$ 是神经元的输出，其中训练输入为 $$x=1$$，$$y=0$$ 则是目标输出。显式地使用权重和偏差来表达这个，我们有 $$a=\sigma(z)$$，其中 $$z=wx+b$$。使用链式法则来求偏导数就有：

![](http://upload-images.jianshu.io/upload_images/42741-fa97259e24bc11d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中我已经将 $$x=1$$ 和 $$y=0$$ 代入了。为了理解这些表达式的行为，让我们仔细看 $$\sigma'(z)$$ 这一项。首先回忆一下 $$\sigma$$ 函数图像：

![](http://upload-images.jianshu.io/upload_images/42741-8ff397e1f73e4090.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们可以从这幅图看出，当神经元的输出接近 $$1$$ 的时候，曲线变得相当平，所以 $$\sigma'(z)$$ 就很小了。方程 (55) 和 (56) 也告诉我们 $$\partial C/\partial w$$ 和 $$\partial C/\partial b$$ 会非常小。这其实就是学习缓慢的原因所在。而且，我们后面也会提到，这种学习速度下降的原因实际上也是更加一般的神经网络学习缓慢的原因，并不仅仅是在这个特例中特有的。

## 引入交叉熵代价函数
那么我们如何解决这个问题呢？研究表明，我们可以通过使用交叉熵代价函数来替换二次代价函数。为了理解什么是交叉熵，我们稍微改变一下之前的简单例子。假设，我们现在要训练一个包含若干输入变量的的神经元，$$x_1,x_2,...$$ 对应的权重为 $$w_1,w_2,...$$ 和偏差，$$b$$：

![](http://upload-images.jianshu.io/upload_images/42741-9f1c9f7efd4815ca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

神经元的输出就是 $$a=\sigma(z)$$，其中 $$z=\sum_jw_jx_j+b$$ 是输入的带权和。我们如下定义交叉熵代价函数：

![](http://upload-images.jianshu.io/upload_images/42741-f3363ceff9e87fb4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 $$n$$ 是训练数据的总数，对所有的训练数据 $$x$$ 和 对应的目标输出 $$y$$ 进行求和。

公式(57)是否解决学习缓慢的问题并不明显。实际上，甚至将这个定义看做是代价函数也不是显而易见的！在解决学习缓慢前，我们来看看交叉熵为何能够解释成一个代价函数。

将交叉熵看做是代价函数有两点原因。第一，它使非负的，$$C>0$$。可以看出：(a) 公式(57)的和式中所有独立的项都是非负的，因为对数函数的定义域是 $$(0,1)$$；(b) 前面有一个负号。

第二，如果神经元实际的输出接近目标值。假设在这个例子中，$$y=0$$ 而 $$a\approx 0$$。这是我们想到得到的结果。我们看到公式(57)中第一个项就消去了，因为 $$y=0$$，而第二项实际上就是 $$-ln(1-a)\approx 0$$。反之，$$y=1$$ 而 $$a\approx 1$$。所以在实际输出和目标输出之间的差距越小，最终的交叉熵的值就越低了。

综上所述，交叉熵是非负的，在神经元达到很好的正确率的时候会接近 $$0$$。这些其实就是我们想要的代价函数的特性。其实这些特性也是二次代价函数具备的。所以，交叉熵就是很好的选择了。但是交叉熵代价函数有一个比二次代价函数更好的特性就是它避免了学习速度下降的问题。为了弄清楚这个情况，我们来算算交叉熵函数关于权重的偏导数。我们将 $$a=\sigma(z)$$ 代入到公式 (57) 中应用两次链式法则，得到：

![](http://upload-images.jianshu.io/upload_images/42741-f5e7c6fea8b38d47.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

将结果合并一下，简化成：

![](http://upload-images.jianshu.io/upload_images/42741-38fcdc9659b6b608.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

根据 $$\sigma(z) = 1/(1+e^{-z})$$ 的定义，和一些运算，我们可以得到 $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$。后面在练习中会要求你计算这个，现在可以直接使用这个结果。我们看到 $$\sigma'$$ 和 $$\sigma(z)(1-\sigma(z))$$ 这两项在方程中直接约去了，所以最终形式就是：

![](http://upload-images.jianshu.io/upload_images/42741-5575d9699b92480b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这是一个优美的公式。它告诉我们权重学习的速度受到 $$\sigma(z)-y$$，也就是输出中的误差的控制。更大的误差，更快的学习速度。这是我们直觉上期待的结果。特别地，这个代价函数还避免了像在二次代价函数中类似公式中 $$\sigma'(z)$$ 导致的学习缓慢，见公式(55)。当我们使用交叉熵的时候，$$\sigma'(z)$$ 被约掉了，所以我们不再需要关心它是不是变得很小。这种约除就是交叉熵带来的特效。实际上，这也并不是非常奇迹的事情。我们在后面可以看到，交叉熵其实只是满足这种特性的一种选择罢了。

根据类似的方法，我们可以计算出关于偏差的偏导数。我这里不再给出详细的过程，你可以轻易验证得到

![](http://upload-images.jianshu.io/upload_images/42741-05567745404c5dcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 练习
* 验证 $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$。

让我们重回最原初的例子，来看看换成了交叉熵之后的学习过程。现在仍然按照前面的参数配置来初始化网络，开始权重为 $$0.6$$，而偏差为 $$0.9$$。点击“Run”按钮看看在换成交叉熵之后网络的学习情况：

![1](http://upload-images.jianshu.io/upload_images/42741-236a80d9625b7533.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![2](http://upload-images.jianshu.io/upload_images/42741-d94d7466e960b38c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![3](http://upload-images.jianshu.io/upload_images/42741-d8971200e55bc88b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![4](http://upload-images.jianshu.io/upload_images/42741-cfa4b9cf7de3b2c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![5](http://upload-images.jianshu.io/upload_images/42741-9aadf7001094c81d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

毫不奇怪，在这个例子中，神经元学习得相当出色，跟之前差不多。现在我们再看看之前出问题的那个例子，权重和偏差都初始化为 $$2.0$$：

![1](http://upload-images.jianshu.io/upload_images/42741-461c7ebcebdc9551.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![2](http://upload-images.jianshu.io/upload_images/42741-bd5ac4ae285ff597.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![3](http://upload-images.jianshu.io/upload_images/42741-24b6e22490e6469e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![4](http://upload-images.jianshu.io/upload_images/42741-4fedc8ddbdb3a5cb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![5](http://upload-images.jianshu.io/upload_images/42741-0860b6265dfa1305.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

成功了！这次神经元的学习速度相当快，跟我们预期的那样。如果你观测的足够仔细，你可以发现代价函数曲线要比二次代价函数训练前面部分要陡很多。正是交叉熵带来的快速下降的坡度让神经元在处于误差很大的情况下能够逃脱出学习缓慢的困境，这才是我们直觉上所期待的效果。

我们还没有提及关于学习率的讨论。刚开始使用二次代价函数的时候，我们使用了 $$\eta=0.15$$。在新例子中，我们还应该使用同样的学习率么？实际上，根据不同的代价函数，我们不能够直接去套用同样的学习率。这好比苹果和橙子的比较。对于这两种代价函数，我只是通过简单的实验来找到一个能够让我们看清楚变化过程的学习率的值。尽管我不愿意提及，但如果你仍然好奇，这个例子中我使用了 $$\eta=0.005$$。

你可能会反对说，上面学习率的改变使得上面的图失去了意义。谁会在意当学习率的选择是任意挑选的时候神经元学习的速度？！这样的反对其实没有抓住重点。上面的图例不是想讨论学习的绝对速度。而是想研究学习速度的变化情况。特别地，当我们使用二次代价函数时，学习在神经元犯了明显的错误的时候却比学习快接近真实值的时候缓慢；而使用交叉熵学习正是在神经元犯了明显错误的时候更快。这些现象并不是因为学习率的改变造成的。

我们已经研究了一个神经元的交叉熵。不过，将其推广到多个神经元的多层神经网络上也是很简单的。特别地，假设 $$y=y_1,y_2,...$$ 是输出神经元上的目标值，而 $$a_1^L,a_2^L,...$$是实际输出值。那么我们定义交叉熵如下

![](http://upload-images.jianshu.io/upload_images/42741-80db42bf50ee6cfa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

除了这里需要对所有输出神经元进行求和外，这个其实和我们早前的公式(57)一样的。这里不会给出一个推算的过程，但需要说明的时使用公式(63)确实会在很多的神经网络中避免学习的缓慢。如果你感兴趣，你可以尝试一下下面问题的推导。

那么我们应该在什么时候用交叉熵来替换二次代价函数？实际上，如果在输出神经元使用 sigmoid 激活函数时，交叉熵一般都是更好的选择。为什么？考虑一下我们初始化网络的时候通常使用某种随机方法。可能会发生这样的情况，这些初始选择会对某些训练输入误差相当明显——比如说，目标输出是 $$1$$，而实际值是 $$0$$，或者完全反过来。如果我们使用二次代价函数，那么这就会导致学习速度的下降。它并不会完全终止学习的过程，因为这些权重会持续从其他的样本中进行学习，但是显然这不是我们想要的效果。

## 练习
* 一个小问题就是刚接触交叉熵时，很难一下子记住那些表达式对应的角色。又比如说，表达式的正确形式是 $$−[y\ln a+(1−y)\ln(1−a)]$$
 或者 $$−[a\ln y+(1−a)\ln(1−y)]$$。在 $$y=0$$ 或者 $$1$$ 的时候这些公式的结果怎样？