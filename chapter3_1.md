## Softmax
本章，我们大多数情况会使用交叉熵来解决学习缓慢的问题。但是，我希望简要介绍一下基于 softmax 神经元层的解决这个问题的另一种观点。我们不会实际在剩下的章节中使用 softmax 层，所以你如果赶时间，就可以跳到下一个小节了。不过，softmax 仍然有其重要价值，一方面它本身很有趣，另一方面，因为我们会在第六章在对深度神经网络的讨论中使用 softmax 层。

softmax 的想法其实就是为神经网络定义一种新式的输出层。开始时和 sigmoid 层一样的，首先计算带权输入 $$z_j^L = \sum_k w_{jk}^La_k^{L-1}+b_j^L$$。不过，这里我们不会使用 sigmoid 函数来获得输出。而是，会应用一种叫做 softmax 函数在 $$z_j^L$$ 上。根据这个函数，激活值 $$a_j^L$$ 就是

![](http://upload-images.jianshu.io/upload_images/42741-b84180005a015ffc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中，分母是对所有的输出神经元进行求和。

如果你不习惯这个函数，方程(78)可能看起来会比较难理解。因为对于使用这个函数的原因你不清楚。为了更好地理解这个公式，假设我们有一个包含四个输出神经元的神经网络，对应四个带权输入，表示为 $$z_1^L, z_2^L, z_3^L, z_4^L$$。下面的例子可以进行对应权值的调整，并给出对应的激活值的图像。可以通过固定其他值，来看看改变 $$z_4^L$$ 的值会产生什么样的影响：

![1](http://upload-images.jianshu.io/upload_images/42741-d21cb21704e16879.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![2](http://upload-images.jianshu.io/upload_images/42741-b2b1987aa0501ef6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![3](http://upload-images.jianshu.io/upload_images/42741-9ba8de22bc879ab7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 这里给出了三个选择的图像，建议去原网站体验

在我们增加 $$z_4^L$$ 的时候，你可以看到对应激活值 $$a_4^L$$ 的增加，而其他的激活值就在下降。类似地，如果你降低 $$z_4^L$$ 那么 $$a_4^L$$ 就随之下降。实际上，如果你仔细看，你会发现在两种情形下，其他激活值的整个改变恰好填补了 $$a_4^L$$ 的变化的空白。原因很简单，根据定义，输出的激活值加起来正好为 $$1$$，使用公式(78)我们可以证明：

![](http://upload-images.jianshu.io/upload_images/42741-56656f179bd9308a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

所以，如果 $$a_4^L$$ 增加，那么其他输出激活值肯定会总共下降相同的量，来保证和式为 $$1$$。当然，类似的结论对其他的激活函数也需要满足。

方程(78)同样保证输出激活值都是正数，因为指数函数是正的。将这两点结合起来，我们看到 softmax 层的输出是一些相加为 $$1$$ 正数的集合。换言之，softmax 层的输出可以被看做是一个概率分布。

这样的效果很令人满意。在很多问题中，将这些激活值作为网络对于某个输出正确的概率的估计非常方便。所以，比如在 MNIST 分类问题中，我们可以将 $$a_j^L$$ 解释成网络估计正确数字分类为 $$j$$ 的概率。

对比一下，如果输出层是 sigmoid 层，那么我们肯定不能假设激活值形成了一个概率分布。我不会证明这一点，但是源自 sigmoid 层的激活值是不能够形成一种概率分布的一般形式的。所以使用 sigmoid 输出层，我们没有关于输出的激活值的简单的解释。

## 练习
* 构造例子表明在使用 sigmoid 输出层的网络中输出激活值 $$a_j^L$$ 的和并不会确保为 $$1$$。

我们现在开始体会到 softmax 函数的形式和行为特征了。来回顾一下：在公式(78)中的指数函数确保了所有的输出激活值是正数。然后分母的求和又保证了 softmax 的输出和为 $$1$$。所以这个特定的形式不再像之前那样难以理解了：反而是一种确保输出激活值形成一个概率分布的自然的方式。你可以将其想象成一种重新调节 $$z_j^L$$   的方法，然后将这个结果整合起来构成一个概率分布。

## 练习
* **softmax 的单调性** 证明如果 $$j=k$$ 则 $$\partial a_j^L/\partial z_k^L$$ 为正，否则为负。结果是，增加 $$z_^L$$ 会提高对应的输出激活值 $$a_k^L$$ 并降低其他所有输出激活值。我们已经在滑条示例中实验性地看到了这一点，这里需要你给出一个严格证明。
* **softmax 的非局部性** sigmoid 层的一个好处是输出 $$a_j^L$$ 是对应带权输入 $$a_j^L = \sigma(z_j^L)$$ 的函数。解释为何对于 softmax 来说，并不是这样的情况：仍和特定的输出激活值 $$a_j^L$$ 依赖所有的带权输入。

## 问题
* **逆转 softmax 层** 假设我们有一个使用 softmax 输出层的神经网络，然后激活值 $$a_j^L$$ 已知。证明对应带权输入的形式为 $$z_j^L = \ln a_j^L + C$$，其中 $$C$$ 是独立于 $$j$$ 的。

**学习缓慢问题**：我们现在已经对 softmax 神经元有了一定的认识。但是我们还没有看到 softmax 会怎么样解决学习缓慢问题。为了理解这点，先定义一个 log-likelihood 代价函数。我们使用 $$x$$ 表示训练输入，$$y$$ 表示对应的目标输出。然后关联这个训练输入样本的 log-likelihood 代价函数就是

![](http://upload-images.jianshu.io/upload_images/42741-c6a82eff60626d3c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

所以，如果我们训练的是 MNIST 图像，输入为 $$7$$ 的图像，那么对应的 log-likelihood 代价就是 $$-\ln a_7^L$$。看看这个直觉上的含义，想想当网络表现很好的时候，也就是确认输入为 $$7$$ 的时候。这时，他会估计一个对应的概率 $$a_7^L$$ 跟 $$1$$ 非常接近，所以代价 $$-\ln a_7^L$$ 就会很小。反之，如果网络的表现糟糕时，概率 $$a_7^L$$ 就变得很小，代价 $$-\ln a_7^L$$ 随之增大。所以 log-likelihood 代价函数也是满足我们期待的代价函数的条件的。

那关于学习缓慢问题呢？为了分析它，回想一下学习缓慢的关键就是量 $$\partial C/\partial w_{jk}^L$$ 和 $$\partial C/\partial b_j^L$$ 的变化情况。我不会显式地给出详细的推导——请你们自己去完成这个过程——但是会给出一些关键的步骤：


![](http://upload-images.jianshu.io/upload_images/42741-2ca3c9d2f0c4d8a6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 请注意这里的表示上的差异，这里的 $$y$$ 和之前的目标输出值不同，是离散的向量表示，对应位的值为 $$1$$，而其他为 $$0$$。
这些方程其实和我们前面对交叉熵得到的类似。就拿方程(82) 和 (67) 比较。尽管后者我对整个训练样本进行了平均，不过形式还是一致的。而且，正如前面的分析，这些表达式确保我们不会遇到学习缓慢的问题。实际上，将 softmax 输出层和 log-likelihood 组合对照 sigmoid 输出层和交叉熵的组合类比着看是非常有用的。

有了这样的相似性，你会使用哪一种呢？实际上，在很多应用场景中，这两种方式的效果都不错。本章剩下的内容，我们会使用 sigmoid 输出层和交叉熵的组合。后面，在第六章中，我们有时候会使用 softmax 输出层和 log-likelihood 的则和。切换的原因就是为了让我们的网络和某些在具有影响力的学术论文中的形式更为相似。作为一种更加通用的视角，softmax 加上 log-likelihood 的组合更加适用于那些需要将输出激活值解释为概率的场景。当然这不总是合理的，但是在诸如 MNIST 这种有着不重叠的分类问题上确实很有用。

## 问题
* 推导方程(81) 和 (82)
* **softmax 这个名称从何处来？** 假设我们改变一下 softmax 函数，使得输出激活值定义如下

![](http://upload-images.jianshu.io/upload_images/42741-5b3e8cf9fceece21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 $$c$$ 是正的常量。注意 $$c=1$$ 对应标准的 softmax 函数。但是如果我们使用不同的 $$c$$ 得到不同的函数，其实最终的量的结果却和原来的 softmax 差不多。特别地，证明输出激活值也会形成一个概率分布。假设我们允许 $$c$$ 足够大，比如说 $$c\rightarrow \infty$$。那么输出激活值 $$a_j^L$$ 的极限值是什么？在解决了这个问题后，你应该能够理解 $$c=1$$ 对应的函数是一个最大化函数的 softened 版本。这就是 softmax 的来源。
> 这让我联想到 EM 算法，对 k-Means 算法的一种推广。

* **softmax 和 log-likelihood 的反向传播** 上一章，我们推到了使用 sigmoid 层的反向传播算法。为了应用在 softmax 层的网络上，我们需要搞清楚最后一层上误差的表示 $$\delta_j^L \equiv \partial C/\partial z_j^L$$。证明形式如下：

![](http://upload-images.jianshu.io/upload_images/42741-73905c0ac973a00d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

使用这个表达式，我们可以应用反向传播在采用了 softmax 输出层和 log-likelihood 的网络上。