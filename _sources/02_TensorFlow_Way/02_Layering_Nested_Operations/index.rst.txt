既然我们说到了多层操作，我们会介绍如何将不同的可以传播数据的层连接起来。

准备
============

这里，我们会介绍如何以最佳的方式连接不同的层，包括用户自定义的层。我们所采用的数据是小型随机图像的一些
代表。因为最好理解的方式是将这些放到简单的例子中，以及我们如何运用内置的层去完成计算。我们会用一个小型的
移动窗口来捕捉2维图片，然后通过一个用户自定义的操作层来输出数据。

在这一节中，我们会看到计算图可能会变得很大，然后很难观察。为了解决这一难题，我们引入命名操作和创建作用域的方法。

.. digraph:: foo

   "bar" -> "baz" -> "quux";

.. py:function:: enumerate(sequence[, start=0])

   返回一个迭代对象,递归式处理字典结构的索引或是其它类似序列内容
   
.. raw:: html

    <video poster="../../_static/images/GCC.png" width="690" height="402" controls="controls">
        <source src="../../_static/videos/Intro2ML/TFIntro1.mp4" type="video/mp4">
    </video>
