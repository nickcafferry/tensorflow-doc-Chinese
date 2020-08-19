现在我们已经介绍了TensorFlow是如何创建张量，运用变量和占位符，我们会接着介绍在计算图中如何使用这些。
然后，我们会设置一个简单的分类器再看看它的表现如何。

.. code:: python
    
    >>> import tensorflow.compat.v1 as tf
    >>> sess = tf.Session()
    >>> tf.disable_eager_execution()
  
  
