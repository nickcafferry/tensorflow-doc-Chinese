现在我们已经介绍了TensorFlow是如何创建张量，运用变量和占位符，我们会接着介绍在计算图中如何使用这些。
然后，我们会设置一个简单的分类器再看看它的表现如何。

计算图中操作
===========

既然我们已经知道如何将对象存在计算图中，我们可以引进操作。

.. code:: python
    
    >>> import tensorflow.compat.v1 as tf
    >>> sess = tf.Session()
    >>> tf.disable_eager_execution()
 
在这个例子中，我们将我们所学的整合起来，将列表中的每个数字输入计算图中的对象，然后打印出结果。

首先，我们声明一下我们的张量和占位符。然后我们创建一个numpy的数组，然后将其输入到我们的操作中：

.. code:: python
    
    >>> import numpy as np
    # 步骤1创建数据
    >>> x_vals = np.array([1.,3.,5.,7.,9.])
    >>> x_data = tf.placeholder(tf.float32)
    >>> m_const = tf.constant(3.)
    # 步骤2创建操作
    >>> my_product = tf.multiply(x_data,m_const)
    # 步骤3输入数据并打印结果
    >>> for x_val in x_vals:
    ...    print(sess.run(my_product,feed_dict={x_data:x_val}))
    3.0
    9.0
    15.0
    21.0
    27.0
   
步骤1和步骤2在计算图中创建数据和操作，在步骤3中，我们通过计算图输入数据让后打印结果。


密集层(Dense Layer)
====================

在这节中，我们将学习如何在同样的计算谱图中放入相同的操作。

知道如何将不同的运算符链接起来很重要。在计算图中这将会设置不同的操作，这里我们展示占位符乘以两个矩阵再执行加法。我们将两个矩阵以三维的 :strong:`numpy` 阵列传入网络。

.. code:: ipython3

    import tensorflow as tf

.. code:: ipython3

    sess = tf.Session()

.. code:: ipython3

    import numpy as np
    my_array = np.array([[1., 3., 5., 7., 9.],[-2., 0., 2., 4., 6.], [-6., -3., 0., 3., 6.]])

.. code:: ipython3

    x_vals = np.array([my_array, my_array+1])
    x_data = tf.placeholder(tf.float32, shape=(3,5))

.. code:: ipython3

    x_vals, x_data




.. parsed-literal::

    (array([[[ 1.,  3.,  5.,  7.,  9.],
             [-2.,  0.,  2.,  4.,  6.],
             [-6., -3.,  0.,  3.,  6.]],
     
            [[ 2.,  4.,  6.,  8., 10.],
             [-1.,  1.,  3.,  5.,  7.],
             [-5., -2.,  1.,  4.,  7.]]]),
     <tf.Tensor 'Placeholder:0' shape=(3, 5) dtype=float32>)



.. code:: ipython3

    m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
    m2 = tf.constant([[2.]])
    a1 = tf.constant([[10.]])

.. code:: ipython3

    m1, m2, a1




.. parsed-literal::

    (<tf.Tensor 'Const_6:0' shape=(5, 1) dtype=float32>,
     <tf.Tensor 'Const_7:0' shape=(1, 1) dtype=float32>,
     <tf.Tensor 'Const_8:0' shape=(1, 1) dtype=float32>)



.. code:: ipython3

    prod1 = tf.matmul(x_data, m1)
    prod2 = tf.matmul(prod1, m2)
    add1 = tf.add(prod2,a1)

.. code:: ipython3

    prod1, prod2, add1




.. parsed-literal::

    (<tf.Tensor 'MatMul:0' shape=(3, 1) dtype=float32>,
     <tf.Tensor 'MatMul_1:0' shape=(3, 1) dtype=float32>,
     <tf.Tensor 'Add:0' shape=(3, 1) dtype=float32>)



.. code:: ipython3

    for x_val in x_vals:
        print(sess.run(add1, feed_dict= {x_data: x_val}))


.. parsed-literal::

    [[102.]
     [ 66.]
     [ 58.]]
    [[114.]
     [ 78.]
     [ 70.]]



.. raw:: html

    <video poster="../../_static/images/GCC.png" width="690" height="402" controls="controls">
        <source src="../../_static/videos/1stModel(IntroML)/IntroML4.mp4" type="video/mp4">
    </video>
