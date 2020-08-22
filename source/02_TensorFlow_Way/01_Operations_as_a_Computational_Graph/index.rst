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

.. raw:: html

    <video poster="../../_static/images/GCC.png" width="690" height="402" controls="controls">
        <source src="../../_static/videos/1stModel(IntroML)/IntroML4.mp4" type="video/mp4">
    </video>
