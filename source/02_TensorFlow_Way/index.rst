.. note::

   After we have established the basic objects and methods in TensorFlow, we now want to 
   establish the components that make up TensorFlow algorithms.  We start by introducing 
   computational graphs, and then move to loss functions and back propagation.  We end with 
   creating a simple classifier and then show an example of evaluating regression and 
   classification algorithms.

下载本章 :download:`Jupyter Notebook <https://github.com/qmlcode/qml/blob/master/qml/wrappers.py>`

.. raw:: html
   
      .wmd1{
         -webkit-transform: scale(.6);
        position:absolute;
        top:180px;
        left:200px;
        perspective: 1000px;
      }
      
      .base{ }
      
      .blades{
        width: 350px;
        height: 350px;
        left: 10%;
        top: 10%;
        z-index:2;
        border-radius: 50%;
        position: absolute;
        margin-top: -30px;
        margin-left: 50px;
      
        animation: spin 6s linear infinite;
      }
      
      .blade1 {
                background: white;
        position:absolute;
            width:41px;
            height:139px;
        top:-10px;
        left:150.5px;
        transform:rotate(0deg);
        display:inline-block;
        background:
          linear-gradient(135deg, transparent 20px, white 0),
          linear-gradient(225deg, transparent 20px, white 0),
          linear-gradient(315deg, transparent 20px, white 0),
          linear-gradient(45deg, transparent  20px,  white 0);
        background-position: top left, top right, bottom right, bottom left;
        background-size: 50% 50%;
        background-repeat: no-repeat;
      }
      
      .blade2 {
                background:white;
        position:absolute;
            width:41px;
            height:139px;
        top:105.5px;
        left:41px;
        transform:rotate(-90deg);
        display:inline-block;
        background:
          linear-gradient(135deg, transparent 20px, white 0),
          linear-gradient(225deg, transparent 20px, white 0),
          linear-gradient(315deg, transparent 20px, white 0),
          linear-gradient(45deg, transparent  20px,  white 0);
        background-position: top left, top right, bottom right, bottom left;
        background-size: 50% 50%;
        background-repeat: no-repeat;
      }
      
      .blade3 {
                background:white;
        position:absolute;
            width:41px;
            height:139px;
        top:105.5px;
        right:41px;
        transform:rotate(-270deg);
        display:inline-block;
        background:
          linear-gradient(135deg, transparent 20px, white 0),
          linear-gradient(225deg, transparent 20px, white 0),
          linear-gradient(315deg, transparent 20px, white 0),
          linear-gradient(45deg, transparent  20px,  white 0);
        background-position: top left, top right, bottom right, bottom left;
        background-size: 50% 50%;
        background-repeat: no-repeat;
      }
      
      .blade4 {
                background:white;
        position:absolute;
            width:41px;
            height:139px;
        bottom:-10px;
        left:150.5px;
        transform:rotate(180deg);
        display:inline-block;
        background:
          linear-gradient(135deg, transparent 20px, white 0),
          linear-gradient(225deg, transparent 20px, white 0),
          linear-gradient(315deg, transparent 20px, white 0),
          linear-gradient(45deg, transparent  20px,  white 0);
        background-position: top left, top right, bottom right, bottom left;
        background-size: 50% 50%;
        background-repeat: no-repeat;
      }
      
      .vane1{
        width:1px;
        height:350px;
        left:175px;
        background:white;
        position:absolute;
        transform:rotate(90deg);
      }
      
      .vane2{
        width:1px;
        height:350px;
        left:171.5px;
        background:white;
        position:absolute;
        transform:rotate(180deg);
      }
      
      .base .bottom_base{
        position:absolute;
        width:90px;
        height:100px;
        left:162px;
        border-right: 16px solid transparent;
        border-left: 16px solid transparent;
        border-bottom: 380px solid white;
        opacity:.8;
        z-index:-1;
        top:42.5px;
      }
      
      ul{
        position:absolute;
        top:180px;
        left:-30px;
      }
      li{
        width:10px;
        height:10px;
        background:white;
        padding:2px;
        display:block;
        margin: 30px;
        box-shadow: inset 0px -2px 0px lightgray; 
      }
      
      li:nth-child(2){
        position:absolute;
        top:-45px;
        left:20px;
      }
      
      li:nth-child(1){
        position:absolute;
        top:35px;
        left:50px;
      }
      
      li:nth-child(3){
        position:absolute;
        top:75px;
        left:50px;
      }
      
      @keyframes spin {
      0% {
             transform:rotate(0deg);
       }
       100% {
             transform:rotate(-360deg);
       }
      }

计算图
----------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/01_Operations_as_a_Computational_Graph/index
       
We show how to create an operation on a computational graph and how to visualize it using Tensorboard.


下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/01_Operations_as_a_Computational_Graph/01_operations_on_a_graph.ipynb>`

------------

分层嵌套操作
---------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/02_Layering_Nested_Operations/index

We show how to create multiple operations on a computational graph and how to visualize them using 
Tensorboard.

.. image:: 

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/02_Layering_Nested_Operations/02_layering_nested_operations.ipynb>`

-----

多层操作
--------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/03_Working_with_Multiple_Layers/index
       
Here we extend the usage of the computational graph to create multiple layers and show how they appear 
in Tensorboard.

.. image:: 

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/03_Working_with_Multiple_Layers/03_multiple_layers.ipynb>`

-----------

载入损失函数
----------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/04_Implementing_Loss_Functions/index

In order to train a model, we must be able to evaluate how well it is doing. This is given by loss functions.
We plot various loss functions and talk about the benefits and limitations of some.

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/04_Implementing_Loss_Functions/04_loss_functions.ipynb>`

-----------

载入反向传播
-------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/05_Implementing_Back_Propagation/index

Here we show how to use loss functions to iterate through data and back propagate errors for regression 
and classification.


 下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/05_Implementing_Back_Propagation/05_back_propagation.ipynb>`

-------------

随机和批量训练
-----------

.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/06_Working_with_Batch_and_Stochastic_Training/index
       

TensorFlow makes it easy to use both batch and stochastic training. We show how to implement both and talk 
about the benefits and limitations of each.

.. image::

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/06_Working_with_Batch_and_Stochastic_Training/06_batch_stochastic_training.ipynb>`

-----------

结合训练
-------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/07_Combining_Everything_Together/index
       
We now combine everything together that we have learned and create a simple classifier.

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/07_Combining_Everything_Together/07_combining_everything_together.ipynb>`

------------

模型评估
----------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/08_Evaluating_Models/index
 
Any model is only as good as it's evaluation.  Here we show two examples of (1) evaluating a regression 
algorithm and (2) a classification algorithm.

下载本章 :download:`Jupyter Notebook </02_TensorFlow_Way/08_Evaluating_Models/08_evaluating_models.ipynb>`


本章学习模块
-----------

.. Submodules
.. ----------

*tensorflow\.zeros* 
^^^^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.zeros
    :members:
    :undoc-members:
    :show-inheritance:

------

*tensorflow\.ones*
^^^^^^^^^^^^^^^^^^

.. automodule:: tensorflow.ones
    :members:
    :undoc-members:
    :show-inheritance:

-------------





   
