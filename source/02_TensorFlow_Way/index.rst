.. note::

   在我们在TensorFlow中建立基本的对象和方法之后，我们想要建立可以组成TensorFlow算法的成分。我们可以从引入计算图开始，然后我们在转向损失函数和反向传播。在本章的末尾，我们会创造一个简单的分类器然后展示一个回归和分类的算法。
   

下载本章 :download:`Jupyter Notebook <https://github.com/qmlcode/qml/blob/master/qml/wrappers.py>`

.. raw:: html
   
   <head>
   <meta charset="utf-8">
   <title>CSS</title>
   
   <style>
   body {
     background: #333;  
   }
   
   #container {
     position: absolute;
     width: 20%;
   	height: 20%;
     top: 1px;
     right: 70px;
   }
   
   #owl {  
     position: absolute;
   	right:80px;
      top:1px	 
     -webkit-transform: translate(-50%, 0%);	 
             transform: translate(-50%, 0%);
   }
   
   #ear1_1 {
     position: absolute;
     width: 190px;
   	height: 68px;
   	background: #779943;
   	border-radius: 130px / 50px;
     top: 20px;
     left: 210px;
     -webkit-transform: scale(0.5) rotate(-80deg);
             transform: scale(0.5) rotate(-80deg);
   }
   
   #ear1_2 {
     position: absolute;
     width: 190px;
   	height: 68px;
   	background: #779943;
   	border-radius: 130px / 50px;
     top: 8px;
     left: 245px;
     -webkit-transform: scale(0.5) rotate(45deg);
             transform: scale(0.5) rotate(45deg);
   }
   
   #ear1_3 {
     position: absolute;
     width: 60px;
   	height: 60px;
   	background: #779943;
   	border-radius: 150px;
     top: 37px;
     left: 295px;
     -webkit-transform: scale(1) rotate(45deg);
             transform: scale(1) rotate(45deg);
   }
   
   #ear2_1 {
     position: absolute;
     width: 190px;
   	height: 68px;
   	background: #779943;
   	border-radius: 130px / 50px;
     top: 20px;
     left: 464px;
     -webkit-transform: scale(0.5) rotate(80deg);
             transform: scale(0.5) rotate(80deg);
   }
   
   #ear2_2 {
     position: absolute;
     width: 190px;
   	height: 68px;
   	background: #779943;
   	border-radius: 130px / 50px;
     top: 8px;
     left: 429px;
     -webkit-transform: scale(0.5) rotate(-45deg);
             transform: scale(0.5) rotate(-45deg);
   }
   
   #ear2_3 {
     position: absolute;
     width: 60px;
   	height: 60px;
   	background: #779943;
   	border-radius: 150px;
     top: 37px;
     left: 499px;
     -webkit-transform: scale(1) rotate(-45deg);
             transform: scale(1) rotate(-45deg);
   }
   
   #owl_body {
     position: absolute;
     width: 220px;
   	height: 250px;
   	background: #779943;
   	border-radius: 180px / 200px;
     top: 157px;
     left: 320px;
     -webkit-transform: scale(2);
             transform: scale(2);
   }
   
   #eye1_1 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 80px;
     left: 265px;
     -webkit-transform: scale(1);
             transform: scale(1);
   }
   
   #eye2_1 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 80px;
     left: 416px;
     -webkit-transform: scale(1);
             transform: scale(1);
   }
   
   #eye1_2 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #779943;
   	border-radius: 150px;
     top: 70px;
     left: 265px;
     -webkit-transform: scale(0.8);
             transform: scale(0.8);
   }
   
   #eye2_2 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #779943;
   	border-radius: 150px;
     top: 70px;
     left: 416px;
     -webkit-transform: scale(0.8);
             transform: scale(0.8);
   }
   
   #eye1_3 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #ffffff;
   	border-radius: 150px;
     top: 70px;
     left: 265px;
     -webkit-transform: scale(0.67);
             transform: scale(0.67);
   }
   
   #eye2_3 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #ffffff;
   	border-radius: 150px;
     top: 70px;
     left: 416px;
     -webkit-transform: scale(0.67);
             transform: scale(0.67);
   }
   
   #eye1_4 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #1c260d;
   	border-radius: 150px;
     top: 70px;
     left: 265px;
     -webkit-transform: scale(0.2);
             transform: scale(0.2);
     -webkit-transition: all 1s;
     transition: all 1s;
   }
   
   #eye2_4 {
     position: absolute;
     width: 180px;
   	height: 180px;
   	background: #1c260d;
   	border-radius: 150px;
     top: 70px;
     left: 416px;
     -webkit-transform: scale(0.2);
             transform: scale(0.2);
     -webkit-transition: all 1s;
     transition: all 1s;
   }
   
   #beak {
     position: absolute;
     top: 200px;
     left: 400px; 
     border-top: 55px solid #fc9627;
     border-left: 30px solid transparent; 
   }
   
   #beak1 {
     position: absolute;
     top: 200px;
     left: 428px;  
     border-top: 55px solid #fc9627;
     border-right: 30px solid transparent;
   }
   
   #bellyContainer {
     position: absolute;
     width: 221px;
   	height: 160px;
   	background: #e7e99e;
   	border-radius: 221px / 160px;
     top: 310px;
     left: 320px;
     -webkit-transform: scale(1.5);
             transform: scale(1.5);
   }
   
   #b11 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #597332;
   	border-radius: 150px;
     top: 400px;
     left: 315px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b11_1 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 385px;
     left: 315px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b12 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #597332;
   	border-radius: 150px;
     top: 400px;
     left: 386px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b12_1 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 385px;
     left: 386px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b13 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #597332;
   	border-radius: 150px;
     top: 400px;
     left: 456px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b13_1 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 385px;
     left: 456px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b21 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #597332;
   	border-radius: 150px;
     top: 340px;
     left: 266px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b21_1 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 325px;
     left: 266px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b22 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #597332;
   	border-radius: 150px;
     top: 340px;
     left: 343px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b22_1 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 325px;
     left: 343px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b23 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #597332;
   	border-radius: 150px;
     top: 340px;
     left: 421px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b23_1 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 325px;
     left: 421px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b24 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #597332;
   	border-radius: 150px;
     top: 340px;
     left: 499px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b24_1 {
     position: absolute;
     width: 100px;
   	height: 100px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 325px;
     left: 499px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b31 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #597332;
   	border-radius: 150px;
     top: 290px;
     left: 315px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b31_1 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 275px;
     left: 315px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b32 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #597332;
   	border-radius: 150px;
     top: 290px;
     left: 386px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b32_1 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 275px;
     left: 386px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b33 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #597332;
   	border-radius: 150px;
     top: 290px;
     left: 456px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #b33_1 {
     position: absolute;
     width: 90px;
   	height: 90px;
   	background: #e7e99e;
   	border-radius: 150px;
     top: 275px;
     left: 456px;
     -webkit-transform: scale(0.8) rotate(45deg);
             transform: scale(0.8) rotate(45deg);
   }
   
   #wing1_1 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #779943;
   	border-radius: 200px / 68px;
     top: 280px;
     left: 590px;
     -webkit-transform: scale(0.7) rotate(42deg);
             transform: scale(0.7) rotate(42deg);
   }
   
   #wing1_2 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #779943;
   	border-radius: 200px / 68px;
     top: 305px;
     left: 574px;
     -webkit-transform: scale(0.7) rotate(65deg);
             transform: scale(0.7) rotate(65deg);
   }
   
   #wing1_3 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #779943;
   	border-radius: 200px / 68px;
     top: 300px;
     left: 545px;
     -webkit-transform: scale(0.7) rotate(90deg);
             transform: scale(0.7) rotate(90deg);
   }
   
   #wing2_1 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #779943;
   	border-radius: 200px / 68px;
     top: 280px;
     left: 68px;
     -webkit-transform: scale(0.7) rotate(-42deg);
             transform: scale(0.7) rotate(-42deg);
   }
   
   #wing2_2 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #779943;
   	border-radius: 200px / 68px;
     top: 305px;
     left: 89px;
     -webkit-transform: scale(0.7) rotate(-65deg);
             transform: scale(0.7) rotate(-65deg);
   }
   
   #wing2_3 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #779943;
   	border-radius: 200px / 68px;
     top: 300px;
     left: 113px;
     -webkit-transform: scale(0.7) rotate(-90deg);
             transform: scale(0.7) rotate(-90deg);
   }
   
   
   #leg1_1 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #fc9627;
   	border-radius: 200px / 68px;
     top: 487px;
     left: 240px;
     -webkit-transform: scale(0.4) rotate(-47deg);
             transform: scale(0.4) rotate(-47deg);
   }
   
   #leg1_2 {
     position: absolute;
     width: 200px;
   	height: 58px;
   	background: #fc9627;
   	border-radius: 200px / 58px;
     top: 495px;
     left: 253px;
     -webkit-transform: scale(0.4) rotate(-85deg);
             transform: scale(0.4) rotate(-85deg);
   }
   
   #leg1_3 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #fc9627;
   	border-radius: 200px / 68px;
     top: 491px;
     left: 266px;
     -webkit-transform: scale(0.4) rotate(-115deg);
             transform: scale(0.4) rotate(-115deg);
   }
   
   #leg2_1 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #fc9627;
   	border-radius: 200px / 68px;
     top: 487px;
     left: 426px;
     -webkit-transform: scale(0.4) rotate(47deg);
             transform: scale(0.4) rotate(47deg);
   }
   
   #leg2_2 {
     position: absolute;
     width: 200px;
   	height: 58px;
   	background: #fc9627;
   	border-radius: 200px / 58px;
     top: 495px;
     left: 413px;
     -webkit-transform: scale(0.4) rotate(85deg);
             transform: scale(0.4) rotate(85deg);
   }
   
   #leg2_3 {
     position: absolute;
     width: 200px;
   	height: 68px;
   	background: #fc9627;
   	border-radius: 200px / 68px;
     top: 491px;
     left: 400px;
     -webkit-transform: scale(0.4) rotate(115deg);
             transform: scale(0.4) rotate(115deg);
   }
   
   #owl:hover #wing1{
     -webkit-transition: transform-rotate 1s;
     transition: transform-rotate 1s;
     -webkit-transform-origin: 0px 0px;
             transform-origin: 0px 0px;
     -webkit-transform: translate(-70px, 360px) rotate(-30deg);
             transform: translate(-70px, 360px) rotate(-30deg);
   }
   
   #owl:hover #wing2{
     -webkit-transform-origin: 0px 0px;
             transform-origin: 0px 0px;
     -webkit-transform: translate(186px, -70px) rotate(30deg);
             transform: translate(186px, -70px) rotate(30deg);
   }
   
   #owl:hover #ear1{
     -webkit-transition: all 0.4s;
     transition: all 0.4s;
     -webkit-transform: translate(-5px, -5px);
             transform: translate(-5px, -5px);
   }
   
   #owl:hover #ear2{
     -webkit-transition: all 0.4s;
     transition: all 0.4s;
     -webkit-transform: translate(5px, -5px);
             transform: translate(5px, -5px);
   }
   
   #owl:hover #eye1_4{
     -webkit-transform: scale(0.3);
             transform: scale(0.3);
   }
   
   #owl:hover #eye1_4{
     -webkit-transform: scale(0.27);
             transform: scale(0.27);
   }
   
   #owl:hover #eye2_4{
     -webkit-transform: scale(0.27);
             transform: scale(0.27);
   }</style>
   </head>
   <body id="background">   
     <div id="container">
     <div id="owl">
       <div id = "ear1">
         <div id = "ear1_1"></div>
         <div id = "ear1_2"></div>
         <div id = "ear1_3"></div>
       </div>
       <div id = "ear2">
         <div id = "ear2_1"></div>
         <div id = "ear2_2"></div>
         <div id = "ear2_3"></div>
       </div>
        <div id="leg1">
         <div id = "leg1_1"></div>
         <div id = "leg1_2"></div>
         <div id = "leg1_3"></div>
       </div>
       <div id="leg2">
         <div id = "leg2_1"></div>
         <div id = "leg2_2"></div>
         <div id = "leg2_3"></div>
       </div>  
       <div id = "owl_body"></div>
       <div id = "eyes">
           <div id = "eye1_1"></div><div id = "eye2_1"></div>
           <div id = "beak"></div>
           <div id = "beak1"></div>
           <div id = "eye1_2"></div><div id = "eye2_2"></div>
           <div id = "eye1_3"></div><div id = "eye2_3"></div>
           <div id = "eye1_4"></div><div id = "eye2_4"></div>
       </div>
       <div id = "belly">
         <div id="bellyContainer"></div>
         <div id="b11"></div><div id="b11_1"></div>
         <div id="b12"></div><div id="b12_1"></div>
         <div id="b13"></div><div id="b13_1"></div>
   
         <div id="b21"></div><div id="b21_1"></div>
         <div id="b22"></div><div id="b22_1"></div>
         <div id="b23"></div><div id="b23_1"></div>
         <div id="b24"></div><div id="b24_1"></div>
   
         <div id="b31"></div><div id="b31_1"></div>
         <div id="b32"></div><div id="b32_1"></div>
         <div id="b33"></div><div id="b33_1"></div>
       </div>
       <div id="wing1">
         <div id="wing1_1"></div>
         <div id="wing1_2"></div>
         <div id="wing1_3"></div>
       </div>
       <div id="wing2">
         <div id="wing2_1"></div>
         <div id="wing2_2"></div>
         <div id="wing2_3"></div>
       </div> 
     </div>
     </div>
   </body>
   <script>
   var owl = document.getElementById('owl');
   owl.style.transform = "translate(-220px, 150px) scale(0.5)";
   
   var eye1 = document.getElementById('eye1_4');
   var eye2 = document.getElementById('eye2_4');
   
   //The eyes movements
   window.onmousemove = function(e){
     
     if (e.clientX > window.innerWidth/2 + 90) {
       eye1.style.left = "285px";
       eye2.style.left = "436px";
     }
   
     if (e.clientX < window.innerWidth/2 - 100) {
         eye1.style.left = "245px";
         eye2.style.left = "396px";
     }
     
     if ((e.clientX <= window.innerWidth/2 + 90) && (e.clientX >= window.innerWidth/2 - 100)) {
        eye1.style.left = "265px";
        eye2.style.left = "416px";
     }
      
      if (e.clientY > 170) {
       eye1.style.top = "90px";
       eye2.style.top = "90px";
     }
   
     if (e.clientY < 300) {
         eye1.style.top = "50px";
         eye2.style.top = "50px";
     }
     
     if ((e.clientY <= 300) && (e.clientY >= 170)) {
        eye1.style.top = "70px";
        eye2.style.top = "70px";
     }    
     
   }
   </script>
   </body>
   </html>



计算图
----------------
.. toctree::
       :maxdepth: 3
       
       /02_TensorFlow_Way/01_Operations_as_a_Computational_Graph/index

我们会展示如何在计算图中创建一个算符，并用Tensorboard来展示它。

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





   
