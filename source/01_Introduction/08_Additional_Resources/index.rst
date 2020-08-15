Official Resources
------------------

 -  `TensorFlow Python API <https://www.tensorflow.org/api_docs/python/>`_
 -  `TensorFlow on Github <https://github.com/tensorflow/tensorflow>`_
 -  `TensorFlow Tutorials <https://www.tensorflow.org/tutorials/>`_
 -  `Udacity Deep Learning Class <https://www.udacity.com/course/deep-learning--ud730>`_
 -  `TensorFlow Playground <http://playground.tensorflow.org/>`_

.. raw:: html

 </head>
 <body>
   <!-- GitHub link -->
   <a class="github-link" href="https://github.com/tensorflow/playground" title="Source on GitHub">
     <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 60.5 60.5" width="60" height="60">
       <polygon class="bg" points="60.5,60.5 0,0 60.5,0 "/>
       <path class="icon" d="M43.1,5.8c-6.6,0-12,5.4-12,12c0,5.3,3.4,9.8,8.2,11.4c0.6,0.1,0.8-0.3,0.8-0.6c0-0.3,0-1,0-2c-3.3,0.7-4-1.6-4-1.6c-0.5-1.4-1.3-1.8-1.3-1.8c-1.1-0.7,0.1-0.7,0.1-0.7c1.2,0.1,1.8,1.2,1.8,1.2c1.1,1.8,2.8,1.3,3.5,1c0.1-0.8,0.4-1.3,0.8-1.6c-2.7-0.3-5.5-1.3-5.5-5.9c0-1.3,0.5-2.4,1.2-3.2c-0.1-0.3-0.5-1.5,0.1-3.2c0,0,1-0.3,3.3,1.2c1-0.3,2-0.4,3-0.4c1,0,2,0.1,3,0.4c2.3-1.6,3.3-1.2,3.3-1.2c0.7,1.7,0.2,2.9,0.1,3.2c0.8,0.8,1.2,1.9,1.2,3.2c0,4.6-2.8,5.6-5.5,5.9c0.4,0.4,0.8,1.1,0.8,2.2c0,1.6,0,2.9,0,3.3c0,0.3,0.2,0.7,0.8,0.6c4.8-1.6,8.2-6.1,8.2-11.4C55.1,11.2,49.7,5.8,43.1,5.8z"/>
     </svg>
   </a>
   <!-- Header -->
   <header>
     <h1 class="l--page">派哥的超级森林实验室</h1>
   </header>
 
   <!-- Top Controls -->
   <div id="top-controls">
     <div class="container l--page">
       <div class="timeline-controls">
         <button class="mdl-button mdl-js-button mdl-button--icon ui-resetButton" id="reset-button" title="Reset the network">
           <i class="material-icons">replay</i>
         </button>
         <button class="mdl-button mdl-js-button mdl-button--fab mdl-button--colored ui-playButton" id="play-pause-button" title="Run/Pause">
           <i class="material-icons">play_arrow</i>
           <i class="material-icons">pause</i>
         </button>
         <button class="mdl-button mdl-js-button mdl-button--icon ui-stepButton" id="next-step-button" title="Step">
           <i class="material-icons">skip_next</i>
         </button>
       </div>
       <div class="control">
         <span class="label">Epoch</span>
         <span class="value" id="iter-number"></span>
       </div>
       <div class="control ui-learningRate">
         <label for="learningRate">Learning rate</label>
         <div class="select">
           <select id="learningRate">
             <option value="0.00001">0.00001</option>
             <option value="0.0001">0.0001</option>
             <option value="0.001">0.001</option>
             <option value="0.003">0.003</option>
             <option value="0.01">0.01</option>
             <option value="0.03">0.03</option>
             <option value="0.1">0.1</option>
             <option value="0.3">0.3</option>
             <option value="1">1</option>
             <option value="3">3</option>
             <option value="10">10</option>
           </select>
         </div>
       </div>
       <div class="control ui-activation">
         <label for="activations">Activation</label>
         <div class="select">
           <select id="activations">
             <option value="relu">ReLU</option>
             <option value="tanh">Tanh</option>
             <option value="sigmoid">Sigmoid</option>
             <option value="linear">Linear</option>
           </select>
         </div>
       </div>
       <div class="control ui-regularization">
         <label for="regularizations">Regularization</label>
         <div class="select">
           <select id="regularizations">
             <option value="none">None</option>
             <option value="L1">L1</option>
             <option value="L2">L2</option>
           </select>
         </div>
       </div>
       <div class="control ui-regularizationRate">
         <label for="regularRate">Regularization rate</label>
         <div class="select">
           <select id="regularRate">
             <option value="0">0</option>
             <option value="0.001">0.001</option>
             <option value="0.003">0.003</option>
             <option value="0.01">0.01</option>
             <option value="0.03">0.03</option>
             <option value="0.1">0.1</option>
             <option value="0.3">0.3</option>
             <option value="1">1</option>
             <option value="3">3</option>
             <option value="10">10</option>
           </select>
         </div>
       </div>
       <div class="control ui-problem">
         <label for="problem">Problem type</label>
         <div class="select">
           <select id="problem">
             <option value="classification">Classification</option>
             <option value="regression">Regression</option>
           </select>
         </div>
       </div>
     </div>
   </div>
 
   <!-- Main Part -->
   <div id="main-part" class="l--page">
 
     <!--  Data Column-->
     <div class="column data">
       <h4>
         <span>Data</span>
       </h4>
       <div class="ui-dataset">
         <p>Which dataset do you want to use?</p>
         <div class="dataset-list">
           <div class="dataset" title="Circle">
             <canvas class="data-thumbnail" data-dataset="circle"></canvas>
           </div>
           <div class="dataset" title="Exclusive or">
             <canvas class="data-thumbnail" data-dataset="xor"></canvas>
           </div>
           <div class="dataset" title="Gaussian">
             <canvas class="data-thumbnail" data-dataset="gauss"></canvas>
           </div>
           <div class="dataset" title="Spiral">
             <canvas class="data-thumbnail" data-dataset="spiral"></canvas>
           </div>
           <div class="dataset" title="Plane">
             <canvas class="data-thumbnail" data-regDataset="reg-plane"></canvas>
           </div>
           <div class="dataset" title="Multi gaussian">
             <canvas class="data-thumbnail" data-regDataset="reg-gauss"></canvas>
           </div>
         </div>
       </div>
       <div>
         <div class="ui-percTrainData">
           <label for="percTrainData">Ratio of training to test data:&nbsp;&nbsp;<span class="value">XX</span>%</label>
           <p class="slider">
             <input class="mdl-slider mdl-js-slider" type="range" id="percTrainData" min="10" max="90" step="10">
           </p>
         </div>
         <div class="ui-noise">
           <label for="noise">Noise:&nbsp;&nbsp;<span class="value">XX</span></label>
           <p class="slider">
             <input class="mdl-slider mdl-js-slider" type="range" id="noise" min="0" max="50" step="5">
           </p>
         </div>
         <div class="ui-batchSize">
           <label for="batchSize">Batch size:&nbsp;&nbsp;<span class="value">XX</span></label>
           <p class="slider">
             <input class="mdl-slider mdl-js-slider" type="range" id="batchSize" min="1" max="30" step="1">
           </p>
         </div>
           <button class="basic-button" id="data-regen-button" title="Regenerate data">
             Regenerate
           </button>
       </div>
     </div>
 
     <!-- Features Column -->
     <div class="column features">
       <h4>Features</h4>
       <p>Which properties do you want to feed in?</p>
       <div id="network">
         <svg id="svg" width="510" height="450">
           <defs>
             <marker id="markerArrow" markerWidth="7" markerHeight="13" refX="1" refY="6" orient="auto" markerUnits="userSpaceOnUse">
               <path d="M2,11 L7,6 L2,2" />
             </marker>
           </defs>
         </svg>
         <!-- Hover card -->
         <div id="hovercard">
           <div style="font-size:10px">Click anywhere to edit.</div>
           <div><span class="type">Weight/Bias</span> is <span class="value">0.2</span><span><input type="number"/></span>.</div>
         </div>
         <div class="callout thumbnail">
           <svg viewBox="0 0 30 30">
             <defs>
               <marker id="arrow" markerWidth="5" markerHeight="5" refx="5" refy="2.5" orient="auto" markerUnits="userSpaceOnUse">
                 <path d="M0,0 L5,2.5 L0,5 z"/>
               </marker>
             </defs>
             <path d="M12,30C5,20 2,15 12,0" marker-end="url(#arrow)">
           </svg>
           <div class="label">
             This is the output from one <b>neuron</b>. Hover to see it larger.
           </div>
         </div>
         <div class="callout weights">
           <svg viewBox="0 0 30 30">
             <defs>
               <marker id="arrow" markerWidth="5" markerHeight="5" refx="5" refy="2.5" orient="auto" markerUnits="userSpaceOnUse">
                 <path d="M0,0 L5,2.5 L0,5 z"/>
               </marker>
             </defs>
             <path d="M12,30C5,20 2,15 12,0" marker-end="url(#arrow)">
           </svg>
           <div class="label">
             The outputs are mixed with varying <b>weights</b>, shown by the thickness of the lines.
           </div>
         </div>
       </div>
     </div>
 
     <!-- Hidden Layers Column -->
     <div class="column hidden-layers">
       <h4>
         <div class="ui-numHiddenLayers">
           <button id="add-layers" class="mdl-button mdl-js-button mdl-button--icon">
             <i class="material-icons">add</i>
           </button>
           <button id="remove-layers" class="mdl-button mdl-js-button mdl-button--icon">
             <i class="material-icons">remove</i>
           </button>
         </div>
         <span id="num-layers"></span>
         <span id="layers-label"></span>
       </h4>
       <div class="bracket"></div>
     </div>
 
     <!-- Output Column -->
     <div class="column output">
       <h4>Output</h4>
       <div class="metrics">
         <div class="output-stats ui-percTrainData">
           <span>Test loss</span>
           <div class="value" id="loss-test"></div>
         </div>
         <div class="output-stats train">
           <span>Training loss</span>
           <div class="value" id="loss-train"></div>
         </div>
         <div id="linechart"></div>
       </div>
       <div id="heatmap"></div>
       <div style="float:left;margin-top:20px">
         <div style="display:flex; align-items:center;">
 
           <!-- Gradient color scale -->
           <div class="label" style="width:105px; margin-right: 10px">
             Colors shows data, neuron and weight values.
           </div>
           <svg width="150" height="30" id="colormap">
             <defs>
               <linearGradient id="gradient" x1="0%" y1="100%" x2="100%" y2="100%">
                 <stop offset="0%" stop-color="#f59322" stop-opacity="1"></stop>
                 <stop offset="50%" stop-color="#e8eaeb" stop-opacity="1"></stop>
                 <stop offset="100%" stop-color="#0877bd" stop-opacity="1"></stop>
               </linearGradient>
             </defs>
             <g class="core" transform="translate(3, 0)">
               <rect width="144" height="10" style="fill: url('#gradient');"></rect>
             </g>
           </svg>
         </div>
         <br/>
         <div style="display:flex;">
           <label class="ui-showTestData mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect" for="show-test-data">
             <input type="checkbox" id="show-test-data" class="mdl-checkbox__input" checked>
             <span class="mdl-checkbox__label label">Show test data</span>
           </label>
           <label class="ui-discretize mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect" for="discretize">
             <input type="checkbox" id="discretize" class="mdl-checkbox__input" checked>
             <span class="mdl-checkbox__label label">Discretize output</span>
           </label>
         </div>
       </div>
     </div>
 
 

Github Tutorials and Examples
-----------------------------

 - `Tutorials by pkmital <https://github.com/pkmital/tensorflow_tutorials>`_
 - `Tutorials by nlintz <https://github.com/nlintz/TensorFlow-Tutorials>`_
 - `Examples by americdamien <https://github.com/aymericdamien/TensorFlow-Examples>`_
 - `TensorFlow Workshop by amygdala <https://github.com/amygdala/tensorflow-workshop>`_

Deep Learning Resources
-----------------------

 - `Efficient Back Prop by Yann LeCun, et. al. <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
 - `Online Deep Learning Book, MIT Press <http://www.deeplearningbook.org/>`_
 - `An Overview of Gradient Descent Algorithms by Sebastian Ruder <http://sebastianruder.com/optimizing-gradient-descent/>`_
 - `Stochastic Optimization by John Duchi, et. al. <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_
 - `ADADELTA Method by Matthew Zeiler <http://arxiv.org/abs/1212.5701>`_
 - `A Friendly Introduction to Cross-Entropy Loss by Rob DiPietro <http://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/>`_


Additional Resources
---------------------

 - `A Curated List of Dedicated TensorFlow Resources <https://github.com/jtoy/awesome-tensorflow/>`_

Arxiv Papers
-------------

 - `TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems <http://arxiv.org/abs/1603.04467>`_
 - `TensorFlow: A system for large-scale machine learning <http://arxiv.org/abs/1605.08695>`_
 - `Distributed TensorFlow with MPI <https://arxiv.org/abs/1603.02339>`_
 - `Comparative Study of Deep Learning Software Frameworks <https://arxiv.org/abs/1511.06435>`_
 - `Wide & Deep Learning for Recommender Systems <https://arxiv.org/abs/1606.07792>`_
