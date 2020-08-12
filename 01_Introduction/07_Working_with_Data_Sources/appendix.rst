Ham/Spam Text Dataset(垃圾邮件分类, UCI)
----------------------------

我们会用到加州大学艾文分校机器学习数据库也建立了一个垃圾邮件分类的数据库。我们可以获取.zip 文件， 并获取相应的数据。以下是它的链接 `Ham/Spam Text Dataset <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>`_ 。顺便提一句，如果一个数据点代表（ :code:`spam` 或者不想要的广告）, 

We will use another UCI ML Repository dataset called the SMS Spam Collection. You can 
read about it `here <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>`. 
As a sidenote about common terms, when predicting if a data point represents 'spam' 
(or unwanted advertisement), the alternative is called 'ham' (or useful information).

This is a great dataset for predicting a binary outcome (spam/ham) from a textual input.
This will be very useful for short text sequences for Natural Language Processing 
(Ch 7) and Recurrent Neural Networks (Ch 9).

.. code:: python

  >>> import requests
  >>> import io
  >>> from zipfile import ZipFile

  # Get/read zip file
  >>> zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
  >>> r = requests.get(zip_url)
  >>> z = ZipFile(io.BytesIO(r.content))
  >>> file = z.read('SMSSpamCollection')
  # Format Data
  >>> text_data = file.decode()
  >>> text_data = text_data.encode('ascii',errors='ignore')
  >>> text_data = text_data.decode().split('\n')
  >>> text_data = [x.split('\t') for x in text_data if len(x)>=1]
  >>> [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
  >>> print(len(text_data_train))
  5574
  >>> print(set(text_data_target))
  {'spam', 'ham'}
  >>> print(text_data_train[1])
  Ok lar... Joking wif u oni...

  
Movie Review Data (Cornell)
---------------------------
The Movie Review database, collected by Bo Pang and Lillian Lee (researchers at Cornell),
serves as a great dataset to use for predicting a numerical number from textual inputs.


You can read more about the dataset and papers using it `here <https://www.cs.cornell.edu/people/pabo/movie-review-data/>`

.. code:: python

  >>> import requests
  >>> import io
  >>> import tarfile

  >>> movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
  >>> r = requests.get(movie_data_url)
  # Stream data into temp object
  >>> stream_data = io.BytesIO(r.content)
  >>> tmp = io.BytesIO()
  >>> while True:
  ...    s = stream_data.read(16384)
  ...    if not s:  
  ...          break
  ...    tmp.write(s)
  stream_data.close()
  tmp.seek(0)
  # Extract tar file
  tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
  pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
  neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
  # Save pos/neg reviews
  pos_data = []
  for line in pos:
      pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
  neg_data = []
  for line in neg:
      neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
  tar_file.close()
  
  print(len(pos_data))
  print(len(neg_data))
  print(neg_data[0])
  
the output::

  5331
  5331
  simplistic , silly and tedious . 

The Complete Works of William Shakespeare (Gutenberg Project)
-------------------------------------------------------------
For training a TensorFlow Model to create text, we will train it on the complete works
of William Shakespeare. This can be accessed through the good work of the Gutenberg 
Project. The Gutenberg Project frees many non-copyright books by making them accessible
for free from the hard work of volunteers.

You can read more about the Shakespeare works `here <http://www.gutenberg.org/ebooks/100>`_

.. code:: python

  # The Works of Shakespeare Data
  import requests

  shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
  # Get Shakespeare text
  response = requests.get(shakespeare_url)
  shakespeare_file = response.content
  # Decode binary into string
  shakespeare_text = shakespeare_file.decode('utf-8')
  # Drop first few descriptive paragraphs.
  shakespeare_text = shakespeare_text[7675:]
  print(len(shakespeare_text))

the output::

  5582212
  
English-German Sentence Translation Database (Manythings/Tatoeba)
-----------------------------------------------------------------

The `Tatoeba Project <http://www.manythings.org/corpus/about.html#info>` is also run 
by volunteers and is set to make the most bilingual sentence translations available 
between many different languages. Manythings.org compiles the data and makes it 
accessible.



`More bilingual sentence pairs <http://www.manythings.org/bilingual/>`_

.. code:: python

  # English-German Sentence Translation Data
  import requests
  import io
  from zipfile import ZipFile
  sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
  r = requests.get(sentence_url)
  z = ZipFile(io.BytesIO(r.content))
  file = z.read('deu.txt')
  # Format Data
  eng_ger_data = file.decode()
  eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
  eng_ger_data = eng_ger_data.decode().split('\n')
  eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
  [english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
  print(len(english_sentence))
  print(len(german_sentence))
  print(eng_ger_data[10])

the output::

  147788
  147788
  ['I won!', 'Ich hab gewonnen!']
  
CIFAR-10 Data
--------------

The `CIFAR-10 data <https://www.cs.toronto.edu/~kriz/cifar.html>`_ contains 60,000 
32x32 color images of 10 classes collected by Alex Krizhevsky, Vinod Nair, and 
Geoffrey Hinton. Alex Krizhevsky maintains the page referenced here. This is such a
common dataset, that there are built in functions in TensorFlow to access this data 
(the keras wrapper has these commands). Note that the keras wrapper for these functions
automatically splits the images into a 50,000 training set and a 10,000 test set.

.. code:: python

  from PIL import Image
  # Running this command requires an internet connection and a few minutes to download all the images.
  (X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()

the output:: 

  Downloading data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  The ten categories are (in order):
  
  Airplane
  Automobile
  Bird
  Car
  Deer
  Dog
  Frog
  Horse
  Ship
  Truck

.. code:: python
  
  X_train.shape
  y_train.shape
  y_train[0,] # this is a frog
  # Plot the 0-th image (a frog)
  %matplotlib inline
  img = Image.fromarray(X_train[0,:,:,:])
  plt.imshow(img)

the output::

  (50000, 32, 32, 3)
  (50000, 1)
  array([6], dtype=uint8)
  <matplotlib.image.AxesImage at 0x7ffb48a47400>
  
  
.. raw:: html

  <table>
    <tr>
        <td class="cifar-class-name">飞机</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">汽车</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">小鸟</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/bird10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">猫</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/cat10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">鹿</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/deer10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">狗</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">青蛙</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/frog10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">马</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/horse10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">船</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/ship10.png" class="cifar-sample" /></td>
    </tr>
    <tr>
        <td class="cifar-class-name">卡车</td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck1.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck2.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck3.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck4.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck5.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck6.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck7.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck8.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck9.png" class="cifar-sample" /></td>
        <td><img src="http://www.cs.toronto.edu/~kriz/cifar-10-sample/truck10.png" class="cifar-sample" /></td>
    </tr>
  </table>
