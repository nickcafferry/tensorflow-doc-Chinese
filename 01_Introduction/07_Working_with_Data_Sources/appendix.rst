MNIST 手写数据代码补充
---------------------

因为篇幅原因，在介绍MNIST手写数据库时，没有对手写的数字可视化进行详细介绍，这里补充一些代码：

.. code:: bash
  
  $ dataset_url = "https://modelarts-cnnorth1-market-dataset.obs.cn-north-1.myhuaweicloud.com/dataset-market/Mnist-Data-Set/archiver/Mnist-Data-Set.zip"
  $ dataset_local_path = 'dataset/'
  $ wget {dataset_url} -P {dataset_local_path}
  $ unzip -d {dataset_local_path} -o {dataset_local_name}
  $ ls $dataset_local_path
  Mnist-Data-Set.zip	       train-images-idx3-ubyte
  t10k-images-idx3-ubyte	   train-images-idx3-ubyte.gz
  t10k-images-idx3-ubyte.gz  train-labels-idx1-ubyte
  t10k-labels-idx1-ubyte	   train-labels-idx1-ubyte.gz
  t10k-labels-idx1-ubyte.gz
  
  $ train_image = os.path.join(dataset_local_path, 'train-images-idx3-ubyte')
  $ train_lable = os.path.join(dataset_local_path, 'train-labels-idx1-ubyte')
  $ eval_image  = os.path.join(dataset_local_path, 't10k-images-idx3-ubyte')
  $ eval_lable  = os.path.join(dataset_local_path, 't10k-labels-idx1-ubyte')
  
  $ pip install mxnet
  Looking in indexes: http://repo.myhuaweicloud.com/repository/pypi/simple
  Collecting mxnet
  Downloading http://repo.myhuaweicloud.com/repository/pypi/packages/81/f5/d79b5b40735086ff1100c680703e0f3efc830fa455e268e9e96f3c857e93/mxnet-1.6.0-py2.py3-none-any.whl (68.7 MB)
     |████████████████████████████████| 68.7 MB 7.1 MB/s eta 0:00:01MB/s eta 0:00:31
  Requirement already satisfied: numpy<2.0.0,>1.16.0 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from mxnet) (1.18.4)
  Requirement already satisfied: requests<3,>=2.20.0 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from mxnet) (2.23.0)
  Requirement already satisfied: graphviz<0.9.0,>=0.8.1 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from mxnet) (0.8.1)
  Requirement already satisfied: chardet<4,>=3.0.2 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from requests<3,>=2.20.0->mxnet) (3.0.4)
  Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from  requests<3,>=2.20.0->mxnet) (1.22)
  Requirement already satisfied: certifi>=2017.4.17 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from requests<3,>=2.20.0->mxnet) (2018.1.18)
  Requirement already satisfied: idna<3,>=2.5 in /home/ma-user/anaconda3/envs/TensorFlow-2.1.0/lib/python3.6/site-packages (from requests<3,>=2.20.0->mxnet) (2.6)
  Installing collected packages: mxnet
  Successfully installed mxnet-1.6.0
  WARNING: You are using pip version 20.1.1; however, version 20.2.1 is available.
  You should consider upgrading via the '/home/ma-user/anaconda3/envs/TensorFlow-2.1.0/bin/python -m pip install --upgrade pip' command.

.. code:: python
  
  >>> import mxnet as mx
  >>> batch_size = 128
  >>> train_data = mx.io.MNISTIter(image = train_image,
  ...                              label = train_lable,
  ...                              data_shqpe = (1,28,28),
  ...                              batch_size = batch_size,
  ...                              shuffle = True,
  ...                              flat    = False,
  ...                              silent  = False)

  >>> eval_data  = mx.io.MNISTIter(image = eval_image,
  ...                              label = eval_lable,
  ...                              data_shqpe = (1,28,28),
  ...                              batch_size = batch_size,
  ...                              shuffle = False)
  
  >>> import matplotlib.pyplot as plt
  >>> train_data.reset()
  >>> next_batch  =  train_data.next()

  >>> for i in range(128):
  ...   show_image  =  next_batch.data[0][i].asnumpy() * 255      
  ...   show_image  =  show_image.astype('uint8').reshape(28, 28)
  ...   plt.figure(168)
  ...   plt.subplot(16,8,1+i)
  ...   plt.imshow(show_image, cmap = plt.cm.gray)
  
  >>> plt.savefig('Handwriting.png', dpi=1000)
  >>> plt.show()


Ham/Spam Text Dataset(垃圾邮件分类, UCI)
----------------------------

我们会用到加州大学艾文分校机器学习数据库也建立了一个垃圾邮件分类的数据库。我们可以获取.zip 文件， 并获取相应的数据。以下是它的链接 `Ham/Spam Text Dataset <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>`_ 。顺便提一句，如果一个数据点代表（ :code:`spam` 或者不想要的广告）, 那么另外一个就是
'ham'.

:strong:`Ham/Spam Text Dataset` 是一个从文本输入当中预测二进制结果（spam or ham）一个很好的数据集。 这将对自然语言处理的短文本处理(第七章)和递归神经网络(第九章)
很有用。

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
  
  >>> text_data_train[0:10]
  ['Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
  'Ok lar... Joking wif u oni...',
  "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
  'U dun say so early hor... U c already then say...',
  "Nah I don't think he goes to usf, he lives around here though",
  "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 1.50 to rcv",
  'Even my brother is not like to speak with me. They treat me like aids patent.',
  "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune",
  'WINNER!! As a valued network customer you have been selected to receivea 900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.',
  'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030']


`电影评论数据库 <http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>`_ (Stanford)
---------------------------

这是个二元情感的数据分类库，包含比之前更多的数据。 这里，我们提供25,000 高度极化的电影评论作为训练集，25,000数据评论作为测试集。还有一些并没有标签的数据也会作为使用。原文本和已经处理过得数据形式也提供了，你可以查看README文件更多细节。

如果你想要理解更多，请点击 ` 这里 <http://ai.stanford.edu/~amaas/data/sentiment/index.html>`_ 

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
  >>> stream_data.close()
  >>> tmp.seek(0)
  # Extract tar file
  >>> tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
  >>> pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
  >>> neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
  # Save pos/neg reviews
  >>> pos_data = []
  >>> for line in pos:
  ...     pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
  >>> neg_data = []
  >>> for line in neg:
  ...     neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
  >>> tar_file.close()
  
  # 数据过大，网速不给力，无法给出结果
  >>> print(len(pos_data))
  >>> print(len(neg_data))
  >>> print(neg_data[0]) 

莎士比亚全集 (古登堡计划)
-------------------------------------------------------------

Project Gutenberg(古登堡计划)是为了出版电子版本的免费书籍而发起的。这个计划把莎士比亚所有作品都编撰在一起。为了训练一个TensorFlow的模型来闯将文本，我们把这个模型放在威廉莎士比亚全集中训练。古登堡计划有很多志愿者为了实现无版权书籍的免费使用，花费了很多精力。在这里，我们可以通过Python的脚本来获取文本文件。

如果你想了解更多莎士比亚全集，请点击 `这里 <http://www.gutenberg.org/ebooks/100>`_ 。

.. code:: python

  # 莎士比亚全集数据
  >>> import requests

  >>> shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
  # 获取莎士比亚文本
  >>> response = requests.get(shakespeare_url)
  >>> shakespeare_file = response.content
  # 将二进制转化为字符串
  >>> shakespeare_text = shakespeare_file.decode('utf-8')
  # 截取几个描述性的段落
  >>> shakespeare_text = shakespeare_text[7675:]
  >>> print(len(shakespeare_text))
  5582212


英语-德语 文本翻译数据库 (Manythings/Tatoeba)
-----------------------------------------------------------------

`Tatoeba Project <http://www.manythings.org/corpus/about.html#info>`_ 也是由志愿者发起的，旨在
让很多不同的语言之间双语翻译可以实现。Manythings.org 组织编撰这些数据,使得句对句翻译可以下载。在这里
我们用到是英语对德语翻译，但是你可以自己想选哪一个就选哪一个。

`双语句对 <http://www.manythings.org/bilingual/>`_

.. code:: python

  # English-German 句对句翻译数据
  >>> import requests
  >>> import io
  >>> from zipfile import ZipFile
  >>> sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
  >>> r = requests.get(sentence_url)
  >>> z = ZipFile(io.BytesIO(r.content))
  >>> file = z.read('deu.txt')
  # 格式化数据
  >>> eng_ger_data = file.decode()
  >>> eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
  >>> eng_ger_data = eng_ger_data.decode().split('\n')
  >>> eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
  >>> [english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
  >>> print(len(english_sentence))
  147788
  >>> print(len(german_sentence))
  147788
  >>> print(eng_ger_data[10])
  ['I won!', 'Ich hab gewonnen!']
  
CIFAR-10 数据库
--------------

加拿大高级研究所(Canadian Institute For Advanced Research, CIFAR)发布了一个图像集，包含了8千万已标记的图片(每个图片尺寸都是32x32像素). 总共有十大类不同图片，分别是飞机，汽车，鸟类，车，鹿，狗，青蛙，马，船只，卡车。CIFAR-10的一个含有60,000图片的子数据集。训练集有50,000,测试集有10,000。你可以手动下载该数据库 `CIFAR-10 data <https://www.cs.toronto.edu/~kriz/cifar.html>`_ ，可以通过下面的代码来获取该数据库。

.. code:: python

  >>> from PIL import Image
  # 运行下面的命令需要网络，下载可能要花上不上时间
  >>> (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  Downloading data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

.. code:: python
  
  >>> X_train.shape
  (50000, 32, 32, 3)
  >>> y_train.shape
  (50000, 1)
  >>> y_train[0,] # 这是个青蛙
  # Plot the 0-th image (a frog)
  $ %matplotlib inline
  >>> img = Image.fromarray(X_train[0,:,:,:])
  >>> plt.imshow(img)
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
