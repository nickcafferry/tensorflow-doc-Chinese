对本书的大部分内容来说，我们会依赖于数据集来拟合机器学习算法。这部分将会告诉你如何通过TensorFlow和Python来获取这些数据资源。

在TensorFlow中，有一些数据资源是Python库中内置的，有些是需要Python的脚本来下载，有些是需要在网上手动下载。 当然，所有这些数据集都是需要网络来获取数据。

首先，我们需要对TensorFlow的 :code:`graph session` 进行初始化：

.. code:: python

  >>> import tensorflow.compat.v1 as tf
  >>> import matplotlib.pyplot as plt
  >>> import numpy as np
  >>> from tensorflow.python.framework import ops
  >>> ops.reset_default_graph()
  >>> tf.disable_eager_execution()
  >>> sess = tf.Session()

Iris Dataset(鸢尾属植物数据集)
------------------------------------

这个数据集( `Iris Dataset <http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris>`_ )无可置疑地是最经典的用于机器学习的数据集，而且可能扩展到所有统计学。这个数据集采集了三种鸢尾花的 :strong:`sepal length` (花萼长度)， :strong:`sepal width` (花萼宽度)，:strong:`petal length` (花瓣长度)，:strong:`petal width` (花瓣宽度)。这三种鸢尾花分别是Iris Setosa(山鸢尾)，Iris Versicolour(杂色鸢尾)，Iris Virginica(维吉尼亚鸢尾)。总共有150项测量，每种鸢尾花有50项。为了在Python中使用这些数据集，我们使用Scikit Learn中的数据函数。

.. code:: python

    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>> iris = load_iris()
    >>> print(len(iris.data))
    150
    >>> print(len(iris.target))
    150
    >>> print(iris.data[0])
    [5.1 3.5 1.4 0.2]
    >>> print(set(iris.target))
    {0, 1, 2}
    >>> pd.DataFrame(data=iris.data, columns=iris.feature_names)

.. raw:: html

    <div>
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepal length (cm)</th>
          <th>sepal width (cm)</th>
          <th>petal length (cm)</th>
          <th>petal width (cm)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>5</th>
          <td>5.4</td>
          <td>3.9</td>
          <td>1.7</td>
          <td>0.4</td>
        </tr>
        <tr>
          <th>6</th>
          <td>4.6</td>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.3</td>
        </tr>
        <tr>
          <th>7</th>
          <td>5.0</td>
          <td>3.4</td>
          <td>1.5</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>8</th>
          <td>4.4</td>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>9</th>
          <td>4.9</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.1</td>
        </tr>
        <tr>
          <th>10</th>
          <td>5.4</td>
          <td>3.7</td>
          <td>1.5</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>11</th>
          <td>4.8</td>
          <td>3.4</td>
          <td>1.6</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>12</th>
          <td>4.8</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.1</td>
        </tr>
        <tr>
          <th>13</th>
          <td>4.3</td>
          <td>3.0</td>
          <td>1.1</td>
          <td>0.1</td>
        </tr>
        <tr>
          <th>14</th>
          <td>5.8</td>
          <td>4.0</td>
          <td>1.2</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>15</th>
          <td>5.7</td>
          <td>4.4</td>
          <td>1.5</td>
          <td>0.4</td>
        </tr>
        <tr>
          <th>16</th>
          <td>5.4</td>
          <td>3.9</td>
          <td>1.3</td>
          <td>0.4</td>
        </tr>
        <tr>
          <th>17</th>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.3</td>
        </tr>
        <tr>
          <th>18</th>
          <td>5.7</td>
          <td>3.8</td>
          <td>1.7</td>
          <td>0.3</td>
        </tr>
        <tr>
          <th>19</th>
          <td>5.1</td>
          <td>3.8</td>
          <td>1.5</td>
          <td>0.3</td>
        </tr>
        <tr>
          <th>20</th>
          <td>5.4</td>
          <td>3.4</td>
          <td>1.7</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>21</th>
          <td>5.1</td>
          <td>3.7</td>
          <td>1.5</td>
          <td>0.4</td>
        </tr>
        <tr>
          <th>22</th>
          <td>4.6</td>
          <td>3.6</td>
          <td>1.0</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>23</th>
          <td>5.1</td>
          <td>3.3</td>
          <td>1.7</td>
          <td>0.5</td>
        </tr>
        <tr>
          <th>24</th>
          <td>4.8</td>
          <td>3.4</td>
          <td>1.9</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>25</th>
          <td>5.0</td>
          <td>3.0</td>
          <td>1.6</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>26</th>
          <td>5.0</td>
          <td>3.4</td>
          <td>1.6</td>
          <td>0.4</td>
        </tr>
        <tr>
          <th>27</th>
          <td>5.2</td>
          <td>3.5</td>
          <td>1.5</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>28</th>
          <td>5.2</td>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>29</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.6</td>
          <td>0.2</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>120</th>
          <td>6.9</td>
          <td>3.2</td>
          <td>5.7</td>
          <td>2.3</td>
        </tr>
        <tr>
          <th>121</th>
          <td>5.6</td>
          <td>2.8</td>
          <td>4.9</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>122</th>
          <td>7.7</td>
          <td>2.8</td>
          <td>6.7</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>123</th>
          <td>6.3</td>
          <td>2.7</td>
          <td>4.9</td>
          <td>1.8</td>
        </tr>
        <tr>
          <th>124</th>
          <td>6.7</td>
          <td>3.3</td>
          <td>5.7</td>
          <td>2.1</td>
        </tr>
        <tr>
          <th>125</th>
          <td>7.2</td>
          <td>3.2</td>
          <td>6.0</td>
          <td>1.8</td>
        </tr>
        <tr>
          <th>126</th>
          <td>6.2</td>
          <td>2.8</td>
          <td>4.8</td>
          <td>1.8</td>
        </tr>
        <tr>
          <th>127</th>
          <td>6.1</td>
          <td>3.0</td>
          <td>4.9</td>
          <td>1.8</td>
        </tr>
        <tr>
          <th>128</th>
          <td>6.4</td>
          <td>2.8</td>
          <td>5.6</td>
          <td>2.1</td>
        </tr>
        <tr>
          <th>129</th>
          <td>7.2</td>
          <td>3.0</td>
          <td>5.8</td>
          <td>1.6</td>
        </tr>
        <tr>
          <th>130</th>
          <td>7.4</td>
          <td>2.8</td>
          <td>6.1</td>
          <td>1.9</td>
        </tr>
        <tr>
          <th>131</th>
          <td>7.9</td>
          <td>3.8</td>
          <td>6.4</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>132</th>
          <td>6.4</td>
          <td>2.8</td>
          <td>5.6</td>
          <td>2.2</td>
        </tr>
        <tr>
          <th>133</th>
          <td>6.3</td>
          <td>2.8</td>
          <td>5.1</td>
          <td>1.5</td>
        </tr>
        <tr>
          <th>134</th>
          <td>6.1</td>
          <td>2.6</td>
          <td>5.6</td>
          <td>1.4</td>
        </tr>
        <tr>
          <th>135</th>
          <td>7.7</td>
          <td>3.0</td>
          <td>6.1</td>
          <td>2.3</td>
        </tr>
        <tr>
          <th>136</th>
          <td>6.3</td>
          <td>3.4</td>
          <td>5.6</td>
          <td>2.4</td>
        </tr>
        <tr>
          <th>137</th>
          <td>6.4</td>
          <td>3.1</td>
          <td>5.5</td>
          <td>1.8</td>
        </tr>
        <tr>
          <th>138</th>
          <td>6.0</td>
          <td>3.0</td>
          <td>4.8</td>
          <td>1.8</td>
        </tr>
        <tr>
          <th>139</th>
          <td>6.9</td>
          <td>3.1</td>
          <td>5.4</td>
          <td>2.1</td>
        </tr>
        <tr>
          <th>140</th>
          <td>6.7</td>
          <td>3.1</td>
          <td>5.6</td>
          <td>2.4</td>
        </tr>
        <tr>
          <th>141</th>
          <td>6.9</td>
          <td>3.1</td>
          <td>5.1</td>
          <td>2.3</td>
        </tr>
        <tr>
          <th>142</th>
          <td>5.8</td>
          <td>2.7</td>
          <td>5.1</td>
          <td>1.9</td>
        </tr>
        <tr>
          <th>143</th>
          <td>6.8</td>
          <td>3.2</td>
          <td>5.9</td>
          <td>2.3</td>
        </tr>
        <tr>
          <th>144</th>
          <td>6.7</td>
          <td>3.3</td>
          <td>5.7</td>
          <td>2.5</td>
        </tr>
        <tr>
          <th>145</th>
          <td>6.7</td>
          <td>3.0</td>
          <td>5.2</td>
          <td>2.3</td>
        </tr>
        <tr>
          <th>146</th>
          <td>6.3</td>
          <td>2.5</td>
          <td>5.0</td>
          <td>1.9</td>
        </tr>
        <tr>
          <th>147</th>
          <td>6.5</td>
          <td>3.0</td>
          <td>5.2</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>148</th>
          <td>6.2</td>
          <td>3.4</td>
          <td>5.4</td>
          <td>2.3</td>
        </tr>
        <tr>
          <th>149</th>
          <td>5.9</td>
          <td>3.0</td>
          <td>5.1</td>
          <td>1.8</td>
        </tr>
      </tbody>
    </table>
    <p>150 rows x 4 columns</p>
    </div>

.. code:: ipython3

    >>> X = iris.data  #只包括样本的特征，150x4
    >>> y = iris.target  #样本的类型，[0, 1, 2]
    >>> features = iris.feature_names  #4个特征的名称
    >>> targets = iris.target_names  #3类鸢尾花的名称，跟y中的3个数字对应   
    ... plt.figure(figsize=(10, 4))
    ... plt.plot(X[:, 2][y==0], X[:, 3][y==0], 'bs', label=targets[0])
    ... plt.plot(X[:, 2][y==1], X[:, 3][y==1], 'kx', label=targets[1])
    ... plt.plot(X[:, 2][y==2], X[:, 3][y==2], 'ro', label=targets[2])
    ... plt.xlabel(features[2])
    ... plt.ylabel(features[3])
    ... plt.title('Iris Data Set')
    ... plt.legend()
    ... plt.savefig('Iris Data Set.png', dpi=200)
    ... plt.show()

.. raw:: html
  
  <img src="data:image/png;base64,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O42HH36YF154gXvvvZfa2lp+97vf8dJLL+U97tlnn+X4448H4IQTTuC5557r0VdVVRVDhw7llltuYc6cObz66qvs3r07ydMTEZFizJkDLS0wblywKHncuOBz1Avmw4yTVCwyYLHcpnT3ZqC5QJtnCH5RWdKeeuoppk6dyqJFi7jwwgs57rjjmDx5MpdeeikPPvggY3OXgw844AB27NgBBLW8jj76aDZs2MD06dPZsGEDRx99dI++1q1bx5o1azjvvPOor6/n5JNPTvNURUQkjDlzkkl4woyTVCwyIBX1OKRDDilufxjjx4/nO9/5DlOmTGHbtm184Qtf4KGHHmLatGncfPPNHH744QCceuqp/OhHP2Lq1KmsW7eOL3zhC/z6179m2rRp/OY3v+GSSy7p0dekSZM49dRTueGGGzjllFMAer3SJiJS0kqpHta8eTBkSHC1aciQ4LPIAJh75tbE92rSpEne0dHRZd9zzz3HRz/60ZQiKl+aVxFJTPd6WBCsbcriLbV586A5z42fhgZYvjz5eCTTzGyju08q1K6iroyJiEgGlVI9rJaW4vaLhKBkTERE0lVK9bDeeae4/SIhKBkTEZF09Vb3Kov1sAYPLm6/SAhKxkREJF2lVA+r89mOYfeLhKBkTERE0lVK9bCWLw8W63deCRs8WIv3ZcAqKhlbsmQJbW1tXfa1tbWxZMmSlCLK74orruj3sbW1tdEFIiKSlDlzYNMmePfd4DWLiVin5cvhz38G9+BViZgMUEUlY9XV1dTX1+9NyNra2qivr6e6ujrlyLq68cYb0w5BRCRZheqMhalDFkUfUYhinKhiLaX6bUnI6ny4e8lsxx13nHf37LPP9tjXl9WrV/uoUaO8sbHRR40a5atXry7q+O4WLlzo9957r7u7L1q0yO+44w4/99xzvaamxufNm7e33cknn+wLFizw0047zd3dd+3a5WeccYbX1NT4Oeec42+//XaXtp12797t559/vk+dOtXPOOMMf+ONN3zPnj1+wQUX+LRp0/zCCy/0N998M++xvbXrHks+xc6riEi/rVjhXlXlHlxrCraqqmB/mO+j6iOJc0mqjyj7KRcpzAfQ4SHym9QTrGK2KJIxd/fGxkYHvLGxsehju3vhhRd87ty57u5++umn+8KFC/2aa65xd/fPfOYz/tRTT7m7+3ve8x6/55579h7X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
  ">


Low Birthrate Dataset (Hosted on Github)
----------------------------------------

马萨诸塞大学艾摩斯特分校(The university of Massachusetts at Amherst)编撰了很多有趣的统计数据集。其中有一项是测量儿童出生重量和其他人口学数据( `Low Birthrate Dataset <https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat>`_ , "Low Infant Birth Weight Risk Factor Study", 1989, Hosmer and Lemeshow)，以及母亲和家庭历史的医学测量。总共测量了11个变量的189观察数据。这里给出如何通过Python来获取其中的数据:

.. code:: python

    >>> import requests
    >>> birthdata_url='https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    >>> birth_file = requests.get(birthdata_url)
    >>> birth_data = birth_file.text.split('\r\n')
    >>> birth_header = birth_data[0].split('\t')
    >>> birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    >>> print(len(birth_data))
    189
    >>> print(len(birth_data[0]))
    9
    >>> print(birth_header)
    ['LOW', 'AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'BWT']

    >>> import pandas as pd
    >>> pd.DataFrame(data=birth_data, columns=birth_header)

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>LOW</th>
          <th>AGE</th>
          <th>LWT</th>
          <th>RACE</th>
          <th>SMOKE</th>
          <th>PTL</th>
          <th>HT</th>
          <th>UI</th>
          <th>BWT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.0</td>
          <td>28.0</td>
          <td>113.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>709.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.0</td>
          <td>29.0</td>
          <td>130.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1021.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.0</td>
          <td>34.0</td>
          <td>187.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1135.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.0</td>
          <td>25.0</td>
          <td>105.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1330.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.0</td>
          <td>25.0</td>
          <td>85.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1474.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>1.0</td>
          <td>27.0</td>
          <td>150.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1588.0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>1.0</td>
          <td>23.0</td>
          <td>97.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1588.0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>1.0</td>
          <td>24.0</td>
          <td>128.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1701.0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>1.0</td>
          <td>24.0</td>
          <td>132.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1729.0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>1.0</td>
          <td>21.0</td>
          <td>165.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1790.0</td>
        </tr>
        <tr>
          <th>10</th>
          <td>1.0</td>
          <td>32.0</td>
          <td>105.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1818.0</td>
        </tr>
        <tr>
          <th>11</th>
          <td>1.0</td>
          <td>19.0</td>
          <td>91.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1885.0</td>
        </tr>
        <tr>
          <th>12</th>
          <td>1.0</td>
          <td>25.0</td>
          <td>115.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1893.0</td>
        </tr>
        <tr>
          <th>13</th>
          <td>1.0</td>
          <td>16.0</td>
          <td>130.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1899.0</td>
        </tr>
        <tr>
          <th>14</th>
          <td>1.0</td>
          <td>25.0</td>
          <td>92.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1928.0</td>
        </tr>
        <tr>
          <th>15</th>
          <td>1.0</td>
          <td>20.0</td>
          <td>150.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1928.0</td>
        </tr>
        <tr>
          <th>16</th>
          <td>1.0</td>
          <td>21.0</td>
          <td>190.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1928.0</td>
        </tr>
        <tr>
          <th>17</th>
          <td>1.0</td>
          <td>24.0</td>
          <td>155.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1936.0</td>
        </tr>
        <tr>
          <th>18</th>
          <td>1.0</td>
          <td>21.0</td>
          <td>103.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1970.0</td>
        </tr>
        <tr>
          <th>19</th>
          <td>1.0</td>
          <td>20.0</td>
          <td>125.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2055.0</td>
        </tr>
        <tr>
          <th>20</th>
          <td>1.0</td>
          <td>25.0</td>
          <td>89.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2055.0</td>
        </tr>
        <tr>
          <th>21</th>
          <td>1.0</td>
          <td>19.0</td>
          <td>102.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2082.0</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1.0</td>
          <td>19.0</td>
          <td>112.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2084.0</td>
        </tr>
        <tr>
          <th>23</th>
          <td>1.0</td>
          <td>26.0</td>
          <td>117.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2084.0</td>
        </tr>
        <tr>
          <th>24</th>
          <td>1.0</td>
          <td>24.0</td>
          <td>138.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2100.0</td>
        </tr>
        <tr>
          <th>25</th>
          <td>1.0</td>
          <td>17.0</td>
          <td>130.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2125.0</td>
        </tr>
        <tr>
          <th>26</th>
          <td>1.0</td>
          <td>20.0</td>
          <td>120.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>2126.0</td>
        </tr>
        <tr>
          <th>27</th>
          <td>1.0</td>
          <td>22.0</td>
          <td>130.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2187.0</td>
        </tr>
        <tr>
          <th>28</th>
          <td>1.0</td>
          <td>27.0</td>
          <td>130.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2187.0</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1.0</td>
          <td>20.0</td>
          <td>80.0</td>
          <td>1.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>2211.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>159</th>
          <td>0.0</td>
          <td>24.0</td>
          <td>110.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3728.0</td>
        </tr>
        <tr>
          <th>160</th>
          <td>0.0</td>
          <td>19.0</td>
          <td>184.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>3756.0</td>
        </tr>
        <tr>
          <th>161</th>
          <td>0.0</td>
          <td>24.0</td>
          <td>110.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3770.0</td>
        </tr>
        <tr>
          <th>162</th>
          <td>0.0</td>
          <td>23.0</td>
          <td>110.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3770.0</td>
        </tr>
        <tr>
          <th>163</th>
          <td>0.0</td>
          <td>20.0</td>
          <td>120.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3770.0</td>
        </tr>
        <tr>
          <th>164</th>
          <td>0.0</td>
          <td>25.0</td>
          <td>141.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>3790.0</td>
        </tr>
        <tr>
          <th>165</th>
          <td>0.0</td>
          <td>30.0</td>
          <td>112.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3799.0</td>
        </tr>
        <tr>
          <th>166</th>
          <td>0.0</td>
          <td>22.0</td>
          <td>169.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3827.0</td>
        </tr>
        <tr>
          <th>167</th>
          <td>0.0</td>
          <td>18.0</td>
          <td>120.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3856.0</td>
        </tr>
        <tr>
          <th>168</th>
          <td>0.0</td>
          <td>16.0</td>
          <td>170.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3860.0</td>
        </tr>
        <tr>
          <th>169</th>
          <td>0.0</td>
          <td>32.0</td>
          <td>186.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3860.0</td>
        </tr>
        <tr>
          <th>170</th>
          <td>0.0</td>
          <td>18.0</td>
          <td>120.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3884.0</td>
        </tr>
        <tr>
          <th>171</th>
          <td>0.0</td>
          <td>29.0</td>
          <td>130.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3884.0</td>
        </tr>
        <tr>
          <th>172</th>
          <td>0.0</td>
          <td>33.0</td>
          <td>117.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>3912.0</td>
        </tr>
        <tr>
          <th>173</th>
          <td>0.0</td>
          <td>20.0</td>
          <td>170.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3940.0</td>
        </tr>
        <tr>
          <th>174</th>
          <td>0.0</td>
          <td>28.0</td>
          <td>134.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3941.0</td>
        </tr>
        <tr>
          <th>175</th>
          <td>0.0</td>
          <td>14.0</td>
          <td>135.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3941.0</td>
        </tr>
        <tr>
          <th>176</th>
          <td>0.0</td>
          <td>28.0</td>
          <td>130.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3969.0</td>
        </tr>
        <tr>
          <th>177</th>
          <td>0.0</td>
          <td>25.0</td>
          <td>120.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3983.0</td>
        </tr>
        <tr>
          <th>178</th>
          <td>0.0</td>
          <td>16.0</td>
          <td>135.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3997.0</td>
        </tr>
        <tr>
          <th>179</th>
          <td>0.0</td>
          <td>20.0</td>
          <td>158.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>3997.0</td>
        </tr>
        <tr>
          <th>180</th>
          <td>0.0</td>
          <td>26.0</td>
          <td>160.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4054.0</td>
        </tr>
        <tr>
          <th>181</th>
          <td>0.0</td>
          <td>21.0</td>
          <td>115.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4054.0</td>
        </tr>
        <tr>
          <th>182</th>
          <td>0.0</td>
          <td>22.0</td>
          <td>129.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4111.0</td>
        </tr>
        <tr>
          <th>183</th>
          <td>0.0</td>
          <td>25.0</td>
          <td>130.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4153.0</td>
        </tr>
        <tr>
          <th>184</th>
          <td>0.0</td>
          <td>31.0</td>
          <td>120.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4167.0</td>
        </tr>
        <tr>
          <th>185</th>
          <td>0.0</td>
          <td>35.0</td>
          <td>170.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4174.0</td>
        </tr>
        <tr>
          <th>186</th>
          <td>0.0</td>
          <td>19.0</td>
          <td>120.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>4238.0</td>
        </tr>
        <tr>
          <th>187</th>
          <td>0.0</td>
          <td>24.0</td>
          <td>216.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4593.0</td>
        </tr>
        <tr>
          <th>188</th>
          <td>0.0</td>
          <td>45.0</td>
          <td>123.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>4990.0</td>
        </tr>
      </tbody>
    </table>
    <p>189 rows x 9 columns</p>
    </div>


  
波士顿房价数据库(University of California at Irvine)
-----------------------------

卡耐基梅隆大学在它的统计学库中保存了很多数据。其中一项，波士顿房价数据( `Boston Housing data <https://archive.ics.uci.edu/ml/datasets/Housing>`_ )可以通过加利福尼亚艾文分校的机器学习仓库来获取。这里总共有房价的506项观察数据和不同人口学数据，以及住宅性质(14个变量)。这里展示如何在Python中获取这些数据：

.. code:: python

  >>> import requests
  >>> housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
  >>> housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
  >>> housing_file = requests.get(housing_url)
  >>> housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
  >>> print(len(housing_data))
  506
  >>> print(len(housing_data[0]))
  14
  >>> pd.DataFrame(data=housing_data,columns=housing_header)

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>CRIM</th>
          <th>ZN</th>
          <th>INDUS</th>
          <th>CHAS</th>
          <th>NOX</th>
          <th>RM</th>
          <th>AGE</th>
          <th>DIS</th>
          <th>RAD</th>
          <th>TAX</th>
          <th>PTRATIO</th>
          <th>B</th>
          <th>LSTAT</th>
          <th>MEDV</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.00632</td>
          <td>18.0</td>
          <td>2.31</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.575</td>
          <td>65.2</td>
          <td>4.0900</td>
          <td>1.0</td>
          <td>296.0</td>
          <td>15.3</td>
          <td>396.90</td>
          <td>4.98</td>
          <td>24.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.02731</td>
          <td>0.0</td>
          <td>7.07</td>
          <td>0.0</td>
          <td>0.469</td>
          <td>6.421</td>
          <td>78.9</td>
          <td>4.9671</td>
          <td>2.0</td>
          <td>242.0</td>
          <td>17.8</td>
          <td>396.90</td>
          <td>9.14</td>
          <td>21.6</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.02729</td>
          <td>0.0</td>
          <td>7.07</td>
          <td>0.0</td>
          <td>0.469</td>
          <td>7.185</td>
          <td>61.1</td>
          <td>4.9671</td>
          <td>2.0</td>
          <td>242.0</td>
          <td>17.8</td>
          <td>392.83</td>
          <td>4.03</td>
          <td>34.7</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.03237</td>
          <td>0.0</td>
          <td>2.18</td>
          <td>0.0</td>
          <td>0.458</td>
          <td>6.998</td>
          <td>45.8</td>
          <td>6.0622</td>
          <td>3.0</td>
          <td>222.0</td>
          <td>18.7</td>
          <td>394.63</td>
          <td>2.94</td>
          <td>33.4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.06905</td>
          <td>0.0</td>
          <td>2.18</td>
          <td>0.0</td>
          <td>0.458</td>
          <td>7.147</td>
          <td>54.2</td>
          <td>6.0622</td>
          <td>3.0</td>
          <td>222.0</td>
          <td>18.7</td>
          <td>396.90</td>
          <td>5.33</td>
          <td>36.2</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.02985</td>
          <td>0.0</td>
          <td>2.18</td>
          <td>0.0</td>
          <td>0.458</td>
          <td>6.430</td>
          <td>58.7</td>
          <td>6.0622</td>
          <td>3.0</td>
          <td>222.0</td>
          <td>18.7</td>
          <td>394.12</td>
          <td>5.21</td>
          <td>28.7</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.08829</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.012</td>
          <td>66.6</td>
          <td>5.5605</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>395.60</td>
          <td>12.43</td>
          <td>22.9</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.14455</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.172</td>
          <td>96.1</td>
          <td>5.9505</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>396.90</td>
          <td>19.15</td>
          <td>27.1</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.21124</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>5.631</td>
          <td>100.0</td>
          <td>6.0821</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>386.63</td>
          <td>29.93</td>
          <td>16.5</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.17004</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.004</td>
          <td>85.9</td>
          <td>6.5921</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>386.71</td>
          <td>17.10</td>
          <td>18.9</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.22489</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.377</td>
          <td>94.3</td>
          <td>6.3467</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>392.52</td>
          <td>20.45</td>
          <td>15.0</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.11747</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.009</td>
          <td>82.9</td>
          <td>6.2267</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>396.90</td>
          <td>13.27</td>
          <td>18.9</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.09378</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>5.889</td>
          <td>39.0</td>
          <td>5.4509</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>390.50</td>
          <td>15.71</td>
          <td>21.7</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.62976</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.949</td>
          <td>61.8</td>
          <td>4.7075</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>8.26</td>
          <td>20.4</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.63796</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.096</td>
          <td>84.5</td>
          <td>4.4619</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>380.02</td>
          <td>10.26</td>
          <td>18.2</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.62739</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.834</td>
          <td>56.5</td>
          <td>4.4986</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>395.62</td>
          <td>8.47</td>
          <td>19.9</td>
        </tr>
        <tr>
          <th>16</th>
          <td>1.05393</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.935</td>
          <td>29.3</td>
          <td>4.4986</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>386.85</td>
          <td>6.58</td>
          <td>23.1</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.78420</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.990</td>
          <td>81.7</td>
          <td>4.2579</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>386.75</td>
          <td>14.67</td>
          <td>17.5</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.80271</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.456</td>
          <td>36.6</td>
          <td>3.7965</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>288.99</td>
          <td>11.69</td>
          <td>20.2</td>
        </tr>
        <tr>
          <th>19</th>
          <td>0.72580</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.727</td>
          <td>69.5</td>
          <td>3.7965</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>390.95</td>
          <td>11.28</td>
          <td>18.2</td>
        </tr>
        <tr>
          <th>20</th>
          <td>1.25179</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.570</td>
          <td>98.1</td>
          <td>3.7979</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>376.57</td>
          <td>21.02</td>
          <td>13.6</td>
        </tr>
        <tr>
          <th>21</th>
          <td>0.85204</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.965</td>
          <td>89.2</td>
          <td>4.0123</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>392.53</td>
          <td>13.83</td>
          <td>19.6</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1.23247</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.142</td>
          <td>91.7</td>
          <td>3.9769</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>18.72</td>
          <td>15.2</td>
        </tr>
        <tr>
          <th>23</th>
          <td>0.98843</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.813</td>
          <td>100.0</td>
          <td>4.0952</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>394.54</td>
          <td>19.88</td>
          <td>14.5</td>
        </tr>
        <tr>
          <th>24</th>
          <td>0.75026</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.924</td>
          <td>94.1</td>
          <td>4.3996</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>394.33</td>
          <td>16.30</td>
          <td>15.6</td>
        </tr>
        <tr>
          <th>25</th>
          <td>0.84054</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.599</td>
          <td>85.7</td>
          <td>4.4546</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>303.42</td>
          <td>16.51</td>
          <td>13.9</td>
        </tr>
        <tr>
          <th>26</th>
          <td>0.67191</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.813</td>
          <td>90.3</td>
          <td>4.6820</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>376.88</td>
          <td>14.81</td>
          <td>16.6</td>
        </tr>
        <tr>
          <th>27</th>
          <td>0.95577</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.047</td>
          <td>88.8</td>
          <td>4.4534</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>306.38</td>
          <td>17.28</td>
          <td>14.8</td>
        </tr>
        <tr>
          <th>28</th>
          <td>0.77299</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.495</td>
          <td>94.4</td>
          <td>4.4547</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>387.94</td>
          <td>12.80</td>
          <td>18.4</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1.00245</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.674</td>
          <td>87.3</td>
          <td>4.2390</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>380.23</td>
          <td>11.98</td>
          <td>21.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>476</th>
          <td>4.87141</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>6.484</td>
          <td>93.6</td>
          <td>2.3053</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>396.21</td>
          <td>18.68</td>
          <td>16.7</td>
        </tr>
        <tr>
          <th>477</th>
          <td>15.02340</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>5.304</td>
          <td>97.3</td>
          <td>2.1007</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>349.48</td>
          <td>24.91</td>
          <td>12.0</td>
        </tr>
        <tr>
          <th>478</th>
          <td>10.23300</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>6.185</td>
          <td>96.7</td>
          <td>2.1705</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>379.70</td>
          <td>18.03</td>
          <td>14.6</td>
        </tr>
        <tr>
          <th>479</th>
          <td>14.33370</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>6.229</td>
          <td>88.0</td>
          <td>1.9512</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>383.32</td>
          <td>13.11</td>
          <td>21.4</td>
        </tr>
        <tr>
          <th>480</th>
          <td>5.82401</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>6.242</td>
          <td>64.7</td>
          <td>3.4242</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>396.90</td>
          <td>10.74</td>
          <td>23.0</td>
        </tr>
        <tr>
          <th>481</th>
          <td>5.70818</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>6.750</td>
          <td>74.9</td>
          <td>3.3317</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>393.07</td>
          <td>7.74</td>
          <td>23.7</td>
        </tr>
        <tr>
          <th>482</th>
          <td>5.73116</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>7.061</td>
          <td>77.0</td>
          <td>3.4106</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>395.28</td>
          <td>7.01</td>
          <td>25.0</td>
        </tr>
        <tr>
          <th>483</th>
          <td>2.81838</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>5.762</td>
          <td>40.3</td>
          <td>4.0983</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>392.92</td>
          <td>10.42</td>
          <td>21.8</td>
        </tr>
        <tr>
          <th>484</th>
          <td>2.37857</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>5.871</td>
          <td>41.9</td>
          <td>3.7240</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>370.73</td>
          <td>13.34</td>
          <td>20.6</td>
        </tr>
        <tr>
          <th>485</th>
          <td>3.67367</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>6.312</td>
          <td>51.9</td>
          <td>3.9917</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>388.62</td>
          <td>10.58</td>
          <td>21.2</td>
        </tr>
        <tr>
          <th>486</th>
          <td>5.69175</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>6.114</td>
          <td>79.8</td>
          <td>3.5459</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>392.68</td>
          <td>14.98</td>
          <td>19.1</td>
        </tr>
        <tr>
          <th>487</th>
          <td>4.83567</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>5.905</td>
          <td>53.2</td>
          <td>3.1523</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>388.22</td>
          <td>11.45</td>
          <td>20.6</td>
        </tr>
        <tr>
          <th>488</th>
          <td>0.15086</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.454</td>
          <td>92.7</td>
          <td>1.8209</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>395.09</td>
          <td>18.06</td>
          <td>15.2</td>
        </tr>
        <tr>
          <th>489</th>
          <td>0.18337</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.414</td>
          <td>98.3</td>
          <td>1.7554</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>344.05</td>
          <td>23.97</td>
          <td>7.0</td>
        </tr>
        <tr>
          <th>490</th>
          <td>0.20746</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.093</td>
          <td>98.0</td>
          <td>1.8226</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>318.43</td>
          <td>29.68</td>
          <td>8.1</td>
        </tr>
        <tr>
          <th>491</th>
          <td>0.10574</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.983</td>
          <td>98.8</td>
          <td>1.8681</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>390.11</td>
          <td>18.07</td>
          <td>13.6</td>
        </tr>
        <tr>
          <th>492</th>
          <td>0.11132</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.983</td>
          <td>83.5</td>
          <td>2.1099</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>396.90</td>
          <td>13.35</td>
          <td>20.1</td>
        </tr>
        <tr>
          <th>493</th>
          <td>0.17331</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.707</td>
          <td>54.0</td>
          <td>2.3817</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>12.01</td>
          <td>21.8</td>
        </tr>
        <tr>
          <th>494</th>
          <td>0.27957</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.926</td>
          <td>42.6</td>
          <td>2.3817</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>13.59</td>
          <td>24.5</td>
        </tr>
        <tr>
          <th>495</th>
          <td>0.17899</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.670</td>
          <td>28.8</td>
          <td>2.7986</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>393.29</td>
          <td>17.60</td>
          <td>23.1</td>
        </tr>
        <tr>
          <th>496</th>
          <td>0.28960</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.390</td>
          <td>72.9</td>
          <td>2.7986</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>21.14</td>
          <td>19.7</td>
        </tr>
        <tr>
          <th>497</th>
          <td>0.26838</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.794</td>
          <td>70.6</td>
          <td>2.8927</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>14.10</td>
          <td>18.3</td>
        </tr>
        <tr>
          <th>498</th>
          <td>0.23912</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>6.019</td>
          <td>65.3</td>
          <td>2.4091</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>12.92</td>
          <td>21.2</td>
        </tr>
        <tr>
          <th>499</th>
          <td>0.17783</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.569</td>
          <td>73.5</td>
          <td>2.3999</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>395.77</td>
          <td>15.10</td>
          <td>17.5</td>
        </tr>
        <tr>
          <th>500</th>
          <td>0.22438</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>6.027</td>
          <td>79.7</td>
          <td>2.4982</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>14.33</td>
          <td>16.8</td>
        </tr>
        <tr>
          <th>501</th>
          <td>0.06263</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.593</td>
          <td>69.1</td>
          <td>2.4786</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>391.99</td>
          <td>9.67</td>
          <td>22.4</td>
        </tr>
        <tr>
          <th>502</th>
          <td>0.04527</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.120</td>
          <td>76.7</td>
          <td>2.2875</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>9.08</td>
          <td>20.6</td>
        </tr>
        <tr>
          <th>503</th>
          <td>0.06076</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.976</td>
          <td>91.0</td>
          <td>2.1675</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>5.64</td>
          <td>23.9</td>
        </tr>
        <tr>
          <th>504</th>
          <td>0.10959</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.794</td>
          <td>89.3</td>
          <td>2.3889</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>393.45</td>
          <td>6.48</td>
          <td>22.0</td>
        </tr>
        <tr>
          <th>505</th>
          <td>0.04741</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.030</td>
          <td>80.8</td>
          <td>2.5050</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>7.88</td>
          <td>11.9</td>
        </tr>
      </tbody>
    </table>
    <p>506 rows x 14 columns</p>
    </div>
  
或者采用另外一种方式:

.. code:: python
  
  >>> from sklearn.datasets import load_boston
  >>> import pandas as pd
  >>> boston = load_boston()
  >>> print(len(boston.data))
  506
  >>> print(len(boston.target))
  506
  >>> print(boston.data[0])
  [6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01
 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00]
  >>> print(set(boston.target))
  {5.0, 6.3, 7.2, 8.8, 7.4, 10.2, 11.8, 12.7, 13.6, 14.5, 15.2, 15.0, 16.5, 17.5, 19.6, 18.9, 18.2, 20.4, 21.6, 22.9, 21.7, 26.6, 26.5, 27.5, 24.0, 23.1, 27.1, 28.7, 24.7, 30.8, 33.4, 34.7, 34.9, 36.2, 35.4, 31.6, 33.0, 38.7, 43.8, 41.3, 37.2, 39.8, 42.3, 48.5, 44.8, 50.0, 46.7, 48.3, 44.0, 48.8, 46.0, 10.5, 11.5, 11.0, 12.5, 12.0, 13.5, 13.0, 14.0, 16.6, 16.0, 16.1, 16.4, 17.4, 17.1, 17.0, 17.6, 17.9, 18.4, 18.6, 18.5, 18.0, 18.1, 19.9, 19.4, 19.5, 19.1, 19.0, 20.1, 20.0, 20.5, 20.9, 20.6, 21.0, 21.4, 21.5, 21.9, 21.1, 22.0, 22.5, 22.6, 22.4, 22.1, 23.4, 23.5, 23.9, 23.6, 23.0, 24.1, 24.6, 24.4, 24.5, 25.0, 25.1, 26.4, 27.0, 27.9, 28.0, 28.4, 28.1, 28.5, 28.6, 29.4, 29.9, 29.6, 29.1, 29.0, 30.5, 30.1, 31.1, 31.5, 31.0, 32.5, 32.0, 32.9, 32.4, 32.2, 33.2, 33.3, 33.8, 33.1, 32.7, 34.6, 8.4, 35.2, 35.1, 10.4, 10.9, 7.0, 36.4, 36.0, 36.5, 36.1, 11.9, 37.9, 37.0, 37.6, 37.3, 13.9, 13.4, 14.4, 14.9, 15.4, 8.5, 41.7, 42.8, 43.1, 43.5, 45.4, 9.5, 8.3, 8.7, 9.7, 10.8, 11.3, 11.7, 12.3, 12.8, 13.2, 13.3, 13.8, 14.8, 14.3, 14.2, 15.7, 15.3, 16.2, 16.8, 16.3, 16.7, 17.3, 17.8, 17.2, 17.7, 18.3, 18.7, 18.8, 19.2, 19.3, 19.7, 19.8, 20.2, 20.8, 20.3, 20.7, 21.2, 21.8, 22.2, 22.8, 22.7, 22.3, 23.3, 23.8, 23.2, 23.7, 24.8, 24.2, 24.3, 25.3, 25.2, 26.7, 26.2, 7.5, 28.2, 29.8, 30.3, 30.7, 5.6, 31.7, 31.2, 8.1, 9.6, 12.1, 12.6, 13.1, 14.6, 14.1, 15.6, 15.1}
  >>> pd.DataFrame(data=boston.data, columns=boston.feature_names)
  
.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>CRIM</th>
          <th>ZN</th>
          <th>INDUS</th>
          <th>CHAS</th>
          <th>NOX</th>
          <th>RM</th>
          <th>AGE</th>
          <th>DIS</th>
          <th>RAD</th>
          <th>TAX</th>
          <th>PTRATIO</th>
          <th>B</th>
          <th>LSTAT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.00632</td>
          <td>18.0</td>
          <td>2.31</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.575</td>
          <td>65.2</td>
          <td>4.0900</td>
          <td>1.0</td>
          <td>296.0</td>
          <td>15.3</td>
          <td>396.90</td>
          <td>4.98</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.02731</td>
          <td>0.0</td>
          <td>7.07</td>
          <td>0.0</td>
          <td>0.469</td>
          <td>6.421</td>
          <td>78.9</td>
          <td>4.9671</td>
          <td>2.0</td>
          <td>242.0</td>
          <td>17.8</td>
          <td>396.90</td>
          <td>9.14</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.02729</td>
          <td>0.0</td>
          <td>7.07</td>
          <td>0.0</td>
          <td>0.469</td>
          <td>7.185</td>
          <td>61.1</td>
          <td>4.9671</td>
          <td>2.0</td>
          <td>242.0</td>
          <td>17.8</td>
          <td>392.83</td>
          <td>4.03</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.03237</td>
          <td>0.0</td>
          <td>2.18</td>
          <td>0.0</td>
          <td>0.458</td>
          <td>6.998</td>
          <td>45.8</td>
          <td>6.0622</td>
          <td>3.0</td>
          <td>222.0</td>
          <td>18.7</td>
          <td>394.63</td>
          <td>2.94</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.06905</td>
          <td>0.0</td>
          <td>2.18</td>
          <td>0.0</td>
          <td>0.458</td>
          <td>7.147</td>
          <td>54.2</td>
          <td>6.0622</td>
          <td>3.0</td>
          <td>222.0</td>
          <td>18.7</td>
          <td>396.90</td>
          <td>5.33</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.02985</td>
          <td>0.0</td>
          <td>2.18</td>
          <td>0.0</td>
          <td>0.458</td>
          <td>6.430</td>
          <td>58.7</td>
          <td>6.0622</td>
          <td>3.0</td>
          <td>222.0</td>
          <td>18.7</td>
          <td>394.12</td>
          <td>5.21</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.08829</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.012</td>
          <td>66.6</td>
          <td>5.5605</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>395.60</td>
          <td>12.43</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.14455</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.172</td>
          <td>96.1</td>
          <td>5.9505</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>396.90</td>
          <td>19.15</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.21124</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>5.631</td>
          <td>100.0</td>
          <td>6.0821</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>386.63</td>
          <td>29.93</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.17004</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.004</td>
          <td>85.9</td>
          <td>6.5921</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>386.71</td>
          <td>17.10</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.22489</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.377</td>
          <td>94.3</td>
          <td>6.3467</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>392.52</td>
          <td>20.45</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.11747</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>6.009</td>
          <td>82.9</td>
          <td>6.2267</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>396.90</td>
          <td>13.27</td>
        </tr>
        <tr>
          <th>12</th>
          <td>0.09378</td>
          <td>12.5</td>
          <td>7.87</td>
          <td>0.0</td>
          <td>0.524</td>
          <td>5.889</td>
          <td>39.0</td>
          <td>5.4509</td>
          <td>5.0</td>
          <td>311.0</td>
          <td>15.2</td>
          <td>390.50</td>
          <td>15.71</td>
        </tr>
        <tr>
          <th>13</th>
          <td>0.62976</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.949</td>
          <td>61.8</td>
          <td>4.7075</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>8.26</td>
        </tr>
        <tr>
          <th>14</th>
          <td>0.63796</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.096</td>
          <td>84.5</td>
          <td>4.4619</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>380.02</td>
          <td>10.26</td>
        </tr>
        <tr>
          <th>15</th>
          <td>0.62739</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.834</td>
          <td>56.5</td>
          <td>4.4986</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>395.62</td>
          <td>8.47</td>
        </tr>
        <tr>
          <th>16</th>
          <td>1.05393</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.935</td>
          <td>29.3</td>
          <td>4.4986</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>386.85</td>
          <td>6.58</td>
        </tr>
        <tr>
          <th>17</th>
          <td>0.78420</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.990</td>
          <td>81.7</td>
          <td>4.2579</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>386.75</td>
          <td>14.67</td>
        </tr>
        <tr>
          <th>18</th>
          <td>0.80271</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.456</td>
          <td>36.6</td>
          <td>3.7965</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>288.99</td>
          <td>11.69</td>
        </tr>
        <tr>
          <th>19</th>
          <td>0.72580</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.727</td>
          <td>69.5</td>
          <td>3.7965</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>390.95</td>
          <td>11.28</td>
        </tr>
        <tr>
          <th>20</th>
          <td>1.25179</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.570</td>
          <td>98.1</td>
          <td>3.7979</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>376.57</td>
          <td>21.02</td>
        </tr>
        <tr>
          <th>21</th>
          <td>0.85204</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.965</td>
          <td>89.2</td>
          <td>4.0123</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>392.53</td>
          <td>13.83</td>
        </tr>
        <tr>
          <th>22</th>
          <td>1.23247</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.142</td>
          <td>91.7</td>
          <td>3.9769</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>18.72</td>
        </tr>
        <tr>
          <th>23</th>
          <td>0.98843</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.813</td>
          <td>100.0</td>
          <td>4.0952</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>394.54</td>
          <td>19.88</td>
        </tr>
        <tr>
          <th>24</th>
          <td>0.75026</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.924</td>
          <td>94.1</td>
          <td>4.3996</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>394.33</td>
          <td>16.30</td>
        </tr>
        <tr>
          <th>25</th>
          <td>0.84054</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.599</td>
          <td>85.7</td>
          <td>4.4546</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>303.42</td>
          <td>16.51</td>
        </tr>
        <tr>
          <th>26</th>
          <td>0.67191</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>5.813</td>
          <td>90.3</td>
          <td>4.6820</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>376.88</td>
          <td>14.81</td>
        </tr>
        <tr>
          <th>27</th>
          <td>0.95577</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.047</td>
          <td>88.8</td>
          <td>4.4534</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>306.38</td>
          <td>17.28</td>
        </tr>
        <tr>
          <th>28</th>
          <td>0.77299</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.495</td>
          <td>94.4</td>
          <td>4.4547</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>387.94</td>
          <td>12.80</td>
        </tr>
        <tr>
          <th>29</th>
          <td>1.00245</td>
          <td>0.0</td>
          <td>8.14</td>
          <td>0.0</td>
          <td>0.538</td>
          <td>6.674</td>
          <td>87.3</td>
          <td>4.2390</td>
          <td>4.0</td>
          <td>307.0</td>
          <td>21.0</td>
          <td>380.23</td>
          <td>11.98</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>476</th>
          <td>4.87141</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>6.484</td>
          <td>93.6</td>
          <td>2.3053</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>396.21</td>
          <td>18.68</td>
        </tr>
        <tr>
          <th>477</th>
          <td>15.02340</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>5.304</td>
          <td>97.3</td>
          <td>2.1007</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>349.48</td>
          <td>24.91</td>
        </tr>
        <tr>
          <th>478</th>
          <td>10.23300</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>6.185</td>
          <td>96.7</td>
          <td>2.1705</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>379.70</td>
          <td>18.03</td>
        </tr>
        <tr>
          <th>479</th>
          <td>14.33370</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.614</td>
          <td>6.229</td>
          <td>88.0</td>
          <td>1.9512</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>383.32</td>
          <td>13.11</td>
        </tr>
        <tr>
          <th>480</th>
          <td>5.82401</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>6.242</td>
          <td>64.7</td>
          <td>3.4242</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>396.90</td>
          <td>10.74</td>
        </tr>
        <tr>
          <th>481</th>
          <td>5.70818</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>6.750</td>
          <td>74.9</td>
          <td>3.3317</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>393.07</td>
          <td>7.74</td>
        </tr>
        <tr>
          <th>482</th>
          <td>5.73116</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>7.061</td>
          <td>77.0</td>
          <td>3.4106</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>395.28</td>
          <td>7.01</td>
        </tr>
        <tr>
          <th>483</th>
          <td>2.81838</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.532</td>
          <td>5.762</td>
          <td>40.3</td>
          <td>4.0983</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>392.92</td>
          <td>10.42</td>
        </tr>
        <tr>
          <th>484</th>
          <td>2.37857</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>5.871</td>
          <td>41.9</td>
          <td>3.7240</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>370.73</td>
          <td>13.34</td>
        </tr>
        <tr>
          <th>485</th>
          <td>3.67367</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>6.312</td>
          <td>51.9</td>
          <td>3.9917</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>388.62</td>
          <td>10.58</td>
        </tr>
        <tr>
          <th>486</th>
          <td>5.69175</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>6.114</td>
          <td>79.8</td>
          <td>3.5459</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>392.68</td>
          <td>14.98</td>
        </tr>
        <tr>
          <th>487</th>
          <td>4.83567</td>
          <td>0.0</td>
          <td>18.10</td>
          <td>0.0</td>
          <td>0.583</td>
          <td>5.905</td>
          <td>53.2</td>
          <td>3.1523</td>
          <td>24.0</td>
          <td>666.0</td>
          <td>20.2</td>
          <td>388.22</td>
          <td>11.45</td>
        </tr>
        <tr>
          <th>488</th>
          <td>0.15086</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.454</td>
          <td>92.7</td>
          <td>1.8209</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>395.09</td>
          <td>18.06</td>
        </tr>
        <tr>
          <th>489</th>
          <td>0.18337</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.414</td>
          <td>98.3</td>
          <td>1.7554</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>344.05</td>
          <td>23.97</td>
        </tr>
        <tr>
          <th>490</th>
          <td>0.20746</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.093</td>
          <td>98.0</td>
          <td>1.8226</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>318.43</td>
          <td>29.68</td>
        </tr>
        <tr>
          <th>491</th>
          <td>0.10574</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.983</td>
          <td>98.8</td>
          <td>1.8681</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>390.11</td>
          <td>18.07</td>
        </tr>
        <tr>
          <th>492</th>
          <td>0.11132</td>
          <td>0.0</td>
          <td>27.74</td>
          <td>0.0</td>
          <td>0.609</td>
          <td>5.983</td>
          <td>83.5</td>
          <td>2.1099</td>
          <td>4.0</td>
          <td>711.0</td>
          <td>20.1</td>
          <td>396.90</td>
          <td>13.35</td>
        </tr>
        <tr>
          <th>493</th>
          <td>0.17331</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.707</td>
          <td>54.0</td>
          <td>2.3817</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>12.01</td>
        </tr>
        <tr>
          <th>494</th>
          <td>0.27957</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.926</td>
          <td>42.6</td>
          <td>2.3817</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>13.59</td>
        </tr>
        <tr>
          <th>495</th>
          <td>0.17899</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.670</td>
          <td>28.8</td>
          <td>2.7986</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>393.29</td>
          <td>17.60</td>
        </tr>
        <tr>
          <th>496</th>
          <td>0.28960</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.390</td>
          <td>72.9</td>
          <td>2.7986</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>21.14</td>
        </tr>
        <tr>
          <th>497</th>
          <td>0.26838</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.794</td>
          <td>70.6</td>
          <td>2.8927</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>14.10</td>
        </tr>
        <tr>
          <th>498</th>
          <td>0.23912</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>6.019</td>
          <td>65.3</td>
          <td>2.4091</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>12.92</td>
        </tr>
        <tr>
          <th>499</th>
          <td>0.17783</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>5.569</td>
          <td>73.5</td>
          <td>2.3999</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>395.77</td>
          <td>15.10</td>
        </tr>
        <tr>
          <th>500</th>
          <td>0.22438</td>
          <td>0.0</td>
          <td>9.69</td>
          <td>0.0</td>
          <td>0.585</td>
          <td>6.027</td>
          <td>79.7</td>
          <td>2.4982</td>
          <td>6.0</td>
          <td>391.0</td>
          <td>19.2</td>
          <td>396.90</td>
          <td>14.33</td>
        </tr>
        <tr>
          <th>501</th>
          <td>0.06263</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.593</td>
          <td>69.1</td>
          <td>2.4786</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>391.99</td>
          <td>9.67</td>
        </tr>
        <tr>
          <th>502</th>
          <td>0.04527</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.120</td>
          <td>76.7</td>
          <td>2.2875</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>9.08</td>
        </tr>
        <tr>
          <th>503</th>
          <td>0.06076</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.976</td>
          <td>91.0</td>
          <td>2.1675</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>5.64</td>
        </tr>
        <tr>
          <th>504</th>
          <td>0.10959</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.794</td>
          <td>89.3</td>
          <td>2.3889</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>393.45</td>
          <td>6.48</td>
        </tr>
        <tr>
          <th>505</th>
          <td>0.04741</td>
          <td>0.0</td>
          <td>11.93</td>
          <td>0.0</td>
          <td>0.573</td>
          <td>6.030</td>
          <td>80.8</td>
          <td>2.5050</td>
          <td>1.0</td>
          <td>273.0</td>
          <td>21.0</td>
          <td>396.90</td>
          <td>7.88</td>
        </tr>
      </tbody>
    </table>
    <p>506 rows x 13 columns</p>
    </div>

.. code:: python
  
  #颜色
  >>> cnames = {'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7', 'aqua': '#00FFFF', 'aquamarine': '#7FFFD4', 'azure': '#F0FFFF', 'beige': '#F5F5DC', 'bisque': '#FFE4C4', 'black': '#000000', 'blanchedalmond': '#FFEBCD', 'blue': '#0000FF', 'blueviolet': '#8A2BE2', 'brown': '#A52A2A', 'burlywood': '#DEB887', 'cadetblue': '#5F9EA0', 'chartreuse': '#7FFF00', 'chocolate': '#D2691E', 'coral': '#FF7F50', 'cornflowerblue': '#6495ED', 'cornsilk': '#FFF8DC', 'crimson': '#DC143C', 'cyan': '#00FFFF', 'darkblue': '#00008B', 'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B', 'darkgray': '#A9A9A9', 'darkgreen': '#006400', 'darkkhaki': '#BDB76B', 'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F', 'darkorange': '#FF8C00', 'darkorchid': '#9932CC', 'darkred': '#8B0000', 'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F', 'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F', 'darkturquoise': '#00CED1', 'darkviolet': '#9400D3', 'deeppink': '#FF1493', 'deepskyblue': '#00BFFF', 'dimgray': '#696969', 'dodgerblue': '#1E90FF', 'firebrick': '#B22222', 'floralwhite': '#FFFAF0', 'forestgreen': '#228B22', 'fuchsia': '#FF00FF', 'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF', 'gold': '#FFD700', 'goldenrod': '#DAA520', 'gray': '#808080', 'green': '#008000', 'greenyellow': '#ADFF2F', 'honeydew': '#F0FFF0', 'hotpink': '#FF69B4', 'indianred': '#CD5C5C', 'indigo': '#4B0082', 'ivory': '#FFFFF0', 'khaki': '#F0E68C', 'lavender': '#E6E6FA', 'lavenderblush': '#FFF0F5', 'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD', 'lightblue': '#ADD8E6', 'lightcoral': '#F08080', 'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2', 'lightgreen': '#90EE90', 'lightgray': '#D3D3D3', 'lightpink': '#FFB6C1', 'lightsalmon': '#FFA07A', 'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA', 'lightslategray': '#778899', 'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0', 'lime': '#00FF00', 'limegreen': '#32CD32', 'linen': '#FAF0E6', 'magenta': '#FF00FF', 'maroon': '#800000', 'mediumaquamarine': '#66CDAA', 'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3', 'mediumpurple': '#9370DB', 'mediumseagreen': '#3CB371', 'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A', 'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585', 'midnightblue': '#191970', 'mintcream': '#F5FFFA', 'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5', 'navajowhite': '#FFDEAD', 'navy': '#000080', 'oldlace': '#FDF5E6', 'olive': '#808000', 'olivedrab': '#6B8E23', 'orange': '#FFA500', 'orangered': '#FF4500', 'orchid': '#DA70D6', 'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98', 'paleturquoise': '#AFEEEE', 'palevioletred': '#DB7093', 'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9', 'peru': '#CD853F', 'pink': '#FFC0CB', 'plum': '#DDA0DD', 'powderblue': '#B0E0E6', 'purple': '#800080', 'red': '#FF0000', 'rosybrown': '#BC8F8F', 'royalblue': '#4169E1', 'saddlebrown': '#8B4513', 'salmon': '#FA8072', 'sandybrown': '#FAA460', 'seagreen': '#2E8B57', 'seashell': '#FFF5EE', 'sienna': '#A0522D', 'silver': '#C0C0C0', 'skyblue': '#87CEEB', 'slateblue': '#6A5ACD', 'slategray': '#708090', 'snow': '#FFFAFA', 'springgreen': '#00FF7F', 'steelblue': '#4682B4', 'tan': '#D2B48C', 'teal': '#008080', 'thistle': '#D8BFD8', 'tomato': '#FF6347', 'turquoise': '#40E0D0', 'violet': '#EE82EE', 'wheat': '#F5DEB3', 'white': '#FFFFFF', 'whitesmoke': '#F5F5F5', 'yellow': '#FFFF00', 'yellowgreen': '#9ACD32'}
  >>> colorname = list(cnames.keys())
  >>> print(colorname)
  ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
  >>> X = boston.data  
  >>> y = boston.target  
  >>> features = boston.feature_names
  >>> def Boston():
  ...     for i, colorn in enumerate(colorname[13:26]):
  ...         if i<12:
  ...           plt.figure(43)
  ...           plt.subplot(5,3,i+1)
  ...           plt.plot(X[:,i], y, color=str(colorn))
  ...         if i==12:
  ...           plt.subplot(5,1,5)
  ...           plt.plot(X[:,i],y,color=str(colorn))
  ...      plt.savefig('Boston_Housing_Data.png', dpi=700)
  ...      plt.show()      
  >>> Boston()

.. raw:: html
  
  <img src="data:image/png;base64,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
  ">

摘取其中LSTAT与Boston House Price的关系图：

.. code:: python
  
  >>> X = boston.data  
  >>> y = boston.target  
  >>> features = boston.feature_names[12]
  >>> targets = 'Boston Housing Price versus %s' %(features)  #3类鸢尾花的名称，跟y中的3个数字对应
  ... plt.figure(figsize=(8, 5))
  ... plt.plot(X[:,12], y, 'bx', label=targets)
  ... plt.title('Boston Housing Data')
  ... plt.legend()
  ... plt.savefig('Boston Housing Data.png', dpi=500)
  ... plt.show()

.. raw:: html
  
  <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAeEAAAE8CAYAAAD36gn/AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAIABJREFUeJztvX+cHVWZ5/85SXe6k8ZuAgmgCEQEiYMMk590CA1pOo2GNL74NSEDdsAh22z7Y9mNTEzUjgoMkCydfQE7ZkYXUPFHJusCjqsYEm4HggRNGMBVwHFY1A0QvwF/jBFxIJ7vH+c+qXNPV9Wturfq3rp1P+/Xq1/3Vt2qU6fq9j2f8zznOc9RWmsQQgghpPZMqHcFCCGEkGaFIkwIIYTUCYowIYQQUicowoQQQkidoAgTQgghdYIiTAghhNQJijBpWpRSn1FKHVBKvayU+plS6qp618kPpdQOpdT7rG2tlGpP4Tr7ki6zWO7PlFKvKqV+qZS6XSk1KY3rENKIUIRJs/PftdZvBbAQwH9VSk2rpBCl1IysinhUtNbHpFj8QgB/AeAMADeWO1gptUgptSjF+hCSCSjChADQWr8I4EUA76ywiBkArkqqPnlEa/0ygNUAVkQ4fFHxj5BcQxEmBIBS6s8AvB3AT4vbf6OU+oVS6idKqSXFfROUUp8vuq9/oZS6qLh/J4B7AZyplNqnlPqCdfyoUupFpdTTSql5xf1fVEp9Sin1mFLqV0qp6yqs85RiWS8rpR5VSp1U3L9DrMiihf6z4vujip/9Uin1lFJqplOedrZ3KKX+U7Huryql/qq4/zCl1Lbide8plvXnEav9FICjlVKHK6WOVkqNFcv5gVLqBKXU9KJb/DoA1xWf5weL1x13fCXPjZAsQREmzc5HlFK/BPBjAHdorX+llFoM4EoApwO4CMDdSqmjYdypAwBOAPBeAP0AoLXuAXAxgMe01sdorf9Dsey/Lp7zTgD/BcD/VEq1FT+7BsBgsZyRCPX8WlGQ7HHbTwCYCOBYAP8DwNfKlPEBAK9orY8GcDOiWZpXA1gM4D8CWFPcNwjgFQDHAzgbwKVa6x9GKAsADhRfDwNwCYBCcTjgXgAf1lrvL7rFbwVwa/F53l08Z9zxEa9JSGahCJNm578XRekkACuVUmcDWALgK1rrX2utnwHwfQA9AJ4H8CcA/xXAuwD85zJlLwHwBa3161rrAoDfAjit+NmXtNbPA9gDoDNCPS8vCpI9brsEwN9prf+ktf4igHf5jGkr6/0PAHQrpT4N4EWt9d9HuO4dWuv9xXp2Ffe9DmASTAegpfgalY7i6wEAfw/gGaXU38GI/FFlzo17PCGZhyJMCICiIO6CCRwCANs1q80h+rcA/gzATgCXA9gWpWi3nOL754vXrXYFFfd8d/vYQx9o/SiABQD2AtiolPpshPKf9yn3OZjOxPMAvqi1/kmM+v45gJe01r8B8A8ALgPwTwA2RDg37vGEZB6KMCEAlFLHAOgG8BMADwC4ojhuORNGmB9VSvUBuAvA/QDWApivlBJL8xUAb1dKTVRKTVVKTSyWc7VSqk0pdQ6AwwH8qHh8EsuXPQBguDj2PAjgX7TWrwL4NwDHFY/5iHWP6wAs11rfCWNVdke4hl89/wOAtVrrY7XWn4xa2eIzXg/jOgdMh+AuGAv9IufwV2Dc/rCs+7DjCWlIKMKk2ZEx4ScB/COAb2mttwO4B8APYayuv9Za/xLAwwB+BxNFvRPAarFktdY/ArC9+NmPYNy1dwH4PwD+L4DbAfyl1vqPCdb9Jhh384sw7tnLi/tvAzCilPougMet4+8EMKCU+v8AfLx4fiV8B8CXlVIvKaX+j1LqQxHO+R7MM34YwPXFff8N5jl/H8ALMC5+4asATizW9a4IxxPSkCiuJ0wIiYNSag+Ai7XWv1BKzQCwqxgsRQiJSUu9K0AIaTi+DuCRYuarAwCijC0TQnygJUwIIYTUCY4JE0IIIXWCIkwIIYTUCYowIYQQUidSD8yaNm2anjFjRtqXIYQQQjLDE0888YrWenq541IX4RkzZmDPnj1pX4YQQgjJDEqpn0c5ju5oQgghpE5QhAkhhJA6QREmhBBC6gQzZhFCQnnjjTewd+9evP766/WuCiGZo729HW9/+9vR2tpa0fkUYUJIKHv37sVb3vIWzJgxA96iUYQQrTVeffVV7N27F+94xzsqKoPuaEJIKK+//jqOPPJICjAhDkopHHnkkVV5iSjChJCyUIAJ8afa30bDiPCGDcDYmPcKlG5v2DD++GuuMX/y+bvfDSxYYPbJeUcdZfbbbNwInH9+tOuef763X/CrT1L3X+l1ws6vtmxC0uSqq67CrFmzsGjRIlx22WU4ePBgrPOfeuopPPXUU4nV5dFHHwUA3HjjjfjiF79YdZn79u3DLbfcUnU5gPesenp6cMEFF+DAgQM1uW6WsL8j4eWXX8Z73/tenHnmmVizZg0A4FOf+hS6u7sxdepULFq0CLt27QJg/l9s13LQcUkRKsJKqRal1C+UUjuKf6cppT6rlNqtlPq7RGtShnnzgGXLgJYW87pxY+n2vHnjj9+8GfjqV4ELLwR+9jPgueeAxx8Hvvxlc94FFwD795v9Gzea8zZuBK67Dli8ONp1Fy82r7ZA+9Unqfuv9Dph51dbNiFCWh26O+64Azt27MDUqVPx4IMPxjo3SRFOg2OOOeaQMCTBHXfcgZ07d+KMM87AV7/61ZpdN8vcdtttuPrqq/HYY4/hqaeewr59+3DjjTdi8+bNmDNnDnbs2IEFCxYAALZu3Yq9e/fiX/7lXwAg8LjE0FoH/gGYDWC9tT0HwEMAFIBPA1gcdr7WGnPmzNFJUShoPW2a1oODWitlXqdNM/uDju/s1LqtTWtA69ZW8wpo3dJiyhgdNX9Kad3T4+2Lc135fGQkvD5J3X+l1wk7v1b3QBqPZ555JvKx8n/k/jaq+X+68sor9c6dO7XWWl9yySX64Ycf1q+//rpevny5Pvvss/Xll1+u//jHP+rXXntNL126VPf09OgLL7xQv/HGG3rNmjX6lFNO0aeccoo+99xztdba99xPf/rT+hOf+ITu6enRp59+un755ZfL1uWGG27Qd999t3711Vf1wMCAPuuss/S111576LgXXnhBa631pz/9aT02NqZ/+ctf6kWLFumFCxfqoaGhQ2W+8MIL+sorryy5xmc/+1l91lln6QULFujXXntN79u3T/f09Oh58+bpD3zgA/oLX/hC2fqtWrVKf+lLX9IvvPCCvvzyy/VVV12lr7rqqsDr7t+/Xy9dulR3d3frFStW6IMHD+p9+/bp973vfXrBggX6pptuCvyOVq5cqZ988kmttdZDQ0P6Bz/4ge+5fnXxey5xnl/YMxBuv/12fcEFF+gXX3yxZP8LL7yg+/r6Svb19vbqa6+9Vt92222hx9n4/UYA7NFl9FFrXdYd3Q1gQCn1A6XUnQD6APyv4gW2AujxO0kpNaSU2qOU2rN///4k+goAgN5eYHgYuOce4KyzzOvwsNkfdPy11wJ//KPZfuMNoKdY4zffNGWsWmX+zjoL2LnT2xfnuvL5DTeE1yep+6/0OmHn1+oeSL7p7QW2bDGelHXrzOuWLdX/P330ox/FzJkz8dJLL2HBggX4whe+gPe85z14+OGHcfLJJ+Ouu+7CM888gwkTJuCRRx7BBz/4QRw4cAA333wz1qxZgzVr1uChhx4CAN9zAeBf//Vf8cgjj+Diiy9GoVAIrcuiRYtw5513AgBuuukmLF++HDt37sRvf/tbfPe73/U9b+fOnTjttNPw6KOP4uyzz8af/vSnwGscOHAAO3fuxMyZM/Hkk0/isccew3vf+17cd999+M1vfoOVK1eG1q+npwevvfYa/uqv/goA8K1vfQvXXHMN7r777sDzbrrpJlxxxRXYtWsXTj31VPz85z/HzTffjMsuuwyPPfYY7r//frz66qu+51566aV44IEHAADPPfcc5s2bF3iuW5eozyXO83P5yEc+giVLlmDRokW46aabAo87cOAAfvWrX2HlypXYunVr5PKroZwI74axducDaAUwGcCLxc9+BeBov5O01p/XWs/VWs+dPr1s/urIjI0BmzYBg4PAo4+a102bxru/7ONvuw1oazPbra1GaAHjTn70UeNe3rjRvO/p8fbFua58PjISXp+k7r/S64SdX6t7IPknjQ7dHXfcgWeeeQbz5s3DLbfcgmeeeQZnnHEGAKC7uxvPPvssZs+ejfe85z0477zzsHXrVkyZMsW3LL9zAWDFihUAgOOPPx7//u//HlqXHTt24Oqrrx5X3hlnnHGoPOEPf/gDAGDJkiU4ePAg+vv78dxzz2HChODm98orryypy4knnohvfOMbuOyyy3DttdeWfVY7d+7Epk2bDs1dPe+889Dd3R163nPPPYf58+cDAP7mb/4GM2bMwE9+8hNs2rQJixYtwu9//3u89NJLvuf29fXh+9//Pp599lnMnTsXAALPdetS7rlU8vxcfvSjH+Hqq6/G008/ja1bt+Lhhx/2Pa5QKOCVV17BRz7yEezatQt/FAsuRcrdxQ+11i8X3+8BcABGiAHgsAjnJ4aMU65dCzzwAHDrreZ17drS8Uz7+AsvBA4eNCI8PGwsYQBobwfWrwemTAE+9jHzd+utwCOPmNfrrvOEuNx1ZYx4yxbg+us9KyBpEZN6VHqdsPOrLZsQm7Q6dBMmTMDUqVPxu9/9Dqeeeioef/xxAMDjjz+OU089FU8//TQWLlyIBx98EL/+9a+xs9jjnjx5Ml577TUAZvjN71wA6OjoqKhefuVNmjQJ+/fvx8GDB7Ft2zYAwK5duzA4OIht27ahUCjg+eefDyzTrcs3v/lN3HXXXXj00UexWAJWYnDYYYeVPWbmzJnYvXs3AGBoaAjbt2/HKaecgltuuQU7duzAmjVrcMQRR/ie29LSgunTp+PrX/86Lr30UgAIPNeti99zqfb5udx4443YtWsXJk+ejHe9612BU4q2bt2K22+/HTt27MDSpUsP/Q+lSTkRvUcpdbpSaiKACwF0ADir+NnpAH6WYt1K2L3biMObb5rXVatKt4v/OyXHL18OXHEFcP/9wIwZwMyZQHc3sGKFOe9b3wKmTzf7xQW9apUR2u3bo113+/ZSd5u449z6JHX/lV4n7PxqyyZESKtD99GPfhQLFy7Etm3b8OEPfxgrV67Ej3/8Y5x99tn46U9/iquuugozZszA7bffjjPPPBP79u07ZJH19/fj3nvvxcKFC7Fz507fc6th7dq12Lx5M8466ywcfvjhOO+887B8+XJ8/OMfx/DwME466SQAwIknnojVq1djwYIFOOqoo3DCCSdEvsacOXNwySWXoK+vDytWrMCLL75Y/qQK7uPLX/4yzjnnHADA4sWLsWbNGtx6661YuHAhvvvd7+Loo32dnwCAiy66CJs3bz5k5UY91++5VPv8Vq5ciblz52Lu3Lm49957MTIygrVr1+Kcc87BG2+8gf7+ft/ztm3bhkWLFgEAzj333MChhSRRZng34EOl3gPgazCBWP8EYATAThir+H0A3qe1fiHsAnPnztXNvJThhg0myth2yY2NGYFbvbp+9SIkKs8++yze7c7jC4D/7+nwmc98Bt/73vcwceJEtLS0YP369YcseFJ//H4jSqkntNZzy50bKsK+Jyg1GcBSAP+stf6/5Y5vdhG2LYPe3vHbhGSdOCJMSDNSjQjHHtPVWv9Ba/2NKALcKKSZrCKtaFFCCCGNT8NkzEqTefOAgYHxwVgtLckJMaf/kEYmrseMkGah2t8GRRhGFG+4wURFr1jhRUPffHMyWaM4/Yc0Mu3t7Xj11VcpxIQ46OIqSu3t7RWXwaUMi6xaBTz1lEnE0dNjBDgJt7E7BtzbS5c0aSze/va3Y+/evUgy8Q4heUHWE64UinCRsTEz/7enxyT0GBxMRiTDpv9QhEkj0NraWvFaqYSQcOiORmlCjmefNQL8la+Mz5xVCatXjxfb3l5O1yCEEEJLGICxSmUMWKzWv/gLE808axYtVkIIIelASxjGKpUMWCK4q1aZjFrMGkUIISQtYifriEuzJ+sghBDSfKSWrCOvpJmwgxBCCPGDIlxk3rzSRPMSrJXEPGFCCCHEDwZmFbHTSw4Pm6QanMtLCCEkTWgJWzC9JCGEkFqSexGOM9bL9JKEEEJqSe5FOOpYb1qLkRNCCCFB5F6Eoy4lGJZekhBCCEmDppknvG6dGesdGTGWLiGEEJIWnCdsETbWy/nBhBBC6kXuRbjcWC/nBxNCCKkXuRfhcmO9UceMCSGEkKRpmjHhcthjxocdZixhW4jHxoxwcwlCQggh5eCYcAzcMeOWFrqoCSGEpE/Tp620x4x7e83fsmVmfWGmsCSEEJImTW8JB40Zv/kmU1gSQghJl6awhDdsiDfGK8ctW+a5qMVKJoQQQpKiKSxhv2lIF1xgxn5tZH4wU1gSQgipBU0hwn7TkK6/Hrj5Zv/gK6awJIQQUguawh0NGAFdssSbhrRqldk/MAB87GOlwVd+bme6owkhhCRN04hwSwvwla8Ag4NGcA8/3FjCl1ziCTNFlhBCSC1pCnf02JgR3FtvBR54wFjE110H/OVfmu0srx/M3NaEEJJfmkKEZYx31Soz3eiee4DFi4EvfSn7wVfMbU0IIfmlqdJWioANDwO33AKsXAl87nPe5xs3Atu3A9/5Tv3q6IddbyYOIYSQ7BM1bWXTjAm7mbEOP9y4pE86yVjI4rLesqXeNR1Pb6+XOIRj14QQkh+aRoTdaUcSHT0yAvzmN9m2MN3c1ozUJoSQfNAUY8JBGbPefNNMT8pyakomDiGEkPzSFCIcFNzU0lJqYWZR2Jg4hBBC8kvTBGa5wU1r13pjwL2948eMCSGEkEppuvWEy82ntYObhoeNK5oWJiGEkHqSGxEuN5/WDW5yx4gBs+23qhIhhBCSBrkRYb9FGvxczZUGNzFzFSGEkKTJjQgD413OYukmEdzEzFWEEEKSJleBWWlnlmLmKkIIIVFousCsWsynDbK0CSGEkErIjQjXYj6tG9yVxXnFhBBCGodcuaPTxJ1HzHnFhBBCgmg6d3TaMHMVIYSQpKElTAghhCRMopawUupopdSTxfd3KqV2KaU+VW0lSTw4V5kQQvJFVHf0rQAmK6UuBjBRa70AwIlKqZPTqxpx4VxlQgjJF2VFWCl1LoDfA9gHYBEAWfb+QQBnBZwzpJTao5Tas3///oSqSsKyghFCCGk8QkVYKTUJwAiANcVdHQBeLL7/FYCj/c7TWn9eaz1Xaz13+vTpSdWVgHOVCSEkT5SzhNcA+JzW+jfF7QMAJhffHxbhfJIwnKtMCCH5oaXM54sBnKuU+jCAvwBwPID/B+BxAKcD+Em61SM27tzk3l66pAkhpJEJtWS11mdrrRdprRcBeApGiAeVUhsBLAPw7fSrWHuyGoXMucqEEJIvYs8TVkpNBdAP4BGt9b5yxzfiPGFmxyKEEFINUecJl3NHj0Nr/Wt4EdK5xI5C5opJhBBC0oKBVQEwCpkQQkjaUIQDYBQyIYSQtIntjm4GwqKQd+82Gapsy3hszOxfvbp+dSaEENJ4NIUlHDfaOSwKmakjCSGEJEVTWMIinH7Rzn74WbRiEQMM2iKEEJIMTWEJ++Vcvvji8cdFnQvMoC1CCCFJ0BQiDIwXzuXLK3crS9BWXx9w++2lru4sJPUghBDSGDSNCLvRzkBlKxLZrux3vhN44w3goovM/rEx4MILgeefT/9+CCGEND5NMSYcFu0s1vHISDS3shu09Y//aIT4b//WfKaUsbIJIYSQcjSFJRwU7bx5c/y5wKtXl5Zz333m/UMPAW++abZrPUac1VzXhBBCwsmtCNvCJMLpCtO99xoxvv56zzVdSVIOSb8dMw13YnDaFCGENCa5FeFywpTEikQyBjxpkgnSmjDBbNvXrIU16hf9HXXaFK1oQgipH7kV4XLCZLuV7XPiZL3avNmMAd93H/DJTwITJwIHD5r9lVqjUUTR7xgAOP30+NOmaEUTQkgd0Vqn+jdnzhxdT0ZGtAbMa9KsX691oeBtFwpad3Zq3den9bRppZ9FpVAoPdfdDjqms1Prri5zn3GvLeVVci4hhJDxANijI2hkrkW4HuKShOhHqbd9jAhwmHDXot6EEEIMTS/CUSzKtK6ZhOhHEUU5pq9v/LUKBWOp17rehBBCKMLjXMVaxxOmoPLscu3toaHkRD+uJVyNcNajs0IIIXknqgjnNjAricArGwlgamkBBgaAD33I2162DHjpJZOPOm60tRtkNTZmMnBdfHHw1Ck7+Ui106uSiBInhBBSGQ0nwvWYUiNlb9kC3HwzsHChSe5x+OFme+1a4PHHvboIvb1GvMPq5kYnb95s5htL1i0/UUxSOJPurBBCCIlBFHO5mr+k3dFpuk+DXNi2q1nGYSdONK89Pd5nldaNY7KEEJIvkFd3dDWJKcoRNGd2+XJzjQsvBEZHgbY24E9/Ak47Ddi5E1iyxMtJHbduYiXbKzzZ+wkhhOSYKEpdzV9agVlpTakJskoLBa2nTDHXnDxZ6+FhrZXSur/fvI6OVlY3d45vV5fZpjVMCCGNC/JqCQPjlyWsJCApCHfdYbFiN282QViSnvJrXwNuvRVYvNi8rlvnLWcYt25KleafViq5+yGEEJJdGm4pw7BlCZNwSbsiKmXeey9w//1m+5prjCjPmuV9PmuW2SeLQkSt2+7dJu3l2Fjpkoq7d9d+NSZCCCG1peEs4TSn1ARN/dm8ufSa//APRpDta/b2Au98Z/y6SRSyLfz2/lrBhRwIIaT2KC1+0JSYO3eu3rNnT6rXSIoNG0xwlm2Bjo0ZEU1LFF3L3t2uFVmpByGE5AGl1BNa67llj6MI15d6CH8QIrzDw8YipwATQkhlRBXhhnNHx6ESF2st3bJZEmAgOCiNEEJIOuRahCtZK3fePJOWcuPG0nNaWjwhrkbc7XNbWoALLjDXkv31XMs3zahzQgghPkSZx1TNX73XE64kG9XoqJn7Ozho5gYPD5eeOzqqdUdHvMxYcszoaOmrzDceHKwsW1ZSC1VwIQdCCEkONPsqSjaVJPYYHDTnnHZaaTIOV0xdcQ8TRTl3cLBUeOValSQeSUo8k151ihBCmhmKcJFKLGE5p6fHPKH+fv8y/MRdzh0aGp9PulAwa/9KzmnAE+Jq8kYz9zQhhGQLirAOtxLLLdYglq5Yrf39pYIbJnyFgkk/OXmyeRUBlvSUdpl+VnYlIppWGk9CCCHxoQjraK5hV6CHhjwBls+Gh0utVvdzP/EUUZw82bwXAbbHhP3Gm6O6gO17s93cU6bQEiaEkHpDES4SRYjDxnXtMWDZP2VK6YINdpn2OSMj3qIPfX3jLXDZHhoyf0Hl+REU6OV2EAghhNQeinCRcoFL5dy4cQOW3DHgzk4jxGErI1WzDnFHx/jIagZUEUJIfaEIW4QtTyj7OzrCrdswbKGW9+JuFjGWseag8Wg5Pm5wFceCCSEke1CEHVyxcqOYZW7w6Oh40RTKBXPZ4h4m6kGWb9ypSoyKJoSQbEIRtvATKxFDWxBHR7Vuawt2H4e5jeMKont80LzjcuczuQYhhGQPinCRKGLlF0jlWqOuaI+MaN3ervXSpd4xYm339UWrmxzvN6ZbTlCZXIMQQrILRbhIVLFypxQFCXWh4B3b1uZZzH5zg8NIajw6a7BzQAghFOFYBCXXcMVR9rW2ei7r0dHxEdDlLNk8u5Kzfm/sJBBCakFUEc71KkpRkJWLLrsM+Pa3gfvuM9sAcP31ZkUhWU1o40bg978H3ngD+NjHgPvvN8ccdhjw2mvAtdea5f96e81avLt3+19z9+7StXrLHd9IyL0sWwasW2des7QucSUraxFCSGpEUepq/rJuCZezjGy3cWtr6RiuTCtqb2eEskuWp04xqpwQkjZI0h0N4AgA/QCmRTne/su6CEdBBEUyZdk5peMuaVhL6uV6bQSRy3IngRDS+EQV4bLuaKXUVAD/G8B8AGNKqelKqTuVUruUUp9K0UjPBGNjwOgoMGsW0NJiXoeHgXvuAd76VuCKK5J1K2/Y4LlK7Tps2BD+mR/1cL3KNbZsMa56cU279a4nY2PApk1mqGHTpmzVjRDSZJRTaQDnAOguvr8VwGUAvljcvgvAyWHnN7IlLBbdwIC3+EJnp7F+29qMGzppK29oqDS6WoLGJKlI3KCnWlulftZ3Jbmx0yLrgWOEkHyApKOjAZwN4JGi8J5f3LccwAfDzsuqCEdx1dpzg23xnTTJi4xOugGXa3V1GeHs6ipNHFKJqNbb9Vqt8CXpVmd0NCGkFiQqwgAUgL8D8E0AdwI4vbj/PABrfI4fArAHwJ7jjz++hrcdnbjCUCho3dKiD80lts9bvz7Zxr1QMNdwryXEEdWsjM9WUw9ar4SQRiNxS9iUiRsAPGu5py8G8Imwc7JqCWsdTxgKBS+blt+avUkKRdi14tY5S+JVjUWelc4EIYREITERBvBxACuK7+8AcCWA64rbnwVwedj5WRZhrcOFwXZHi1t4cNC4pP2yYiUhFPa1+vqMCNsJRDo7vfHVcqKaJddrEs+m3m51QgiJSpIiPBXAtuJ48OcAdAF4GsDGolXcFXZ+lkW4nDDI50uXGvGTVwnQkmApW9SqFYqhofGpMKdM8YKbXPHPUtBTEElY5LSECSGNRCruaF0qzMsAHFPu2KyKcFRhKBRMQNbgoBcwJQIpSxiKGNtC0d5uoqrdssqJo2u9ihD39YV3FJJwOadlOVdbbtbc6oQQUo5URTjOX9ZEWATBFgZ7208YbOvWzyKzI5ple8oUb31iuYYt2DblBKmcdZ2UlZhVscuSW50QQqJAEQ6gkqhoV+D8RFHcyPZxktbSFWz3+mGrKEUV2HJCHVXI6PYlhJDqoQiHEFVo/ATTnsPrnusnhH773Ou7843lc3e/PV5slyVu8bD7idP5YAAUIYRUB0W4DFGEJmh81o1OHhryBNMW1jBxdK/v1zGIcv2gZB5hQizX8HONj44ay5yWMCGEVA5FOIRKXa5Llox3G4+Oaj3c1I6XAAAgAElEQVRzZun47+ioebJBizsEXT9KxyCKkIaNl/qNb8v5o6P+49hpCDHHeQkheYYiHEA1wUdB5/pZwkuX+k8dEus4yPUcpWNQqbs4KKhM9skqUe45aQhjVoPACCEkCSjCASQ1XSbqmHCU60sAly1IQYIYZfw3rN5+HYjBwdJ612ruMYPAkoceBkKyAUU4RaIEW8URlCBhdt3ZccZ/o1yjUDAWu1JGiMUit6dbxb1OXBgEliz0MBCSDSjCKRHm0k264ZNy+vpKlzOUz8LmNscpX1zhg4PemHAtrFRawpVRztrlcyWk/lCEUyDMpSv77HzTdqNYqVCKpeiu3JRE42o35nKdwUGvrmlaqbTYKifKs6OHgZD6QhFOgSjjbUmKi23RiCs6DevGtrjtvNViHdtu8aTg2GV1hFm7tIQJqT8U4ZSII8TlIo7DyvITc1necHAw/PpxsK8j486ycpO4qd2kISQbhMUm0MNASH2hCKdE1EbOdu8GHR9Wll+ijs5OrWfPjj6XN0qHwe867e1az5o1vm5Zt1KbyboOsnab6RkQkmUowilSzt3nfh42B9gv53Q5F7ck1XAFPqieca2iRh1PbBYrsFnuk5BGhiKcMkFCFdRAunNx7ePFzeyXxUprf+smqDyXuOODjT6e2Oj1jwKtXUKyD0U4RcIa+rA5v0FBNJ2dRognTy6dn1vJ9f2IatnmxcJqVEueEJIfKMIpEVeowo633/tNRUry+lEEOw8WVjNYwoSQ7EMRTom4QhV2vD2n2J6K5KaMrPT6ebFso9Js90sIyS5RRViZY9Nj7ty5es+ePaleI6ts2ADMmwf09nr7xsaA3buB1au97WXLgC1bzHHudtrXzxPNdr+EkOyilHpCaz237HEU4fSIIrDnnw8sXgysWuWdt3EjsH078J3v1KaeFC9CCEmWqCI8oRaVyTsbNhjRshER27LFCO+6dcDAANDdXXrc4sXAJz8JXHONd97IiNnvlrdhQ/j15PO4dd+5E7joIq+MD33IdA6ef766sgkhhITTtCKcpJDNm2eEVsoTi1dEbHgYuOEGYOFC89n555vXjRuNAB88CHz1q0aoly0zx15/fakwX3AB0NJSer2NG737WLbM7K+k7tu3A6+/boR4+3Zg0yZTJ8DU4aKLSsumKBNCSEJEGTiu5i+rgVlJBPHYQVL2fGBZG1jSQLa1ad3fbxJstLdrPWmS1hMmmLA4SWs5aZI3rUamLXV1jU/4ETdhRxQGBsy1pU7yOmtWdcsnpkkeIrkJIfkFjI4uT7XTWVxBkgQaktu5UDAC3N5uxLa93QjnEUeY40R0RVBbWrRubfVEz53vKteThRbshB1xBMgvVeXEiV6dAFMPN4FI0HMqJ4hpCGbcThRFmxBSSyjCEak2sYNtAbuW6fr1RmAlI9aUKVq/9a2lYidW5/vfX3rc8LB/gg+pb1ubZym71mrUOsvxw8OldZI/+17CnlM5QUxr6lCcThSnLxFCaglFOALVWMK2ZSUWaX+/1kuWlLqOJS1la6sRacC8dnVpPWeOtw0YS3lkxLwCRhzXrzfzhu1VjeR6ra3meHFdx6m/m06ztdUIu20JyzrJYcso+s117uwcP9c5rSQacTpRTORBCKkVFOEyVGsZyfGSklLGfIeHPUtWFmfo6tL6Xe/yBK6jw3yulNbveEepEA8Oei5spYyFXCiYc9razHZXl9l2x5HDVkeSOtsuYhHgk082ZbW3a33SSd649cCAdw8iqmHPra/Ps+Ttz+WatmAm4R6uRFSZ0pIQUgsowmVIQgRGRz0L2BbkM8/UJYFNsj1njtk+7jhPYKdMMccBWh97rHnt6THCN3OmZxEPDXnjtC0tZl9XlxE+uXbYyksi5LIEooxD9/d7Y9GyRKIt+mINhz0nOwBNOhm2dexayn6BZpV2guKcT0uYZBHGK+QTinAKuD+W9es9l7IEY8n4quwXS/f97zefFwrG0hT3rwiUuKBPO82/vOOO8yxpsbpFMAcGStcYDkqHOTBQKoBijZ90kmfFh7mUgygUTM5rqbcsSCHj1EGCGbbEY9zvQsoNarg4JkyyCv838wlFOAXcH4dtTdpBWSJuIpzuIygUtJ4/f3xgVXe3EWMJ8hKRbWnxXNaTJ5tjzjzT7JeVl2xL2K6nuzCEjFFLXe2I7vXrK3PXypi1iKmU2ddnPg8TzErdw0nm8Cak3tBLkz8owinhRkOL9WkHZ02bZl7FsrWP09oTBBGg2bONcLe0ePOGZ8401rI7dWjmTM/tK58FRSwHLZEo1xUhdl3Esm27om3r2hb7oaHK5y9X0vC4Vr68d+thH2tv+7ntayXElXQE2HloHhivkC8owilii5jWpcLc2mpcz7YQiYUromaPuYpYT5rkTUsSMRfhffe7zZ87hailxRsTdsduCwVPrO25vnaEtd05EPGUbRFxEW973Fcs9ClTjItbhG1oyNR/6VJPPIJWhbJFVMTRdcmVWxnK7mj4TdEK8lzY91hLq6OacWy6KvMNLeH8QRFOkLDMWK54SPap4WHv+I4OI85Ll5a6q9vbPfeyPZ4q4qmUNw2pq8ubUyzzhAFPEN25wkuXllq7hYI3Z9lP+IKsRNttLlHb8iqBaDK27Y5NDw2Z682fP15ch4bGR1zL9csJjd1gydzqIOshKCCsXo1dJY0tG+h8w45WPqEIJ4hrRYr4yHQk27KaP9/sswWms9MLirLd1hLMNDJSOjYr+ydPNq5q1zIWV7Q7H1lcsq6Va08zcsVQXoPcnkNDpUFXIv5tbV6dZX6zPJvOTi+rlz2/2X11LXc7G1hYXbUuHesuJ06um6/ebr9KpmrVu84kPTjkkE8owgkjFq1YlraguNZbkOUiDWlPjz40n9YOzBJRk0xZ4pqWwKzjj/eirsUKHh01CUIE11Uudbfdwn49bXEvu67a4WHPYrcFWOrkdiYKBf+Ul9KJCBsrlrrbHgY/4Zb7sce6g6yHpCzhpBrKoPqEWUG0hAlpPCjCKRCUy9mvcZTEFfaxXV3GslXKCFlnp2c5trebseOODm+/JOSYPt2b1iTCN2WKCdLya8D7+sZn0JLywhpyN6hK3Oajo14HwR7HlkCyjg4jiJIZrK2tdJzWDlIrl/pS3N7S4ZH62NaxPQbudn7C5kpXMyachMswqIywjgFdlYQ0JhThhCln3drC4gYMDQwYURGBle2ZM72x1dFRI2CSmGP+fE+M3exWrst5/XpTlp0ko6PDiKSMx9qucFcE/VJwnnaaqY8dhGW7o2VMfNKkUiteLHm5R7GY58wx9Z01q7zQSBIU12tgu6Xd78bPIpUUovZ9Dg+Xeg7iWLPVWqSVTNWiq5KQxoQinCDlLBgZx7Qts9FRb/6sTCUaGPCOETFubzevWpfOO3ZTP9qucHdur+0CFtEUsRTRjmJtyTFisfb3l967ZO2SukngVXe3ee3r86Kj7UCyY48199vRYfYvXVoaROYGhcn9ynMTr0E50fNbHSos5WYlpDE2S3czIfmDIpwgftaIHWksrmZp8O1xPnsusF9D67qA7bFTrYNdqnZ0thwnkcITJnhi7jcOK50D+54kM5ecI5ar3XE47jgv9aU8k/Z2I5ZuB0XKmTDBCLdtpcv1/cTR7hDIWLNMi5L7tS1B973bWQpbfKKS77ycSz9KOeXqnBUhphVOSOVQhFPGz+qycznbguJmknKtKNsF7Lccop8FLtajX8IQKcuexiNiKuV0dpbO550yxQv6koQjMm/ZDo6SiGcJKBN3uk13tyf+0jGQPNT28/LrlEh9RKjlno47zgsSEzf90qWlHgOtx2fvKhRKn3tUYUlqPDlMZGWqlvt/NDSUDaHLcgeBkKxDEa4DtqsyzIL1a9hsF7C9327wwqwzW4DFDWzPVXbnEbsWtUy3soOypFzXNS7jvJIKU/bb1rFYrfa8Z3G7uxnDBgdLhVDuSQRKrHQJ+rKXXXQ7AIVCabS2+9zjLBxhdxRsr4P9eRSxDOpwNILI0VVOSGVQhGuM21i581z9pjO5847thRnkvHKN/Pr1pXOIR0eN2E2cOL4smUfsWubude2xab+gM3uqkm3x20Jsi59Y3baHQCxqCTCz3dW221kscfsepZNhT8Oy6yfTl1yhdp97FGFJagw4qJxGEDnOUSYkPhThGlLOoglygUr0rmsxuy7WMCSS+Mgjx7uOBwa0PuWU0g6BNKhHHmlESqy84WEvyYZYwjKP2Z2fK9HQMtXKHV+We3bd7/Y922O1dsYwtwMjYt7SUpoxTNztrivXHYuXjoJ9TNSFI5ISyHLlZFnkGqGTQEgWoQjXkGoCWOxMUH6u3bAyXMvOzZQlx9iZq7q6vEjq9vZSy9SefmRbtgMDnjUr47FDQ142r9mzg+vm54IVS9fOemVb3baY2u5vpTwRlmlQdgdg/XpTR8lOJksyDgz4zx8OE5akXMXlysmyyDWCu5yQrEIRriOViHIlDZ59HXt81U1DKcInQibzkSVJSH+/2SfpJ9vavPFbWZSho8PMOR4dNXODJ0703Ml2xLR9PT83sNTLXuFJ5k+7Vve0aWZese2CPvZYL0Cro8MLOpNyOzrKL0wR5TknFRkcVo4bpS6ehqjrOKcNo6MJqRyKcB2p1IKIYhWFBWfZmbJc4RMxc8dyTz7ZEznb5WyvliRzfCdN8qxSe6zVFTq/VZGkjva4r3Qa3NWa3Kjmk0/2MnXJ+LFY47JohFjv8+eXBpWJa33JEs8dHjQ0EGd/EkIkOb3t/xN7XjMhpHGhCNeZSt2M5cYHXUG3BdB2PRcKpQFWMn47aZLnzhX3tR3oJGXbojk0VCq+sv6xPWbsrsBk19VNoynWdV9f8NxfSQTiFzQm49B+U7/kmuLelulWfotGhD1X19Wflks2y+5oQkjlJCbCALoAPADgQQD3AZgE4E4AuwB8qtz5zSrCWscPuInaINvHuVNnRIglwErEyLZ+xQoFjCjbVqe78INdhp0v2p7D63d/7jSkyZNNPefP99zPIratrZ77W2tzXHu7OdZNfiLWr+TmFkvanp8t1rE97auryyvPfq5Sph21HjSVKC2hzHJgVhLQrU2akSRF+EMA+ovvNwFYAeCLxe27AJwcdn6zinDchjuuCzus4bYFqqPDWK4y71byJ4v1684RdiOz7fnHtiXtWqN+jayd2ENWhDrppFLLVtY9ls5AoWBEWfJW22PE4r4VEbdXnZJjZIUnsdxFiCdM8ALMbNe3vR30XN1I76C1lysRlWawhBngRZqRVNzRAL5RtIjPL24vB/DBsHOaUYSrDbKyy/Fr2MMabhGWtjZ/N7UkvpAgLJke9P73+y/JKGkp7YQgIpLllka0pyH5RV9LEJgk95AFIQCzIIU9Rix1PeMMLyirr8+r3/CwF8ktAmxPabLzattBYW5glN9zDVtdKur360fWxClNi7UZOhuE2CQuwgAWAHio6Io+vbjvPABrfI4dArAHwJ7jjz++dnedEWrRmPkt5WfPMZZxV3fN3fnzjXCJqMgSieKuFitPEnuI+/eMM0rPOeOMcEvQdUf39/u7yQcHvUAw2+L2y7/d1mb+hoe9rFiTJ5sOBGCsahnLtsVXsoe55blzmN3nKl6Cri5v2pMEstnBX24e7Kjfc7n/k1q7cdPuFOTd7U6ITaIiDOCIoqieAOA2AN3F/RcD+ETYuc1oCaeJNMyuxSqia48PS6PX1zf+fLFqe3r8VxqS6UjiHga0PvNML9mGUiZHdJRpVyK4kkbTXixCLEtbMN0lGWXlJpmjbK+t3Npq9g8Pe67o1lbj9rbHsUVcxaXtWsJ+z3V01BsDt5+ZlCfbtis9jaCtWlrKaVmsUcrl2DHJE0mOCU8qWsAyLrwCwHXF958FcHnY+RTh9Cjnli73mazV66Z/FFeyiNTSpZ7oSZCTuJGDGmlbMGQc2k6FKe5ve+1g2xK2Xed25HNnp1eXd7yjVIilbFkCUcqSMWR7AQoJ+HLHiN0ALTe9pli+tvvcLyd4HMLEpx5uXIkpcNOVViqGUTsTWXPPE1INSYrwMIBfA9hR/LsSwNMANgJ4FkBX2PkU4XTxc/GFNWby3i/Llli5drliVXZ1eWOtLS3+qSpt/ITFzlVtRzvbU6TElXzyyd5YtJ2Pu7vbmzMMaH3MMeb12GNLRV7Kkw6FJBuxLXx5DgMDZr/9zOzn6kaO2x0Hv8+1Ln2Wgjxjl3LiU0s3rtsBS8LCTyregZBGItV5wgCmAlgG4Jhyx1KE0yOowQpr9MT6tKc2ybaMrcr4qSto9tQmv8UT4tRVOgGzZ48PcpLxYXEDS51FoCXb19veZraPOMITd5nyNGlSqYi405Bkvys4bl0lq5id11qWUOzr86K1Z80qjSyXaVL2MwZMJ8Lvewn6LmspSu6zEbe9nVCkFnDsmOQBJuvIMUEZqaIu/BCW0UqEWNY2bm/3RHHiRPMnlqtYlEND4WvgSpCTG01sB2fZx3Z2GutbGn85fmDACKAsZ6iUtwbynDmmrq7whrk9R0ZKlz20P7NF2q6HdATsVZ7sKHT7XPmsp8d7ln5i71q8fX3B4/5pCrHbefOLKUgbWsIkL1CEc4w08EG5meOU4zZ469d7c2tbWsw+WRpRxlhlWpMELdnJN/zwE30Z77XHUt1xZHeecFeXEQTpFLjZsESg3eQlfh0EO+Lavn/pUGjtuZRt1+nAgJk+NW2aN0VKpke51ysUSl349j2GWbwi/H4R8NWMzcahHmLIMWGSJyjCOUcaKHGJ2g1VnIbadf1J5POMGaVjxTL9R6KLbXFxrx/kDhdLzl3tSe6l3BrMYrW2tHjucXftZTvdpl2O+3zsecsitLbV63euvW0vmBF0jBt05o63+1nfsm3Psa61ENVLDBkdTfIERbgJkMbcdkPHaTBda8dNQiGCbM+x1bo0i5bf2F1U8XLPCco9LWOzfvOey7mbgyLDRfDtRBzt7aazYSNufr8VntzxbckMNjrqlStzicV9Ls/S9T743YNflHItoBgSUj0U4ZzjClRYCsmw822hlCUObcQSluPtrFnizvVzRfsJYSUuTrujEcc9GxTcExa1HWTV+i0Q4R4jaytLGTNneqIuQi3j6fa57qpJssShLeyyljNFMNuw81JKsz8PinCO8RMCN8CoHJX8QGwBtufdBo0J20JYzsUZtESjHYUc1eUeNNYd5iL3s2r9tt0FIKSeknrTTijidlAGBkrnVovb2RbhQsE/+Mt2Z5NswjHtUpr9eVCEc4wrKNKY2ysJJU2hYETmbW/zFleQ/TLP1k1jWU68bCF1f6D2Eo1+gV1BIuyWMzRkno2MFdsR3e3t5s/OFtbZOd6qLTe+bAd1icvczpMd9lz8viu7HFvYm8WCaGQq8fbkmWZ+HhThJqEWvU3X/etO2wkTpjh1EqGXpB3uPGY7SCuoPL8Oirh8RdglI1d39/gAsY4OL2LaTQMqx/jlyJbP3OAxu3whyjxYv/zWtaCRXYhx6p7mfXKecynN+jwowk1CLRrNuJZ33DrZx9tZtWzBdd3CcToZIq6Dg56FKjmzZWEGe2pQpR0a23Usrmg/C77cfcjztS3hsDok+T/QyC7EOHVP6z6b2fLzo5mfB0WYpEqSvVtXaCWJh53estpriri3tXk5s9vaShdisJNk2G7yKGIm92CXJ1Hr9ph0FPe262lwOwdB105KUBq54YxT96Tvs5E7MGnQ7M+DIkxSI41G2l2z182mVck17ZWRJMuWnfkLMFHM9sIMYhnHvSd3FSaJanYjn8tZrBIdbQu3jDfLdi1yLjeyCzFO3ZO8z0Z25adBsz8PijBJ5UeQVu92/frxU4XsseFKx5jFily61Mv0JWku5f373+9FM/uN4UYlLYs06tSspASFlnB9aXbxygsUYZKKYKbVQEjd/LJpRQmQCkKio0dGzKIOgJf7urvbW5s4KJo5CL/nYKe8jFvPIOzx4bAkJX6CUsl31cguxCyMCSdBlutGokMRJlrr+vb2g0RAkn/Y+8R1u2TJ+AxdQUsARkUsxMmTPWt70iRvPWD5rNpkJ+65tmvZPi9ssYty9Q/LAubWpRIPQiNbYVmJjk6CrFrpJDoUYXKIeo3vRRUHsVZtKy/KdKQoSNl9fV5SEUkjKVOJKl07t1xDKZ0Lu2x7Owr2NWQpyShZwGxLmI15Y9LI4/KEIkyK1LsRDrp+WL2SqrM9Jmxn95K5wJL9y28ZwkoXwPCrg6zL665bHKX+dp0qXd+XjXnjUe/fLakeijDJzNhSkAiEiUMSwmFHF9tJQNraTIav4eHSNXqjur3dSGixst30k+vXlwac2Sk844w7u2IcZ11hNub+RHVH18NtnZXfLakOijDJxLhXPS1hG3uB+iSEzXYry5KFsoSiXbZ81tZmLOFyay/7Iesa20TpMFTTmGfhfydNoj6beghi3p99s0ARJnUnbEzYjni297timESjV0lHoByyrrK90ENnZ2kWMRFreyxagsLk3m1r1y9BiF/ebHfN5CCqacybwRqL+v3Tm0AqgSJM6k5YdLSfsNjLFbrnVGoFlBOTatze7rn2tkRGy/3IZ7Nnm4QgdjatsFzcdgfFbwpXmjSD+ET9/jmuTuJCESZVk7ZbrNJGPqmpKJVcPygzlqwb7C6B6BdU5bdecEeHfy5u2xKWdJuSVzsulXyfeRYfWsIkTSjCpGpq4ZKspJFPol6VluFnubrr/drWq2TisqcpdXSY/XaGsKD1oKUsSeN52mmVW8Jx7znP4pPlMWGSDyjCJBHSbIijlB1kvcnYcaX1qsbKt7NwTZvmWbbuusHr15euTSznTpniLaM4Y4Y+NE5sW9F2WcPDpQLsJjOJc29xrb+8ik+WoqMZiJVPKMIkMdJwSSZhidTDVSoNpn3toAZT6irjuIODnuu5q6t06lJ7uzl+YGB82s62Nq1PPtkTcz+h9rtuNePgFIbaUWmHpxG+o0aoY1pQhEkipGUJx/lx+tUhbVdpmAUu05P81iB26yz7bdezWMNTppiMXUoZS7ivz5QnlvKSJeZYEe0gS9mvznYHQILAgp4lqT+VfC+N4K1ohDqmBUWYVE2WfkCu5Zl2vYKuMTrqCa8EVfllwQoTRIkEl/sRd7O73d8/vvywew0T/rB7ylqD2KzWUzXxEVnuVDVCHdOAIkyqJiuNofsjTnIaU9zxU9cdPThYem23Hn7C19ZmrFspd2DAW0iipUWXLLUoyUWi3qufC9ytexLPLS6VeD6y3FlI+llWI1SNEMHuJsuxqVcHK+3fA0WY5IK0G+RKxk9doXOjou26uT/0QsFYtpLiUjJqtbebwCt7qcXZsyu713IWsB+1mo4W9XvMuvWU5P9lNWVl/TlpXVrHOJ6dWtUrrbpQhEkuqIX1FtSQ+e13f6ijo6UWZ5CVLss3um5qWbpRLOCjjjKvEyf6Xy/KvXR0lFrAWpdf67gW1uf8+cYNbz/PsPSbtbbw4v6vJSWAlf6PN4LHwK+OdkxFveubZieGIkxIDNwGP6iB8xNZsTzDxqvD1vYVa9heXtGdd1xtqskoDXahYERy1qzSYLNCIf4ayH7Ifcqzkg6MPec5KBmK24FIg6iiZoumPSxRa5dqVoaLwgiqY19fdlzoaXX2KMKERMSvN+zXePhZkxJk5VrLftao5Hx2e91iDUvCjrY2sy0WYtSGtVyjHKXXLx2KtjbvXoIiwOMiIi8dDaXMeLjbEZCVrpK+ftQ6lntGbseqlqlE80CWXOi0hAmpM3FcekGuaL8x4aBxWdtqtssVF5006H4rMtlUagWF9frtOiplhLCS9Yv9sO9D0m+2tPh7CNrbx3dWamnhRbGM3GEI9z6IP1lyoaddF4owIRGoZBzQXpvYdRmL69YvQtnOF+2O19rBKtLAz5pVepzftCeZMxylAQnr9bvnu2sgx32W7upQkgN76VJzbz09pZawW696RfxGtYzsdaLtIYwsuYKzSJZc6IyOJqRBkca3v99s24LoLsVoHxvU6/ZrDIIsZrsMya5lu72DGpFyvX5X4Lu6jCXc2hrdEnbHoMvl2LY9CX7zwGvtrqzEK5IFlyrJJhRhQlJAGl9ZUEHEVaxcO3DLdu+2to4fMywnmGEWq2TrkvnFfkFhNlF7/X7iGWdM1q6fGwU7f/74ZzA6avaHTWGxPQ7lnl01xHlGWXGpkuxCESYkYdzG1l7ZqJx7N2pDHeU8OyK3q8sEdJUbu42zYIEbAS4u9qiiZ1u15dzK7v1JWlD7+hL8lhXRy5JLlWQXijAhCeM3JisJNiQAy+9YIUpDHTXC2Y7IFaGzhcqlVtZbmCVcrYUu5Urqz3LnEFJPKMI5h73x+mGPAddyioo7juyXxzpsPm3a45jlxoTDkplI8FbY/7TtAWh0dzB/v/mHIpxzOC6VDJU0hiKAfnOA/QKdopYbt66FQqnwRvkfSDPqOCw6WurnBq65wl3uM3slqUYOjOLvN/9QhJsARmhWT6WNYVS3cZqNbBJpFuthkYX93/p9FvQs/SLIGwn+fvMNRbhJaIQVVLJOWo1hlhrZICELS6eZJrZr2a2nm9LQr6MQlH2s0eDvN79QhJuALDXyjU5ajWFWGtm4SzaWO6ca/ILLZH+UYK68uHL5+803FOGck5eGqBYkkVO5EhqpkfVLqO+OcyfxP+aWYad/jLrMXR6Cmvj7zT8U4ZyTh4aoVkQJ+Em6MWykRrZQ8OYbiwi67uqkOhJ+/7ezZ+txC77bVnoe/6f5+80/FGFCLGrtcm2URtbtkHR2jk/84edST+r+/K6Z5Q5LGI3ynZPaQBEmxCHri8TXA7eO8oz6+sx2UOclCUvf7QC41nitSLJD0SjeD5I+FGFCLOKOzybRMDdao+w+IztyOmxudKWu6nIdgFqR5PdUiziARujckYRFGMDRAHYW37cC+BaA7wH463LnUoRJvamkkU2qYW6U4Cy/+7XTQ9pjxHZEdSdjBHQAAAkiSURBVFJzdev9nJK8ftoel0br3DUriYkwgKkAvgvgn4vbqwB8pvj+OwDeEnY+RZjUm0oth6Qa5qxMUwojyjMKspSrfT5ZEZUkvqdadSbq3Wkh5UlShDsBdAHYUdz+JwB/Vny/BkBv2PkUYdLIVNswl2ssG821mEb+5lqk+CxXZhKiVuvORCN07pqZxMeELRF+CEBX8f0QgOU+xw4B2ANgz/HHH1+zmyYkSaptmKM0ylmxAqNgP4+0VjKqR5BUUt9BLTtUtISzT5oi/E0Ax2jPNX152Hm0hEkjkkTDXMkyfVltUGvVWahHkFSjeSMaqePWzKQpwusAXFp8/yUAZ4adRxEmjUitG+asuxYb1crL+nOthEbrNDQrUUVYmWPLo5TaobVepJQ6oRiQtR3AmQC6tdYHg86bO3eu3rNnT6RrENKMjI0By5YBw8PApk3Ali1Ab2+9a1Vf1q0DbrgBGBkBrr++sjL4XEk9UUo9obWeW+64CVEL1FovKr7+HEA/zBSlxWECTAgJR4RiyxYjNlu2mO2xsfSuuWHD+PLHxsz+LDA2ZkRzZMS8VvIsyj3XrD+DMBq57mQ8kUXYRmv9ktZ6i9b6t0lXiJBmYvfuUgutt9ds795deZnlGul580oFSQRr3rzKr5kUSXVKyj3XOM8ga6KX5e+PVEAUn3U1fxwTJqR2BGW2mjLFPyo7jWCwasYs7XPlvX1ukmOfUZ9BFgOhshjMx7HqUsC0lYQ0H3ZmK781e23SClpKSrRqIX5Rn0EtRC+uiGUt6CyLnZV6QhEmpEmRxq+nRx9KrBF0TFqiklT5adYzbtlZSkeZRUtY6+zWqx5QhAlpYiSnc09P/RKEJCVaaYhf3GdQK3GJcp2sW5xZs9DrBUWYkCZldNS4oCW1pDtGXIuxu6xbwnGeQa1Fr5yIZXnslZawB0WYkCakUDDLCwatflSrOmRhTDgpsWrURCV+pHkvWbfQaw1FmJAmJKiRXbKkdkKSFfHLoiiE3VPc+lbyfNJ8Jlm20OsBRZgQcogsCpJLGo141tyjYd9DpSs/xf1Os/ZMkiRLHQGKMCGkhKw3vml1FLIWKJTk91BpWVl7JkmRpc4mRZgQMo6sN75JdxSy2vFI8nuIW1ZWn0lSZOX+KMKEkBKy0jiVIymBypJV5FeveljCWX0mSZOFziZFmBByiEZpfJMUqCyND9rXj/I9RKl7Jd9pFp9J0mSls0kRJoQcohEa30bpKFRD1O8hyrNohO+01mTpfyiqCEdeT7hSuJ4wISQKGzaYlYDsNX/HxszKR6tX169e9YLrIccnS/9DUdcTpggTQkhGWbcOuOEGs7by9dfXuzYkDlFFuKL1hAkhJE9kbc1guf6mTUaAN22Kv6YyaQwowoSQpmfePOP6FaETV/C8efWpj1x/yxZjAW/ZUlo/kh8owoSQpqe31xO6des8AazXGOzu3aXXl/rt3l2f+pD0oAgTQlIji27eIHp7TRDUDTeY13oGQa1ePf76vb1mfyM9U1IeijAhJDWy5uYNo1HGYBvpmZLyUIQJIamRNTdvEI00Btsoz5REgyJMCEmVLLl5g2i0MdhGeKYkGpwnTAhJFSadSB4+0+zDecKEkLrTSG7eRoHPNF9QhAkhqdFobt5GgM80X9AdTQghhCQM3dGEEEJIxqEIE0IIIXWCIkwIIYTUCYowIYQQUicowoQQQkidoAgTQgghdYIiTAghhNQJijAhhBBSJ1JP1qGU2g/g9wBeSfVC9WUaeH+NDO+vseH9NTZ5vb8TtNbTyx2UuggDgFJqT5TMIY0K76+x4f01Nry/xibv91cOuqMJIYSQOkERJoQQQupErUT48zW6Tr3g/TU2vL/GhvfX2OT9/kKpyZgwIYQQQsZDdzQhhJCaoZQ6QinVr5SaVu+6ZAGKcBUopVqUUr9QSu0o/p1W7zqRaCiljlZK7Sy+P1Yptdf6HstOKyD1QSnVpZR6QCn1oFLqPqXUpDz+BvMqVEqpqQD+N4D5AMaUUtPz+P3FoRbzhO8E8GcAvq21vjHVi9UYpdRsAJdprT9e77okjVLqaADf0Fr3KKVaAdwL4AgAd2qt76pv7aqj2BB8HcBRWuvZSqmLARyttd5U56olglKqC8BmABNh5uhfBmATcvA7VEp9CMBPtdbblFKbALwMoCNPv8Hi/+e3i3/LAZwL4Bbk4/s7B8AftdaPK6VuBbAfwBF5+v7ikqolXGzcJmqtFwA4USl1cprXqwPdAAaUUj9QSt2plGqpd4WSoNgIfAlAR3HXRwE8obVeCOBSpdRb6la5ZDgII0z/VtzuBrBSKfXPSqmb6letxLgCwEat9XkA9sE05Ln4HWqtP6e13lbcnA7gTeTvN/jnAFZprf8WwFYYEc7L9/dwUYDPhrGG/4D8fX+xSNsdvQjAluL7BwGclfL1as1uAIu11vMBtAI4v871SQpXpBbB+x4fAdDQE+u11v+mtf6ttesBmHucB2CBUurP61KxhPARqg8gZ79DpdQCAFMBbEPOfoM+QvVe5Oj7U0opmPbl1wCeRM6+v7ikLcIdAF4svv8VgKNTvl6t+aHW+uXi+z0AGraHauMjUnn/Hh/TWv9Oa30QplHIxfdoCdX/Q46+P6XUEQDuAPDXyOlv0BEqjRx9f9rwYQA/BPC2PH5/cUhbhA8AmFx8f1gNrldr7lFKna6UmgjgQgBP17tCKZH373GrUuqtSqkpAM4D8KN6V6haHKHKzfenlJoE4H8CWKu1/jly+ht0hOpM5Of7+7hSakVx83AAf5/H7y8OaX+ZT8BznZwO4GcpX6/WXA/gHgBPAdiltd5e5/qkRd6/x88CGAPwOIC/11r/pM71qQofocrT93c1gNkAPqmU2gHgx8jZb9BHqG5Bfr6/zwMYVEo9AhM4eDZy9v3FJdXoaKVUJ4CdAB4CsARAt+PmJBlGKbVDa71IKXUCgO8A2A7TK+8uum5JBlFKDQO4CZ5VcTeAVeDvsCEoBkZuAdAG45VZCxOLwe8vh9RiitJUAP0AHtFa70v1YiQ1lFJvg+mNb2UD0Hjwd9jY8PvLL0xbSQghhNSJhh3gJ4QQQhodijAhhBBSJyjChBBCSJ2gCBNCCCF1giJMCCGE1In/H5f1UaBMnJ8BAAAAAElFTkSuQmCC
  ">

MNIST Handwriting Dataset (Yann LeCun)
-------------------------------------

MNIST(Mixed National Institute of Standards and Technology)只是更大NIST手写数据库的子集，但它是图片识别领域的 :strong:`Hello World` . 著名的科学家，Yann LeCun, 将这个数据集存放在 `mnist <http://yann.lecun.com/exdb/mnist/>`_ . 但是因为它经常用，所以很多数据库，包括TensorFlow, 也将它囊括进去了。
数据集MNIST来自美国国家标准与技术研究所(NIST),其分为训练集和测试集，训练集有60000张图片，测试集有10000张图片,每张图片都有标签。数据集开源地址: `MNIST <http://yann.lecun.com/exdb/mnist/>`_ ，共有四部分::
  
  - train-images-idx3-ubyte.gz (训练集样本)
  - train-labels-idx1-ubyte.gz (训练集标签)
  - t10k-images-idx3-ubyte.gz (测试集样本)
  - t10k-labels-idx1-ubyte.gz (测试集标签)
  

.. code:: python

  from tensorflow.examples.tutorials.mnist import input_data
  
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  print(len(mnist.train.images))
  print(len(mnist.test.images))
  print(len(mnist.validation.images))
  print(mnist.train.labels[1,:])

the output::

  Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
  Extracting MNIST_data/train-images-idx3-ubyte.gz
  Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
  Extracting MNIST_data/train-labels-idx1-ubyte.gz
  Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
  Extracting MNIST_data/t10k-images-idx3-ubyte.gz
  Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
  Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
  55000
  10000
  5000
  [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]

<img src="data:image/png;base64,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
">  <img src="data:image/png;base64,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
">  <img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADCBJREFUeJzt3WFoXfUZx/Hfb7GCi04idkEHFYSCVmcgpLNVKxlMoTK0OpkVtzc6Chv4wvrCjY7hxvTFQJkU1hHoRhHm0OFGR1usnRbrZrVJNzcriGPYbk7B6rTTF4rx2YucrjFNzr05Oefe2zzfD4Se3OfcnIfT++N/cv4n5zgiBCCHz3S7AQCdQ+CBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRyWtMbsM2lfEDzjkbE0lYrMcIDi8PhdlaqHHjbW20/Z/v7VX8GgM6qFHjbN0nqi4jVki60vbzetgA0oeoIPyrp0WJ5t6Srphdtb7A9bnt8Ab0BqFnVwPdLer1YfkfS4PRiRIxFxEhEjCykOQD1qhr49yWdUSyfuYCfA6CDqgZ1QicO44ckvVZLNwAaVXUe/neS9tk+X9JaSavqawlAUyqN8BFxTFMn7vZL+nJEvFdnUwCaUflKu4j4j06cqQdwCuBkG5AIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4nMO/C2T7N9xPbe4uuLTTQGoH5VHhd9maRHIuKeupsB0Kwqh/SrJH3V9gu2t9qu/Ix5AJ1VJfAHJH0lIr4kaYmk62auYHuD7XHb4wttEEB9qozOf42ID4vlcUnLZ64QEWOSxiTJdlRvD0CdqozwD9sest0naZ2kF2vuCUBDqozwP5L0K0mWtD0i9tTbEoCmzDvwEfGSps7UAzjFcOENkAiBBxIh8EAiBB5IhMADiRB4IBGug+9R/f39pfV169bNWdu0aVPpey+++OLS+rPPPltaf/zxx0vrDz300Jy1Tz75pPS9aBYjPJAIgQcSIfBAIgQeSITAA4kQeCARAg8k4ohmb0jDHW9mNzw8XFrftm1baX3FihV1tlOrAwcOzFmbnJwsfe/dd99dWt+/f3+lnhKYiIiRVisxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIvw9fEMuv/zy0vqOHTtK6wMDA3W28ynHjh0rrb/77rul9WXLlpXWV65cOe+ejlu7dm1pnXn4hWGEBxIh8EAiBB5IhMADiRB4IBECDyRC4IFEmIevaGSk/E+Pt2/fXlpvcp59YmKitL5x48bS+tGjR0vrhw4dmndP6A1tjfC2B23vK5aX2P697T/avr3Z9gDUqWXgbQ9I2ibp+KNQ7tTU3TWulHSz7bMa7A9AjdoZ4Scl3SLp+PWYo5IeLZafkdTytjoAekPL3+Ej4pgk2T7+Ur+k14vldyQNznyP7Q2SNtTTIoC6VDlL/76kM4rlM2f7GRExFhEj7dxUD0DnVAn8hKSriuUhSa/V1g2ARlWZltsmaaftNZJWSHq+3pYANKXtwEfEaPHvYdvXaGqU/0FElN9ofJG6/vrrS+vnnntuo9vfs2fPnLVWvfX19ZXWW/2tPk5dlS68iYh/68SZegCnCC6tBRIh8EAiBB5IhMADiRB4IBH+PLaiph/XvHPnztL6rbfeOmftiiuuKH3vAw88UFofGhoqrePUxQgPJELggUQIPJAIgQcSIfBAIgQeSITAA4kwD1/Ryy+/XFq/8cYbF/TzWz3SecuWLXPWbrjhhtL39vf3l9axeDHCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAizMNX1OpWzq3mwi+99NLS+vr16+fdE9AKIzyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJMI8fEXPP/98ab3VveHvvffe0vpdd91VWv/4449L62Xuu+++0vrBgwdL69u3b6+8bXRXWyO87UHb+4rlL9j+l+29xdfSZlsEUJeWI7ztAUnbJB2/Tcrlku6LiLlvuQKgJ7Uzwk9KukXS8XsurZL0LdsHbd/fWGcAatcy8BFxLCLem/bSLkmjklZKWm37spnvsb3B9rjt8do6BbBgVc7S/yki/hsRk5L+LGn5zBUiYiwiRiJiZMEdAqhNlcA/Yfs825+VdK2kl2ruCUBDqkzL/VDS05I+kvTziHil3pYANKXtwEfEaPHv05IuaqqhxeKDDz4orW/atKm0/tRTT5XWd+3aNe+e2nXRRfz3LlZcaQckQuCBRAg8kAiBBxIh8EAiBB5IhD+P7ZKPPvqotN7ktBvyYoQHEiHwQCIEHkiEwAOJEHggEQIPJELggUSYh0dPOXToULdbWNQY4YFECDyQCIEHEiHwQCIEHkiEwAOJEHggEebh0VMuueSSbrewqDHCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IpGXgbZ9te5ft3bZ/a/t021ttP2f7+51oEkA92hnhb5P0YERcK+lNSesl9UXEakkX2l7eZIMA6tPy0tqI+Nm0b5dK+oaknxbf75Z0laRX628NQN3a/h3e9mpJA5L+Ken14uV3JA3Osu4G2+O2x2vpEkAt2gq87XMkbZZ0u6T3JZ1RlM6c7WdExFhEjETESF2NAli4dk7anS7pMUnfi4jDkiY0dRgvSUOSXmusOwC1ameEv0PSsKRNtvdKsqRv2n5Q0tcl7WiuPQB1auek3RZJW6a/Znu7pGsk/SQi3muoNwA1q3QDjIj4j6RHa+4FQMO40g5IhMADiRB4IBECDyRC4IFEuE01TvL222+X1o8cOVJaX7ZsWZ3toEaM8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCPPwOMlbb71VWj98+HBpnXn43sUIDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJMA+PeTt48GBpfc2aNR3qBPPFCA8kQuCBRAg8kAiBBxIh8EAiBB5IhMADibSch7d9tqRfS+qT9IGkWyT9XdI/ilXujIi/NdYhes7GjRtL68PDw3PWmKPvrnZG+NskPRgR10p6U9J3JT0SEaPFF2EHThEtAx8RP4uIJ4tvl0r6WNJXbb9ge6ttrtYDThFt/w5ve7WkAUlPSvpKRHxJ0hJJ182y7gbb47bHa+sUwIK1NTrbPkfSZklfk/RmRHxYlMYlLZ+5fkSMSRor3hv1tApgoVqO8LZPl/SYpO9FxGFJD9sest0naZ2kFxvuEUBN2jmkv0PSsKRNtvdKOiTpYUl/kfRcROxprj0AdXJEs0fcHNLns2TJkjlrmzdvLn3v1VdfXVpfsWJFpZ4SmIiIkVYrceENkAiBBxIh8EAiBB5IhMADiRB4IBECDyTCPDywODAPD+DTCDyQCIEHEiHwQCIEHkiEwAOJEHggkU7cgPKopMPTvj+3eK0X0Vs19DZ/dfd1QTsrNX7hzUkbtMfbuUCgG+itGnqbv271xSE9kAiBBxLpRuDHurDNdtFbNfQ2f13pq+O/wwPoHg7pgUQIvCTbp9k+Yntv8fXFbvfU62wP2t5XLH/B9r+m7b+l3e6v19g+2/Yu27tt/9b26d34zHX0kN72VkkrJO2IiB93bMMt2B6WdEtE3NPtXqazPSjpNxGxxvYSSY9LOkfS1oj4RRf7GpD0iKTPR8Sw7ZskDUbElm71VPQ126PNt6gHPnO2vyPp1Yh40vYWSW9I6u/0Z65jI3zxoeiLiNWSLrR90jPpumiVeuyJuEWotknqL166U1M3ObhS0s22z+pac9KkpsJ0rPh+laRv2T5o+/7utXXSo83Xq0c+c73yFOZOHtKPSnq0WN4t6aoObruVA2rxRNwumBmqUZ3Yf89I6trFJBFxLCLem/bSLk31t1LSatuXdamvmaH6hnrsMzefpzA3oZOB75f0erH8jqTBDm67lb9GxBvF8qxPxO20WULVy/vvTxHx34iYlPRndXn/TQvVP9VD+2zaU5hvV5c+c50M/PuSziiWz+zwtls5FZ6I28v77wnb59n+rKRrJb3UrUZmhKpn9lmvPIW5kztgQicOqYYkvdbBbbfyI/X+E3F7ef/9UNLTkvZL+nlEvNKNJmYJVS/ts554CnPHztLb/pykfZL+IGmtpFUzDlkxC9t7I2LU9gWSdkraI+kKTe2/ye5211tsf1vS/ToxWv5S0kbxmfu/Tk/LDUi6RtIzEfFmxza8SNg+X1Mj1hPZP7jt4jP3aVxaCyTSSyd+ADSMwAOJEHggEQIPJELggUT+B0ClD7eD5/fnAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADNdJREFUeJzt3WGIVfeZx/HfT52QOOmKsu7QmtAgSErBCGLV2aZhNtSA0hfaCAqKBLcIXQiB5kVTtglYdvNiIWWhocoEU5KQdUmXbdJgQ0wkEtnq2pl27ZoXpTGMtlklmDRqllDJ+OwLT9aJzj33zp1z7r0zz/cDg+fe59w5D4f783/u+Z+5xxEhADnM6XYDADqHwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSGRe3RuwzaV8QP3OR8TiZisxwgOzw+lWVmo78Lb32T5q+/vt/g4AndVW4G1/U9LciBiUtNT2smrbAlCHdkf4IUkvFMsHJd09sWh7l+0R2yPT6A1AxdoNfL+kd4vlDyQNTCxGxHBErIqIVdNpDkC12g38R5JuKZZvncbvAdBB7QZ1VNcO41dIGqukGwC1ance/kVJR2x/QdJ6SWurawlAXdoa4SPioq6euDsm6W8i4kKVTQGoR9tX2kXEn3TtTD2AGYCTbUAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCCR2m8XjZlnyZIlpfWhoaHS+ubNmxvWNm7cWPraiPK7i4+OjpbWt27d2rB26tSp0tdmwAgPJELggUQIPJAIgQcSIfBAIgQeSITAA4m42bzntDdg17sBTFlfX19p/emnny6tb9u2rcp2KnXo0KGGtXXr1nWwk44bjYhVzVaa8ghve57tM7YPFz/L2+sPQKe1c6XdXZL2R8R3q24GQL3a+Qy/VtI3bB+3vc82l+cCM0Q7gf+VpK9HxGpJfZI2XL+C7V22R2yPTLdBANVpZ3T+bUT8uVgekbTs+hUiYljSsMRJO6CXtDPCP2d7he25kjZKOlFxTwBq0s4I/wNJ/yLJkn4eEa9X2xKAujAPPwvdfvvtpfXdu3eX1h944IEKu+ms8fHxhrU777yz9LXvvPNO1e10Uj3z8ABmLgIPJELggUQIPJAIgQcSIfBAIlwHP0OtWbOmYW3//v2lr73jjjsq7qZ3XLlypWFthk+7VYIRHkiEwAOJEHggEQIPJELggUQIPJAIgQcSYR6+Ry1durS0/uKLLzasDQwMVN3OjDFnTuMxrNk+zTBPzwgPJELggUQIPJAIgQcSIfBAIgQeSITAA4kwD9+jHn744dJ65rn2Mnv37m1YyzDP3gwjPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kwjx8lyxatKi0vmPHjg51Mrs8//zz3W6hp7U0wtsesH2kWO6z/bLt/7C9s972AFSpaeBtL5T0jKT+4qkHdfXm81+VtNn252rsD0CFWhnhxyVtkXSxeDwk6YVi+U1Jq6pvC0Admn6Gj4iLkmT706f6Jb1bLH8g6YaLum3vkrSrmhYBVKWds/QfSbqlWL51st8REcMRsSoiGP2BHtJO4Ecl3V0sr5A0Vlk3AGrVzrTcM5J+Yftrkr4s6T+rbQlAXVoOfEQMFf+etr1OV0f5xyJivKbeZrT58+eX1g8cOFBa7+/vL63PVpcvXy6tv/TSS6X1EydOVNnOrNPWhTcR8T+6dqYewAzBpbVAIgQeSITAA4kQeCARAg8kwp/H1mTTpk2l9TVr1nSokxt9+OGHpfUnn3yytH727NnS+sqVKxvWmk2rvffee6X148ePl9ZRjhEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxJhHr5Nq1evLq0/9dRTHerkRs3m2Tds2FBaP3bsWJXtoIcwwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIszDt2nLli2l9ZtvvrlDndzooYceKq03m2dv9hXZ69atK62fO3euYe2tt94qfe2lS5dK65geRnggEQIPJELggUQIPJAIgQcSIfBAIgQeSMQRUe8G7Ho3UKMlS5Y0rI2NjZW+du7cuRV307qjR4+W1svmySXp3nvvLa0vWLBgyj196pNPPimtv/zyy6X1nTt3ltYvXLgw5Z5midGIWNVspZZGeNsDto8Uy0ts/9H24eJn8XQ7BdAZTa+0s71Q0jOSPr38ao2kf4yIPXU2BqB6rYzw45K2SLpYPF4r6Vu2f2378do6A1C5poGPiIsRMfGD0SuShiR9RdKg7buuf43tXbZHbI9U1imAaWvnLP0vI+JSRIxL+o2kZdevEBHDEbGqlZMIADqnncC/avvztudLuk/SyYp7AlCTdv48drekNyRdlrQ3In5XbUsA6tJy4CNiqPj3DUlfqquhXjI0NNSw1s159mYGBwe73UJD8+aVv+U2bdpUWr9y5UppfceOHQ1rH3/8celrM+BKOyARAg8kQuCBRAg8kAiBBxIh8EAifE01ZpT777+/tP7EE080rHEbbEZ4IBUCDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhO+lL/H22283rDW7bfGcOfxfWoexsbHS+smTJzvTyAzV9F1pe4HtV2wftP0z2zfZ3mf7qO3vd6JJANVoZRjaJumHEXGfpHOStkqaGxGDkpbaXlZngwCq0/SQPiJ+POHhYknbJf1z8figpLsl/b761gBUreUPmrYHJS2U9AdJ7xZPfyBpYJJ1d9kesT1SSZcAKtFS4G0vkvQjSTslfSTplqJ062S/IyKGI2JVRKyqqlEA09fKSbubJP1U0vci4rSkUV09jJekFZLGausOQKUcEeUr2N+W9LikE8VTP5H0HUmHJK2XtDYiLpS8vnwDM9Szzz5bWt++fXuHOpldzp8/X1pfv359aX10dLTKdmaS0VaOqFs5abdH0p6Jz9n+uaR1kv6pLOwAektbF95ExJ8kvVBxLwBqxuVgQCIEHkiEwAOJEHggEQIPJNJ0Hn7aG5il8/C33XZbaf3AgQOl9eXLl1fZTk+5fPlyw9qpU6dKX3vPPfeU1t9///22ekqgpXl4RnggEQIPJELggUQIPJAIgQcSIfBAIgQeSIR5+Jr09fWV1h999NHS+iOPPFJanzeve98wfubMmdL6Y4891rDW7HsE0Dbm4QF8FoEHEiHwQCIEHkiEwAOJEHggEQIPJMI8PDA7MA8P4LMIPJAIgQcSIfBAIgQeSITAA4kQeCCRpn9UbXuBpH+VNFfS/0raIultSe8UqzwYEf9dW4cAKtP0whvbfyfp9xHxmu09ks5K6o+I77a0AS68ATqhmgtvIuLHEfFa8XCxpE8kfcP2cdv7bHfvq1cATEnLn+FtD0paKOk1SV+PiNWS+iRtmGTdXbZHbI9U1imAaWtpdLa9SNKPJN0v6VxE/LkojUhadv36ETEsabh4LYf0QI9oOsLbvknSTyV9LyJOS3rO9grbcyVtlHSi5h4BVKSVQ/q/lbRS0t/bPizpLUnPSfovSUcj4vX62gNQJf48Fpgd+PNYAJ9F4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4l04gsoz0s6PeHxXxbP9SJ6aw+9TV3VfX2xlZVq/wKMGzZoj7Tyh/rdQG/tobep61ZfHNIDiRB4IJFuBH64C9tsFb21h96mrit9dfwzPIDu4ZAeSITAS7I9z/YZ24eLn+Xd7qnX2R6wfaRYXmL7jxP23+Ju99drbC+w/Yrtg7Z/ZvumbrznOnpIb3ufpC9LOhAR/9CxDTdhe6WkLa3eEbdTbA9I+reI+JrtPkn/LmmRpH0R8XQX+1ooab+kv4qIlba/KWkgIvZ0q6eir8lubb5HPfCem+5dmKvSsRG+eFPMjYhBSUtt33BPui5aqx67I24Rqmck9RdPPairNxv4qqTNtj/XteakcV0N08Xi8VpJ37L9a9uPd68tbZP0w4i4T9I5SVvVI++5XrkLcycP6YckvVAsH5R0dwe33cyv1OSOuF1wfaiGdG3/vSmpaxeTRMTFiLgw4alXdLW/r0gatH1Xl/q6PlTb1WPvuanchbkOnQx8v6R3i+UPJA10cNvN/DYizhbLk94Rt9MmCVUv779fRsSliBiX9Bt1ef9NCNUf1EP7bMJdmHeqS++5Tgb+I0m3FMu3dnjbzcyEO+L28v571fbnbc+XdJ+kk91q5LpQ9cw+65W7MHdyB4zq2iHVCkljHdx2Mz9Q798Rt5f3325Jb0g6JmlvRPyuG01MEqpe2mc9cRfmjp2lt/0Xko5IOiRpvaS11x2yYhK2D0fEkO0vSvqFpNcl/bWu7r/x7nbXW2x/W9LjujZa/kTSd8R77v91elpuoaR1kt6MiHMd2/AsYfsLujpivZr9jdsq3nOfxaW1QCK9dOIHQM0IPJAIgQcSIfBAIgQeSOT/ACzuZdwH/zsBAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADL1JREFUeJzt3W+IXfWdx/HPx0QlTbIhQ7NjrSEgRpZKEwjT7mSbQhYSIZ0KpVaMtvVBGgIr+MBFKMW4mrrrg31QhIIpY7JFhc2SyjZUrBpdjUaT/plJ2toFS8Ji2s7GByXFyfigpeN3H8xxM44z5945c869d/J9vyDkzP3ee8+Xm/vJ78z593NECEAOV3S7AQCdQ+CBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDySytOkV2OZUPqB5f4iINa2exAgPXB7OtfOkyoG3fdD2Sdt7q74HgM6qFHjbX5a0JCI2S7re9vp62wLQhKoj/FZJh4vlo5K2TC/a3mN7xPbIAnoDULOqgV8uaaxYviCpf3oxIoYjYiAiBhbSHIB6VQ38hKRlxfKKBbwPgA6qGtRRXdqM3yjp7Vq6AdCoqsfhj0g6bvtaSTskDdbXEoCmVBrhI2JcUzvufiLp7yPi3TqbAtCMymfaRcQfdWlPPYBFgJ1tQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCCRxm9TjdktWbKktL579+7S+tDQUKWaJJ0+fbq03t/fX1o/fLj8mqnHH398ztpbb71V+lo0ixEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxJxRLOzOTNd9Oweeuih0vrevc3N0Wm7tN7kd2Lt2rWl9fPnzze27svcaDszPTHCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiHIfvksnJydJ6k/8uTz/9dGl9cLB89u/rrruu8rrvu+++0vqjjz5a+b2Ta+Y4vO2ltn9r+1jx59PV+gPQaVXueLNB0qGI+GbdzQBoVpXf4QclfdH2z2wftM1tsoBFokrgfy5pW0R8VtKVkr4w8wm299gesT2y0AYB1KfK6PyriPhTsTwiaf3MJ0TEsKRhiZ12QC+pMsI/ZXuj7SWSviTplzX3BKAhVUb4b0v6d0mW9KOIeKnelgA0hePwXXLHHXeU1ltdF37mzJnK6x4bGyut9/X1ldY3bNhQWj9y5Ejldd90002ldcyJ6+EBfBiBBxIh8EAiBB5IhMADiRB4IBHOg++SQ4cOdbuFOV24cKG0fuzYsdL6+Pj4nLUDBw5UaQk1YYQHEiHwQCIEHkiEwAOJEHggEQIPJELggUQ4Do/alV1y3fTl2CjHCA8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiXAcHvPWarroZcuWzVlbyO21sXCM8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCMfhMW+Dg4Ol9bLppp999tm628E8tDXC2+63fbxYvtL2M7bfsL2r2fYA1Kll4G2vlvSEpOXFQ/doavL5z0n6iu2VDfYHoEbtjPCTkm6X9MH8QVslHS6WX5M0UH9bAJrQ8nf4iBiXJNsfPLRc0lixfEFS/8zX2N4jaU89LQKoS5W99BOSPrg6YsVs7xERwxExEBGM/kAPqRL4UUlbiuWNkt6urRsAjapyWO4JST+2/XlJn5L003pbAtCUtgMfEVuLv8/Z3q6pUf6fImKyod7QJXfddVdp/YEHHiitP/PMM3W2gxpVOvEmIv5Xl/bUA1gkOLUWSITAA4kQeCARAg8kQuCBRLg8NqHt27eX1h9++OHS+smTJ0vrO3funHdP6AxGeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhOPwl6EbbrihtP7888+X1t98883S+t69e+fdE3oDIzyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJMJx+EVqaGhoztqTTz5Z+tqJiYnS+t13311aP3v2bGkdvYsRHkiEwAOJEHggEQIPJELggUQIPJAIgQcS4Th8j+rr6yut79u3b87a0qXl/6yrVq2q1BMWv7ZGeNv9to8Xy5+0/Xvbx4o/a5ptEUBdWo7wtldLekLS8uKhv5X0LxGxv8nGANSvnRF+UtLtksaLnwcl7bZ9yvYjjXUGoHYtAx8R4xHx7rSHnpO0VdJnJG22vWHma2zvsT1ie6S2TgEsWJW99Cci4mJETEo6LWn9zCdExHBEDETEwII7BFCbKoF/wfYnbH9M0s2Sfl1zTwAaUuWw3D5Jr0j6s6TvRcRv6m0JQFPaDnxEbC3+fkXS3zTVUBbr1q0rrR84cKC0vnbt2jlrO3bsqNQTLn+caQckQuCBRAg8kAiBBxIh8EAiBB5IhMtjG9JqyuZXX321tN7qEtdbb711ztqJEydKX4u8GOGBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBGOwzfkwQcfLK1fc801pfVdu3aV1l9//fV59wQwwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIhyHr+i2224rrd95552l9SuuKP+/9uqrry6tl035vHLlytLXtrJp06bS+sWLF0vrZecYDAyUT0Y0NDRUWl+//iMTHbWt1TTZExMTld97sWCEBxIh8EAiBB5IhMADiRB4IBECDyRC4IFEOA5fUUQsqP7++++X1h977LHS+r333jtnrdWxatul9Va9L8RC172Q3o4cOVJa37ZtW+X3XixajvC2V9l+zvZR2z+0fZXtg7ZP2t7biSYB1KOdTfqvSvpORNws6R1JOyUtiYjNkq63Xf3UJwAd1XKTPiKmb1uukfQ1SY8WPx+VtEXSmfpbA1C3tnfa2d4sabWk30kaKx6+IKl/lufusT1ie6SWLgHUoq3A2+6T9F1JuyRNSFpWlFbM9h4RMRwRAxFRfqUEgI5qZ6fdVZJ+IOlbEXFO0qimNuMlaaOktxvrDkCt2jks9w1JmyTdb/t+Sd+X9HXb10raIWmwwf561ssvv1xaf+ONN0rrW7ZsKa23spDLRLM6dOhQt1vounZ22u2XtH/6Y7Z/JGm7pH+NiHcb6g1AzSqdeBMRf5R0uOZeADSMU2uBRAg8kAiBBxIh8EAiBB5IxE1eCilJtptdQY9asWJFaf2WW24prbe6VfSNN944Z63VrZ7fe++90vro6OiC6mVaXR47NjZWWj979mxp/dSpU5Xfe5EbbefMVkZ4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiE4/DA5YHj8AA+jMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSaTl7rO1Vkv5D0hJJ70m6XdJZSf9TPOWeiHizsQ4B1KblDTBs3y3pTES8aHu/pPOSlkfEN9taATfAADqhnhtgRMRjEfFi8eMaSX+R9EXbP7N90HalOeYBdF7bv8Pb3ixptaQXJW2LiM9KulLSF2Z57h7bI7ZHausUwIK1NTrb7pP0XUm3SnonIv5UlEYkrZ/5/IgYljRcvJZNeqBHtBzhbV8l6QeSvhUR5yQ9ZXuj7SWSviTplw33CKAm7WzSf0PSJkn32z4m6b8lPSXpF5JORsRLzbUHoE7cphq4PHCbagAfRuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJdOIGlH+QdG7azx8vHutF9FYNvc1f3X2ta+dJjd8A4yMrtEfauVC/G+itGnqbv271xSY9kAiBBxLpRuCHu7DOdtFbNfQ2f13pq+O/wwPoHjbpgUQIvCTbS23/1vax4s+nu91Tr7Pdb/t4sfxJ27+f9vmt6XZ/vcb2KtvP2T5q+4e2r+rGd66jm/S2D0r6lKRnI+KfO7biFmxvknR7uzPidortfklPR8TnbV8p6T8l9Uk6GBH/1sW+Vks6JOmvI2KT7S9L6o+I/d3qqehrtqnN96sHvnMLnYW5Lh0b4YsvxZKI2CzpetsfmZOuiwbVYzPiFqF6QtLy4qF7NDXZwOckfcX2yq41J01qKkzjxc+DknbbPmX7ke61pa9K+k5E3CzpHUk71SPfuV6ZhbmTm/RbJR0ulo9K2tLBdbfyc7WYEbcLZoZqqy59fq9J6trJJBExHhHvTnvoOU319xlJm21v6FJfM0P1NfXYd24+szA3oZOBXy5prFi+IKm/g+tu5VcRcb5YnnVG3E6bJVS9/PmdiIiLETEp6bS6/PlNC9Xv1EOf2bRZmHepS9+5TgZ+QtKyYnlFh9fdymKYEbeXP78XbH/C9sck3Szp191qZEaoeuYz65VZmDv5AYzq0ibVRklvd3DdrXxbvT8jbi9/fvskvSLpJ5K+FxG/6UYTs4Sqlz6znpiFuWN76W3/laTjkv5L0g5JgzM2WTEL28ciYqvtdZJ+LOklSX+nqc9vsrvd9Rbb/yDpEV0aLb8v6R/Fd+7/dfqw3GpJ2yW9FhHvdGzFlwnb12pqxHoh+xe3XXznPoxTa4FEemnHD4CGEXggEQIPJELggUQIPJDI/wFanVdoa3DhhwAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAAC0FJREFUeJzt3VGIHeUZxvHnaUwgJkYiTZfohSBEa9QE4sZumghb0QXFC7FKBNMblYUWvOlNFLWitCIFpSgYWUglCDXEUoulBqMlwVBNzUajTS/EWjaaNLlIIlmTi5Quby92bNZ195yzszPnnOT9/2DZOfPNnHkZ5uGbMzPnfI4IAcjhO50uAED7EHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4lcUPcGbPMoH1C/YxGxpNlC9PDA+eFgKwuVDrztzbbfs/1o2fcA0F6lAm/7TklzImKNpCtsL6u2LAB1KNvD90vaVkzvkLRuYqPtQdvDtodnURuAipUN/AJJh4vpE5J6JjZGxFBE9EZE72yKA1CtsoE/JWl+Mb1wFu8DoI3KBnWfzp7Gr5Q0Ukk1AGpV9j78HyXttn2ppFsl9VVXEoC6lOrhI2JU4xfu9kj6UUScrLIoAPUo/aRdRHyps1fqAZwDuNgGJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEql9uGh0n+uvv75h+86dOxu279mzp2H7wMDAjGtCe9DDA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAi3IdPaO3atQ3bFy5c2LB96dKlpduPHDnScF3Ua8Y9vO0LbH9ue1fxd10dhQGoXpkefoWkVyJiY9XFAKhXmc/wfZJut/2+7c22+VgAnCPKBH6vpJsj4gZJcyXdNnkB24O2h20Pz7ZAANUp0zt/HBFniulhScsmLxARQ5KGJMl2lC8PQJXK9PAv215pe46kOyR9VHFNAGpSpod/UtLvJFnS6xHxdrUlAajLjAMfEQc0fqUeXeqqq65q2L5x4+xusFxzzTUN25cvXz5tG/fhO4sn7YBECDyQCIEHEiHwQCIEHkiEwAOJ8Bz8eWjDhg0N25t9vXW2rr766mnb9u/f33Dd48ePV10OJqCHBxIh8EAiBB5IhMADiRB4IBECDyRC4IFEuA+Pyj333HPTth07dqzhulu3bq26HExADw8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiXAfHpU7dOjQtG179+5tYyWYjB4eSITAA4kQeCARAg8kQuCBRAg8kAiBBxLhPjwq1+g775999lkbK8FkLfXwtnts7y6m59r+k+2/2r6v3vIAVKlp4G0vlrRF0oJi1oOS9kXEWkl32b6oxvoAVKiVHn5M0npJo8Xrfknbiul3JPVWXxaAOjT9DB8Ro5Jk++tZCyQdLqZPSOqZvI7tQUmD1ZQIoCplrtKfkjS/mF441XtExFBE9EYEvT/QRcoEfp+kdcX0SkkjlVUDoFZlbsttkfSG7RslLZf0t2pLAlCXlgMfEf3F/4O2b9F4L/+LiBirqTacox566KFOl4BplHrwJiL+rbNX6gGcI3i0FkiEwAOJEHggEQIPJELggUT4eiwqNzbGndpuRQ8PJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJ8Lv0mDHbnS4BJbXUw9vusb27mL7M9iHbu4q/JfWWCKAqTXt424slbZG0oJj1A0m/iohNdRYGoHqt9PBjktZLGi1e90l6wPYHtp+qrTIAlWsa+IgYjYiTE2Ztl9QvabWkNbZXTF7H9qDtYdvDlVUKYNbKXKV/NyK+iogxSR9KWjZ5gYgYiojeiOiddYUAKlMm8G/aXmr7QkkDkg5UXBOAmpS5LfeEpJ2S/iPpxYj4pNqSANSl5cBHRH/xf6ek79dVELpfRHS6BJTEk3ZAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQYLhpd5dprr23YfuAA457MBj08kAiBBxIh8EAiBB5IhMADiRB4IBECDyTCfXhUbsWKFdO23X333Q3XvfLKKxu233TTTaVqwrimPbzti21vt73D9mu259nebPs924+2o0gA1WjllP5eSc9GxICko5LukTQnItZIusL2sjoLBFCdpqf0EfHChJdLJG2Q9Jvi9Q5J6yR9Wn1pAKrW8kU722skLZb0haTDxewTknqmWHbQ9rDt4UqqBFCJlgJv+xJJz0u6T9IpSfOLpoVTvUdEDEVEb0T0VlUogNlr5aLdPEmvSno4Ig5K2qfx03hJWilppLbqAFSqldty90taJekR249IeknST2xfKulWSX011odz0DPPPDNt2/Hjxxuuu3r16qrLwQStXLTbJGnTxHm2X5d0i6RfR8TJmmoDULFSD95ExJeStlVcC4Ca8WgtkAiBBxIh8EAiBB5IhMADifD12PPQ6dOnO7r9sbGxaduGhoYarjsyMlJxNZiIHh5IhMADiRB4IBECDyRC4IFECDyQCIEHEuE+/Hno6aefbth+5syZhu2PP/54w/ZFixaV3v5jjz3WcF3Uix4eSITAA4kQeCARAg8kQuCBRAg8kAiBBxJxRNS7AbveDQCQpH2tjPREDw8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiTT9PrztiyVtlTRH0mlJ6yX9U9K/ikUejIi/11YhgMo0ffDG9s8kfRoRb9neJOmIpAURsbGlDfDgDdAO1Tx4ExEvRMRbxcslkv4r6Xbb79vebJtfzQHOES1/hre9RtJiSW9JujkibpA0V9JtUyw7aHvY9nBllQKYtZZ6Z9uXSHpe0o8lHY2Ir38UbVjSssnLR8SQpKFiXU7pgS7RtIe3PU/Sq5IejoiDkl62vdL2HEl3SPqo5hoBVKSVU/r7Ja2S9IjtXZL+IellSfslvRcRb9dXHoAq8fVY4PzA12MBfBOBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJNKOH6A8JunghNffLeZ1I2orh9pmruq6Lm9lodp/AONbG7SHW/mifidQWznUNnOdqotTeiARAg8k0onAD3Vgm62itnKobeY6UlfbP8MD6BxO6YFECLwk2xfY/tz2ruLvuk7X1O1s99jeXUxfZvvQhP23pNP1dRvbF9vebnuH7ddsz+vEMdfWU3rbmyUtl/TniPhl2zbchO1Vkta3OiJuu9jukfT7iLjR9lxJf5B0iaTNEfHbDta1WNIrkr4XEats3ympJyI2daqmoq6phjbfpC445mY7CnNV2tbDFwfFnIhYI+kK298ak66D+tRlI+IWodoiaUEx60GNDzawVtJdti/qWHHSmMbDNFq87pP0gO0PbD/VubJ0r6RnI2JA0lFJ96hLjrluGYW5naf0/ZK2FdM7JK1r47ab2asmI+J2wORQ9evs/ntHUsceJomI0Yg4OWHWdo3Xt1rSGtsrOlTX5FBtUJcdczMZhbkO7Qz8AkmHi+kTknrauO1mPo6II8X0lCPittsUoerm/fduRHwVEWOSPlSH99+EUH2hLtpnE0Zhvk8dOubaGfhTkuYX0wvbvO1mzoURcbt5/71pe6ntCyUNSDrQqUImhapr9lm3jMLczh2wT2dPqVZKGmnjtpt5Ut0/Im43778nJO2UtEfSixHxSSeKmCJU3bTPumIU5rZdpbe9SNJuSX+RdKukvkmnrJiC7V0R0W/7cklvSHpb0g81vv/GOltdd7H9U0lP6Wxv+ZKkn4tj7v/afVtusaRbJL0TEUfbtuHzhO1LNd5jvZn9wG0Vx9w38WgtkEg3XfgBUDMCDyRC4IFECDyQCIEHEvkfe/LWCiwpppgAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADOBJREFUeJzt3W+IXfWdx/HPZ5MYzKQriZsdMxEqSlAiSUSn2cnWSBbagZSitVsw0Cz+SYnsog9cH9SiT1p2FXxQVoJNGYz1D6xil82aulFjlgbDxm47adJuFgwVnTTVKFSbpO6DqON3H8x1M5nMnHtz5px778z3/YKBc+/3nnO+XO6H35nzu/ccR4QA5PAnnW4AQPsQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADicytewe2+SofUL/fR8SSZi9ihAdmh6OtvKh04G1vt/2a7QfKbgNAe5UKvO2vS5oTEWslXW57ebVtAahD2RF+vaTnGsu7JV0/vmh7i+1h28PT6A1AxcoGvkfS243lDyT1ji9GxFBE9EdE/3SaA1CtsoH/UNKFjeWF09gOgDYqG9QDOnMYv1rSSCXdAKhV2Xn4f5O0z3afpA2SBqprCUBdSo3wEXFKYyfufibpryLiZJVNAahH6W/aRcQfdOZMPYAZgJNtQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJHLegbc91/Zvbe9t/K2sozEA1Stzu+hVkp6JiG9X3QyAepU5pB+Q9FXbP7e93Xbpe8wDaK8ygf+FpC9FxBpJ8yR9ZeILbG+xPWx7eLoNAqhOmdH51xFxurE8LGn5xBdExJCkIUmyHeXbA1ClMiP807ZX254j6WuSflVxTwBqUmaE/56kf5ZkSTsjYk+1LQGoy3kHPiIOa+xMPQo8/vjjhfXbb7+9sB5R/J+Q7dLr1u3ee++dsrZ169bCdT/55JOq28E4fPEGSITAA4kQeCARAg8kQuCBRAg8kIjrnsKZrd+06+npKazv2rWrsD4wMFBYnzt3dv5E4fXXXy+s33DDDYX1999/v8p2ZpMDEdHf7EWM8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCPPwHTI4OFhYX7duXWF98eLFU9aOHDlSqqdW3XnnnYX1q666qvS2r7zyysL6G2+8UXrbsxzz8ADORuCBRAg8kAiBBxIh8EAiBB5IhMADiczOH13PALt3755WvU5Fc/ySdPXVVxfWpzMPj3oxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIszDJ9TfX/yz6Z07dxbWe3t7S+/7nXfeKax//PHHpbeN5loa4W332t7XWJ5n+ye2/9P2HfW2B6BKTQNve5GkJyV9dquVuzV2dY0vSvqG7c/V2B+ACrUywo9KukXSqcbj9ZKeayy/KqnpZXUAdIem/8NHxClJsv3ZUz2S3m4sfyDpnH/obG+RtKWaFgFUpcxZ+g8lXdhYXjjZNiJiKCL6W7moHoD2KRP4A5KubyyvljRSWTcAalVmWu5JSbtsr5O0QtJ/VdsSgLqUui697T6NjfIvR8TJJq/luvRt1mye/fnnny+sX3LJJYX1Y8eOFdYPHTo0Ze3WW28tXPfkycKPE6bW0nXpS33xJiLe0Zkz9QBmCL5aCyRC4IFECDyQCIEHEiHwQCLcLnqGuuyyy6as7du3r3Ddvr6+ae37pptuKqy/8MIL09o+SuF20QDORuCBRAg8kAiBBxIh8EAiBB5IhMADiXCZ6hlq/vz5U9YWLlxY676HhoYK60888UTpdUdGRkp0hFYxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIvwefhZqdpnq6667rrB+1113FdZXrFhx3j19ptntotesWVNYP378eOl9z3L8Hh7A2Qg8kAiBBxIh8EAiBB5IhMADiRB4IBHm4XGOxYsXF9YfeeSRwvqNN944Za3Zb/X3799fWL/nnnsK68PDw4X1Way6eXjbvbb3NZaX2f6d7b2NvyXT7RRAezS94o3tRZKelNTTeOovJP1jRGyrszEA1WtlhB+VdIukU43HA5K+ZfuXth+srTMAlWsa+Ig4FREnxz31oqT1kr4gaa3tVRPXsb3F9rDttP9QAd2ozFn6/RHxx4gYlXRQ0vKJL4iIoYjob+UkAoD2KRP4l20vtb1A0qCkwxX3BKAmZS5T/V1JP5X0kaQfRsSRalsCUBfm4VG5jRs3Tlm77777CtdduXJlYf2ll14qrN98881T1j766KPCdWc4fg8P4GwEHkiEwAOJEHggEQIPJELggUSYlkNbNfvp7cGDBwvrl156aWG9r69vytp7771XuO4Mx7QcgLMReCARAg8kQuCBRAg8kAiBBxIh8EAiZX4PD5TW7FbWS5YUXwT56NGjhfXTp0+fd0+ZMMIDiRB4IBECDyRC4IFECDyQCIEHEiHwQCLMw6Nyy5Ytm7L20EMPFa47f/78wvqePXsK6ydOnCisZ8cIDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJMA+Pym3evHnK2jXXXDOtbT/66KPTWj+7piO87Ytsv2h7t+0dti+wvd32a7YfaEeTAKrRyiH9NyV9PyIGJb0raaOkORGxVtLltpfX2SCA6jQ9pI+IH4x7uETSJkn/1Hi8W9L1kn5TfWsAqtbySTvbayUtknRM0tuNpz+Q1DvJa7fYHrY9XEmXACrRUuBtL5a0VdIdkj6UdGGjtHCybUTEUET0t3JzOwDt08pJuwsk/VjSdyLiqKQDGjuMl6TVkkZq6w5ApVqZltss6VpJ99u+X9KPJP2N7T5JGyQN1NgfutCOHTsK6xs2bCi97QceKJ74OXz4cOlto7WTdtskbRv/nO2dkr4s6eGIOFlTbwAqVuqLNxHxB0nPVdwLgJrx1VogEQIPJELggUQIPJAIgQcS4eexCS1durSw3myevdlPXOfNmzdlrdk8+8MPP1xYHx0dLayjGCM8kAiBBxIh8EAiBB5IhMADiRB4IBECDyTCPHyXuvjiiwvrmzZtKr3t2267rbC+atWqwvpbb71VWC+aS3/ssccK1/30008L65geRnggEQIPJELggUQIPJAIgQcSIfBAIgQeSIR5+A654oorCuuHDh0qrC9YsKDKds4yMjJSWB8cHCysv/nmmxV2gyoxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIk3n4W1fJOlZSXMk/a+kWyS9Iemzyda7I+K/a+twljpx4kRhvdlc+IoVK6asPfvss4Xr7tmzp7D+1FNPFda5NvzM1coI/01J34+IQUnvSrpP0jMRsb7xR9iBGaJp4CPiBxHxSuPhEkmfSPqq7Z/b3m6bb+sBM0TL/8PbXitpkaRXJH0pItZImifpK5O8dovtYdvDlXUKYNpaGp1tL5a0VdJfS3o3Ik43SsOSlk98fUQMSRpqrBvVtApgupqO8LYvkPRjSd+JiKOSnra92vYcSV+T9KuaewRQkVYO6TdLulbS/bb3SvofSU9LOiTptYgoPuULoGs4ot4jbg7pgbY4EBH9zV7EF2+ARAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggkXZcgPL3ko6Oe/xnjee6Eb2VQ2/nr+q+Pt/Ki2q/AMY5O7SHW/mhfifQWzn0dv461ReH9EAiBB5IpBOBH+rAPltFb+XQ2/nrSF9t/x8eQOdwSA8kQuAl2Z5r+7e29zb+Vna6p25nu9f2vsbyMtu/G/f+Lel0f93G9kW2X7S92/YO2xd04jPX1kN629slrZD07xHxD23bcRO2r5V0S0R8u9O9jGe7V9K/RMQ62/Mk/aukxZK2R8TjHexrkaRnJP15RFxr++uSeiNiW6d6avQ12a3Nt6kLPnO2/07SbyLiFdvbJB2X1NPuz1zbRvjGh2JORKyVdLntc+5J10ED6rI74jZC9aSknsZTd2vsZgNflPQN25/rWHPSqMbCdKrxeEDSt2z/0vaDnWvrnFubb1SXfOa65S7M7TykXy/pucbybknXt3HfzfxCTe6I2wETQ7VeZ96/VyV17MskEXEqIk6Oe+pFjfX3BUlrba/qUF8TQ7VJXfaZO5+7MNehnYHvkfR2Y/kDSb1t3Hczv46I443lSe+I226ThKqb37/9EfHHiBiVdFAdfv/GheqYuug9G3cX5jvUoc9cOwP/oaQLG8sL27zvZmbCHXG7+f172fZS2wskDUo63KlGJoSqa96zbrkLczvfgAM6c0i1WtJIG/fdzPfU/XfE7eb377uSfirpZ5J+GBFHOtHEJKHqpvesK+7C3Laz9Lb/VNI+Sf8haYOkgQmHrJiE7b0Rsd725yXtkrRH0l9q7P0b7Wx33cX230p6UGdGyx9J+nvxmft/7Z6WWyTpy5JejYh327bjWcJ2n8ZGrJezf3BbxWfubHy1Fkikm078AKgZgQcSIfBAIgQeSITAA4n8H8v+YCXldTeTAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADdxJREFUeJzt3XvInPWZxvHrMiaavmpMNAm2ghCJf1RMICY10RQjHiBNlVIrqbQiZEvEFREaUIp1oWUVD1AUoSmBrIbodknVrl1s8FAjyjbZ9n3bTe1KJMsS28YmkoNJs2jM4d4/Mm4OZn4zeWaemUnu7wdefN6555nf7TBXnnmf088RIQA5nNbvBgD0DoEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJDI6XUPYJtT+YD6bYuIia2exBYeODW8186TKgfe9nLba21/v+prAOitSoG3/XVJoyJijqQptqd2ty0Adai6hZ8naVVj+RVJc48s2l5se9j2cAe9AeiyqoEfkrS5sbxD0uQjixGxLCJmRsTMTpoD0F1VA79H0tjG8lkdvA6AHqoa1BEd/ho/XdKmrnQDoFZVj8P/q6S3bH9e0nxJs7vXEoC6VNrCR8RuHdpxt07SNRGxq5tNAahH5TPtImKnDu+pB3ASYGcbkAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSKT26aJxfKedVv639sorryzWr7nmmm62c0L27dtXrD/zzDNNa++//35x3YMHD1bqCe1hCw8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiTgi6h3ArneAAXX++ecX60899VSxvmDBgm62MzBWrlxZrN9+++096uSUMxIRM1s96YS38LZPt/0n2280fi6r1h+AXqtypt00ST+NiPu63QyAelX5G362pK/a/o3t5bY5PRc4SVQJ/G8lXRcRX5I0WtJXjn2C7cW2h20Pd9oggO6psnX+Q0TsbSwPS5p67BMiYpmkZVLenXbAIKqyhV9pe7rtUZK+Jml9l3sCUJMqW/gfSvpnSZb0i4h4rbstAagLx+EranWc/b77ygcxlixZUqxv3769WF+xYkXT2oYNG4rrtjJhwoRi/YEHHijWh4aGKo/90UcfFeuXXnppsb5p06bKY5/k6jkOD+DkReCBRAg8kAiBBxIh8EAiBB5IhPPgK7rwwguL9euuu65Yf/jhh4v1559/vlgfGRkp1us0ZcqUYn3x4sWVX/vMM88s1hcuXFisP/LII5XHzoAtPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kwuWxOGGnn14+fWPNmjVNa1dddVVHY69fX77fyuzZs5vW9u7d27R2CuDyWABHI/BAIgQeSITAA4kQeCARAg8kQuCBRDgOj66bNGlS09rGjRuL65599tkdjb1o0aKmtaeffrqj1x5wHIcHcDQCDyRC4IFECDyQCIEHEiHwQCIEHkiE+9Kj66644oqmtU6Ps7eyevXqWl//ZNfWFt72ZNtvNZZH2/432/9uu/lZDgAGTsvA2x4vaYWkocZDd+vQWT1XSfqG7Xr/yQbQNe1s4Q9IWihpd+P3eZJWNZbflNTydD4Ag6Hl3/ARsVuSbH/60JCkzY3lHZImH7uO7cWSqk8wBqAWVfbS75E0trF81vFeIyKWRcTMdk7mB9A7VQI/ImluY3m6pE1d6wZAraocllsh6Ze2vyzpi5L+o7stAahL24GPiHmN/75n+3od2sr/Q0QcqKk3nKQuuuii2l5769atxfrHH39c29ingkon3kTE+zq8px7ASYJTa4FECDyQCIEHEiHwQCIEHkiEy2PRdXUeGtuzZ0+xftppbMNKeHeARAg8kAiBBxIh8EAiBB5IhMADiRB4IBGOw1c0evToYn3s2LHF+rRp04r1yy+/vFifP39+09oZZ5xRXHfXrl3F+po1a4r1Vv/v999/f7Fesn///mL9lltuKdZ37txZeewM2MIDiRB4IBECDyRC4IFECDyQCIEHEiHwQCIchy8oHfO99957i+u2Oo4+yG666aa+jX3w4MFi/YMPPuhRJ6cmtvBAIgQeSITAA4kQeCARAg8kQuCBRAg8kIgjot4B7HoH6MCMGTOK9dWrVzetTZw4saOxt23bVqy3mha5ThdccEGxPmHChB518lmt7kt/5513Nq09++yz3W5nkIxExMxWT2prC297su23GstfsP0X2280fjr75APomZZn2tkeL2mFpKHGQ1dIejAiltbZGIDua2cLf0DSQkm7G7/PlvQd27+z/VBtnQHoupaBj4jdEXHkTdBWS5onaZakObY/c3M224ttD9se7lqnADpW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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADMpJREFUeJzt3X+IXfWZx/HPx0mMNroS3Tg2/SMgBJZiEpBpOrGJRkwi1oClViy0IrglmBUR+oehbllscQddoS4UkjI4FhGsGLHaZRuNFoPR2m0nren6u1pM0qSK0WKSBbtuePaPObsZx5lzb849596bed4vGHLmPvee83C9H79nzvfccxwRApDDKb1uAED3EHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4nMaXoDtjmVD2jeoYhY2OpJjPDA7LC3nSdVDrztMdsv2P5u1XUA6K5Kgbf9VUkDEbFS0vm2l9TbFoAmVB3h10h6uFjeIWnV5KLtjbbHbY930BuAmlUN/HxJB4rlDyQNTi5GxGhEDEXEUCfNAahX1cAflXR6sXxGB+sB0EVVg7pbx3fjl0t6u5ZuADSq6jz8Y5J22V4k6QpJw/W1BKAplUb4iDisiQN3v5J0aUR8WGdTAJpR+Uy7iPiLjh+pB3AS4GAbkAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkik8ctUY/Y599xzS+vbtm2bsfbYY4+Vvvaee+6p1BPawwgPJELggUQIPJAIgQcSIfBAIgQeSITAA4kwD49PWbRoUWm91Vz60NDMNxxavXp16WvnzZtXWr/zzjtL6yjHCA8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiTAPn1CrefZHH320tF42z97Knj17Suv33ntv5XWjtRMe4W3Psb3P9s7iZ2kTjQGoX5URfpmkn0TE5rqbAdCsKn/DD0vaYPvXtsds82cBcJKoEvjfSFobESskzZX05alPsL3R9rjt8U4bBFCfKqPz7yPir8XyuKQlU58QEaOSRiXJdlRvD0CdqozwD9hebntA0lcklR92BdA3qozw35f0oCRL+llEPF1vSwCacsKBj4iXNHGkHiepSy65pLS+YsWKjtb/3nvvzVi78sorS1976NChjraNcpxpByRC4IFECDyQCIEHEiHwQCIEHkiE8+ATuv766xtdf9lXXA8ePNjotlGOER5IhMADiRB4IBECDyRC4IFECDyQCIEHEmEefhY688wzS+vLlnX27eY333yztH7fffd1tH40hxEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxJhHn4WanWp57lz53a0/k2bNpXW33rrrY7Wj+YwwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIszDn6SGh4dnrJ1ySmf/H9+3b19pff/+/R2tH73T1ifD9qDtXcXyXNv/Zvt52zc02x6AOrUMvO0Fku6XNL946GZJuyPiS5K+Zrv88ioA+kY7I/wxSddKOlz8vkbSw8Xys5KG6m8LQBNa/g0fEYclyfb/PTRf0oFi+QNJg1NfY3ujpI31tAigLlWO7hyVdHqxfMZ064iI0YgYighGf6CPVAn8bkmriuXlkt6urRsAjaoyLXe/pJ/bXi3p85L+o96WADSl7cBHxJri372212lilP+niDjWUG+pzZlT/p9m8+bNM9YGBgY62vZDDz1UWn/jjTc6Wj96p9KJNxFxUMeP1AM4SXBqLZAIgQcSIfBAIgQeSITAA4nw9dg+ddttt5XWr7rqqsrr3r59e2n99ttvr7xuSZo3b96MtYsvvrj0tWvXri2tr1+/vrS+YcOGGWsHDhyYsZYFIzyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJMI8fJ9atGhRY+t++eWXS+sfffRRaX3VqlWl9ZGRkcqv7dRNN900Y63VuQ0ZMMIDiRB4IBECDyRC4IFECDyQCIEHEiHwQCLMw89CR48eLa1v2bKltH7ppZeW1h988MHS+nnnnTdjLSJKX9upJUuWNLr+kx0jPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kwjx8j5xzzjml9YsuuqjyusfGxkrrd911V2n9mmuuqbxtqfm59jKPP/54z7Z9MmhrhLc9aHtXsfw523+yvbP4WdhsiwDq0nKEt71A0v2S5hcPfVHSP0fE1iYbA1C/dkb4Y5KulXS4+H1Y0rds/9b2zNcyAtB3WgY+Ig5HxIeTHtouaY2kL0haaXvZ1NfY3mh73PZ4bZ0C6FiVo/S/jIgjEXFM0u8kferbChExGhFDETHUcYcAalMl8E/a/qztz0haL+mlmnsC0JAq03Lfk/SMpP+W9KOIeL3elgA0pe3AR8Sa4t9nJP1dUw1lcfXVV5fWL7jggsrrbnUf9FtuuaXyuvvdc8891+sW+hpn2gGJEHggEQIPJELggUQIPJAIgQcS4euxPXL55Zc3tu477rijsXX3WqtbXR85cqRLnZycGOGBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBHm4Weh118vv0TB0qVLG92+7RlrrS5h/dprr5XWW52/8P7775fWs2OEBxIh8EAiBB5IhMADiRB4IBECDyRC4IFEmIefhZ5//vnSetPz8GVz7a3m4e++++7S+sGDByv1hAmM8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCPPwPfLuu+82tu4bb7yxsXW3o+w76dddd13pa5944om628EkLUd422fZ3m57h+2f2j7V9pjtF2x/txtNAqhHO7v035D0g4hYL+kdSV+XNBARKyWdb3tJkw0CqE/LXfqI2DLp14WSvinpX4vfd0haJekP9bcGoG5tH7SzvVLSAkn7JR0oHv5A0uA0z91oe9z2eC1dAqhFW4G3fbakH0q6QdJRSacXpTOmW0dEjEbEUEQM1dUogM61c9DuVEnbJH0nIvZK2q2J3XhJWi7p7ca6A1Art/q6ou1NkkYk7Ske+rGkb0v6haQrJA1HxIclry/fQFKLFy8urb/66qul9dNOO63Odj7h448/Lq2/8sorpfWRkZEZa9u2bavUE1ra3c4edTsH7bZK2jr5Mds/k7RO0r+UhR1Af6l04k1E/EXSwzX3AqBhnFoLJELggUQIPJAIgQcSIfBAIi3n4TveAPPwlaxbt660fuutt85Yu+yyy0pf++KLL5bWy+bRJemRRx4praMn2pqHZ4QHEiHwQCIEHkiEwAOJEHggEQIPJELggUSYhwdmB+bhAXwSgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyTS8u6xts+S9JCkAUn/JelaSW9K+mPxlJsj4j8b6xBAbVpeAMP2P0j6Q0Q8ZXurpD9Lmh8Rm9vaABfAALqhngtgRMSWiHiq+HWhpP+RtMH2r22P2a50j3kA3df23/C2V0paIOkpSWsjYoWkuZK+PM1zN9oetz1eW6cAOtbW6Gz7bEk/lHS1pHci4q9FaVzSkqnPj4hRSaPFa9mlB/pEyxHe9qmStkn6TkTslfSA7eW2ByR9RdKehnsEUJN2dun/XtKFkv7R9k5JL0t6QNKLkl6IiKebaw9AnbhMNTA7cJlqAJ9E4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4l04wKUhyTtnfT73xaP9SN6q4beTlzdfS1u50mNXwDjUxu0x9v5on4v0Fs19HbietUXu/RAIgQeSKQXgR/twTbbRW/V0NuJ60lfXf8bHkDvsEsPJELgJdmeY3uf7Z3Fz9Je99TvbA/a3lUsf872nya9fwt73V+/sX2W7e22d9j+qe1Te/GZ6+ouve0xSZ+X9O8RcUfXNtyC7QslXdvuHXG7xfagpEciYrXtuZIelXS2pLGIuK+HfS2Q9BNJ50bEhba/KmkwIrb2qqeir+lubb5VffCZ6/QuzHXp2ghffCgGImKlpPNtf+qedD00rD67I24RqvslzS8eulkTNxv4kqSv2T6zZ81JxzQRpsPF78OSvmX7t7ZHeteWviHpBxGxXtI7kr6uPvnM9ctdmLu5S79G0sPF8g5Jq7q47VZ+oxZ3xO2BqaFao+Pv37OSenYySUQcjogPJz20XRP9fUHSStvLetTX1FB9U332mTuRuzA3oZuBny/pQLH8gaTBLm67ld9HxJ+L5WnviNtt04Sqn9+/X0bEkYg4Jul36vH7NylU+9VH79mkuzDfoB595roZ+KOSTi+Wz+jytls5Ge6I28/v35O2P2v7M5LWS3qpV41MCVXfvGf9chfmbr4Bu3V8l2q5pLe7uO1Wvq/+vyNuP79/35P0jKRfSfpRRLzeiyamCVU/vWd9cRfmrh2lt/03knZJ+oWkKyQNT9llxTRs74yINbYXS/q5pKclXaSJ9+9Yb7vrL7Y3SRrR8dHyx5K+LT5z/6/b03ILJK2T9GxEvNO1Dc8SthdpYsR6MvsHt1185j6JU2uBRPrpwA+AhhF4IBECDyRC4IFECDyQyP8CpGxKCebgXu8AAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADPlJREFUeJzt3W+IXfWdx/HPx0TRJlmT6HSIfRAR8qSoEZl2M1uDEVvRGiTWgIGmojaMuuITEbtFBVt3fbBgWQk2ZSBbNbARu27XitEkShNDa/9Mmv6J0dJVYtrZiFRr/ihWOn73wVw3k8ncc2/uPefeO/N9v2Dg3Pu9956v1/vJ78z5nTs/R4QA5HBKtxsA0DkEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIrOr3oFtLuUDqvfniOhr9CBGeGBmeLOZB7UceNsbbb9s+95WXwNAZ7UUeNtfkTQrIgYlnWd7SbltAahCqyP8CklP1ra3SbpkYtH2kO0R2yNt9AagZK0Gfo6k0dr2u5L6JxYjYjgiBiJioJ3mAJSr1cAflXRGbXtuG68DoINaDepuHTuMXyppfyndAKhUq/Pw/y1pl+1zJF0laVl5LQGoSksjfEQc1viJu59JuiwiDpXZFIBqtHylXUT8RcfO1AOYBjjZBiRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCInHXjbs20fsL2j9nNBFY0BKF8ry0VfKGlzRHyj7GYAVKuVQ/plklba/oXtjbZbXmMeQGe1EvhfSvpiRHxe0qmSvjz5AbaHbI/YHmm3QQDlaWV0/m1E/LW2PSJpyeQHRMSwpGFJsh2ttwegTK2M8JtsL7U9S9IqSb8puScAFWllhP+2pP+QZEk/iogXym0JQFVOOvARsVfjZ+rRhrlz5xbW169fX1i/4YYb6tb27dtX+NzLL7+8sP72228X1jF9ceENkAiBBxIh8EAiBB5IhMADiRB4IBFHVHshHFfaTa2/v7+wPjo6Wtm+X3311cL6nj17CuvPPvtsW69fpQMHDtStvffeex3spON2R8RAowcxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIszDd8nVV19dWH/66ac71MmJbBfWq/7MFGnU2969e+vWDh482Na+16xZU1jv8jw/8/AAjkfggUQIPJAIgQcSIfBAIgQeSITAA4kwD98ls2cX/4Xw5cuXF9Yff/zxurVFixa11NMnpvM8fJW9Nfqe/wUXdHUhZebhARyPwAOJEHggEQIPJELggUQIPJAIgQcSYR5+mpo/f37d2rp16wqf29fXV3Y7x1m9enXd2uLFi9t67W7Ow59//vmF9ddee62yfTehvHl42/22d9W2T7X9jO2f2L653S4BdE7DwNteIOkxSXNqd92h8X9NviBpte15FfYHoETNjPBjkq6XdLh2e4WkJ2vbL0lqeBgBoDcUX9AtKSIOS8f97jRH0icLn70r6YRF0mwPSRoqp0UAZWnlLP1RSWfUtudO9RoRMRwRA82cRADQOa0EfrekS2rbSyXtL60bAJVqeEg/hcckbbG9XNJnJf283JYAVKWleXjb52h8lN8aEYcaPJZ5eJTmkUceKazfeuutle37gQceKKzff//9le27CU3Nw7cywisi/lfHztQDmCa4tBZIhMADiRB4IBECDyRC4IFE+HosppWxsbHCejuf53379hXWBwcHC+vvv/9+y/suAX+mGsDxCDyQCIEHEiHwQCIEHkiEwAOJEHggkZa+LQe0qtFS1jfeeGNl+37nnXcK66Ojo4X1Rkt8TweM8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCN+HR0dt3769sH7ZZZcV1ttZLnrt2rWFz33iiScK6z2O78MDOB6BBxIh8EAiBB5IhMADiRB4IBECDyQy/b/gi47r6+srrN933311a8uXL29r3zt37iysb9mypW7tqaeeamvfM0FTI7ztftu7atufsf0n2ztqP8X/9wH0jIYjvO0Fkh6TNKd2199L+peI2FBlYwDK18wIPybpekmHa7eXSVpn+1e2H6ysMwClaxj4iDgcEYcm3PWcpBWSPidp0PaFk59je8j2iO2R0joF0LZWztL/NCKORMSYpD2Slkx+QEQMR8RAMxfzA+icVgK/1fYi25+SdIWkvSX3BKAirUzLfUvSjyV9JOl7EfH7clsCUBW+D48TnHvuuYX1119/vbJ9f/DBB4X1efPmVbbvaY7vwwM4HoEHEiHwQCIEHkiEwAOJEHggEb4eOwPNnz+/sH7t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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADOpJREFUeJzt3X+oXPWZx/HPZ5Mo+bV6xewlDVgRIlpoAvGmm7QJRKhCSsFSqxbSv7QGuiBi/KMWy5IGG7BoEWqaGMgWFbeLlc3adau5WhKM1m57k2g3ayhdxPwy/lGMSdVQMTz9446bm5vMd+bOPWdmbp73Cy45M89873kY5+P33HPOnOOIEIAc/q7XDQDoHgIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCCR6XWvwDan8gH1+3NEzGv1ImZ44MJwsJ0XdRx429tsv2b7+53+DgDd1VHgbX9d0rSIWC7pKtsLq20LQB06neFXSXq6sTwsacXYou21tkdsj0yiNwAV6zTwsyUdbSy/J2lwbDEitkbEUEQMTaY5ANXqNPAfSJrZWJ4zid8DoIs6DeoendmMXyzp7Uq6AVCrTo/D/4ek3bY/I2m1pGXVtQSgLh3N8BFxUqM77n4r6fqIOFFlUwDq0fGZdhFxXGf21AOYAtjZBiRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSqf0y1cjnuuuua1pbv359ceypU6eK9VtvvbWTltDADA8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiXAcHhM2f/78Yn14eLhpbWBgoDj20Ucf7agntIcZHkiEwAOJEHggEQIPJELggUQIPJAIgQcScUTUuwK73hWgcgsWLCjWn3322WJ9yZIlTWuHDx8ujr3yyiuL9bo/r1PYnogYavWiCc/wtqfbPmR7V+Pn8531B6DbOjnTbpGkn0fEd6tuBkC9Ovkbfpmkr9r+ne1ttjk9F5giOgn87yV9OSK+IGmGpK+Mf4HttbZHbI9MtkEA1elkdv5DRPy1sTwiaeH4F0TEVklbJXbaAf2kkxn+SduLbU+T9DVJb1TcE4CadDLDb5D0r5Is6ZcR8VK1LQGoy4QDHxH7NbqnHlPUnDlzivUHHnigWG91nH779u1Na++8805xLMfZ68WZdkAiBB5IhMADiRB4IBECDyRC4IFEOA8+oWuuuaZYv/nmm4v1ffv2Fevr1q1rWrv44ouLY1EvZnggEQIPJELggUQIPJAIgQcSIfBAIgQeSITLVCf05ptvFuutjtOXLkMtSa+//vqEe8Kk1XOZagBTF4EHEiHwQCIEHkiEwAOJEHggEQIPJML34S9A119/fbG+cOE5Nws6y6ZNm4r1/fv3T7gn9AdmeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhO/DX4B27NhRrK9YsaJYv/baa4v1Q4cOTbgn1K6678PbHrS9u7E8w/Z/2n7V9u2T7RJA97QMvO0BSY9Lmt146i6N/t/kS5K+YXtujf0BqFA7M/xpSbdJOtl4vErS043llyW13IwA0B9anksfESclyfanT82WdLSx/J6kwfFjbK+VtLaaFgFUpZO99B9ImtlYnnO+3xERWyNiqJ2dCAC6p5PA75H06W7exZLerqwbALXq5Ouxj0v6le2Vkj4n6b+rbQlAXdoOfESsavx70PYNGp3l/zkiTtfUGwpmzZrVtLZo0aLi2GeeeaZY5zj7haujC2BExDs6s6cewBTBqbVAIgQeSITAA4kQeCARAg8kwmWqp6iVK1c2rQ0OnnO281kefvjhqtvBFMEMDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJcBx+irrlllua1o4dO1Yce/jw4arbwRTBDA8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiXAcfopavXp109rRo0eb1iTp+PHjVbeDKYIZHkiEwAOJEHggEQIPJELggUQIPJAIgQcS4Tg8Juzqq68u1m+66aamteHh4eLYAwcOFOsff/xxsY6ytmZ424O2dzeWF9g+YntX42devS0CqErLGd72gKTHJc1uPPWPkn4YEZvrbAxA9dqZ4U9Luk3SycbjZZK+bXuv7Y21dQagci0DHxEnI+LEmKeel7RK0lJJy20vGj/G9lrbI7ZHKusUwKR1spf+NxHxl4g4LWmfpIXjXxARWyNiKCKGJt0hgMp0EvgdtufbniXpRkn7K+4JQE06OSz3A0k7JX0saUtE/LHalgDUpe3AR8Sqxr87JV1TV0MYtXjx4mL98ssvb1p74oknimOnTy//Z3/ooYeK9TvvvLNYnzlzZtPagw8+WBz71FNPFet33HFHsc5x+jLOtAMSIfBAIgQeSITAA4kQeCARAg8kwtdj+9TcuXOL9RkzZjStnThxomlNkjZt2lSstzrs9sorrxTrpUNvd999d3HsmjVrivUtW7YU66+++mqxnh0zPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kwnH4PnXs2LFi/cMPP2xau+eee4pj580rX2j4scceK9bvvffeYv2jjz5qWlu/fn1xbEQU65988kmxjjJmeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IxK2Oe056BXa9K0jq6NGjTWvz588vjn3rrbeK9UWLzrl72FlKx9klaWBgoGlt7969xbFHjhwp1leuXFmsJ7annTs9McMDiRB4IBECDyRC4IFECDyQCIEHEiHwQCJ8H36KKt0S+r777iuOPXDgQLF+6tSpYr10q2pJeu6555rWrrjiiuLYVtetx+S0nOFtX2L7edvDtrfbvsj2Ntuv2f5+N5oEUI12NunXSPpxRNwo6V1J35Q0LSKWS7rK9sI6GwRQnZab9BHx0zEP50n6lqRHGo+HJa2Q9KfqWwNQtbZ32tleLmlA0mFJn57I/Z6kwfO8dq3tEdsjlXQJoBJtBd72ZZJ+Iul2SR9ImtkozTnf74iIrREx1M7J/AC6p52ddhdJ+oWk70XEQUl7NLoZL0mLJb1dW3cAKtXy67G2vyNpo6Q3Gk/9TNI6Sb+WtFrSsohoen9ivh5bj9JXUF944YXi2KVLlxbrO3funNT40q2uN27cWBx7//33F+toqq2vx7az026zpM1jn7P9S0k3SPpRKewA+ktHJ95ExHFJT1fcC4CacWotkAiBBxIh8EAiBB5IhMADiXCZ6gvQpZdeWqwPDw8X60NDkztB8pFHHmla27BhQ3Hs+++/P6l1J8ZlqgGcjcADiRB4IBECDyRC4IFECDyQCIEHEuE4PHBh4Dg8gLMReCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIt7x5r+xJJ/yZpmqQPJd0m6f8kvdV4yV0R8T+1dQigMi0vgGH7nyT9KSJetL1Z0jFJsyPiu22tgAtgAN1QzQUwIuKnEfFi4+E8SZ9I+qrt39neZruje8wD6L62/4a3vVzSgKQXJX05Ir4gaYakr5zntWttj9geqaxTAJPW1uxs+zJJP5F0s6R3I+KvjdKIpIXjXx8RWyVtbYxlkx7oEy1neNsXSfqFpO9FxEFJT9pebHuapK9JeqPmHgFUpJ1N+jskLZF0v+1dkv5X0pOSXpf0WkS8VF97AKrEZaqBCwOXqQZwNgIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IpBsXoPyzpINjHl/eeK4f0Vtn6G3iqu7rs+28qPYLYJyzQnuknS/q9wK9dYbeJq5XfbFJDyRC4IFEehH4rT1YZ7vorTP0NnE96avrf8MD6B026YFECLwk29NtH7K9q/Hz+V731O9sD9re3VheYPvImPdvXq/76ze2L7H9vO1h29ttX9SLz1xXN+ltb5P0OUn/FREPdG3FLdheIum2du+I2y22ByU9ExErbc+Q9O+SLpO0LSL+pYd9DUj6uaR/iIgltr8uaTAiNveqp0Zf57u1+Wb1wWdusndhrkrXZvjGh2JaRCyXdJXtc+5J10PL1Gd3xG2E6nFJsxtP3aXRmw18SdI3bM/tWXPSaY2G6WTj8TJJ37a91/bG3rWlNZJ+HBE3SnpX0jfVJ5+5frkLczc36VdJerqxPCxpRRfX3crv1eKOuD0wPlSrdOb9e1lSz04miYiTEXFizFPPa7S/pZKW217Uo77Gh+pb6rPP3ETuwlyHbgZ+tqSjjeX3JA12cd2t/CEijjWWz3tH3G47T6j6+f37TUT8JSJOS9qnHr9/Y0J1WH30no25C/Pt6tFnrpuB/0DSzMbynC6vu5WpcEfcfn7/dtieb3uWpBsl7e9VI+NC1TfvWb/chbmbb8AendmkWizp7S6uu5UN6v874vbz+/cDSTsl/VbSloj4Yy+aOE+o+uk964u7MHdtL73tv5e0W9KvJa2WtGzcJivOw/auiFhl+7OSfiXpJUlf1Oj7d7q33fUX29+RtFFnZsufSVonPnP/r9uH5QYk3SDp5Yh4t2srvkDY/oxGZ6wd2T+47eIzdzZOrQUS6acdPwBqRuCBRAg8kAiBBxIh8EAifwPwgG8H+EJRYwAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADFpJREFUeJzt3W+IHPUdx/HPpxeDJqbhQq+H+iAaCJRgDcRre1GLKUYhGiG2gor2iZGAFZ8UoYp9UmNFChah0MhhKkGo/6qW1CqJisHQGvUS/7R9UFqKtkkTUJScKTFi8u2Dmzbn5W52Mzezu5fv+wVHZvc7O/Nl2E9+szOzO44IAcjhS91uAEDnEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4nMaXoFtrmUD2jehxEx0GomRnjg1PB+OzNVDrztzbZfs/3jqssA0FmVAm/7u5L6ImKlpCW2l9bbFoAmVB3hV0l6spjeLumSiUXbG2yP2h6dQW8AalY18PMl7SumP5I0OLEYESMRMRQRQzNpDkC9qgb+kKQziukzZ7AcAB1UNai7dXw3frmk92rpBkCjqp6H/62knbbPlrRG0nB9LQFoSqURPiLGNH7gbpek70TEwTqbAtCMylfaRcTHOn6kHsAswME2IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQSOWbSQLTOffcc6etvfzyy6WvPe+880rrN9xwQ2n9iSeeKK1nd9IjvO05tv9pe0fx9/UmGgNQvyoj/AWSHouIH9XdDIBmVfkMPyxpre03bG+2zccCYJaoEvg3Ja2OiG9KOk3SlZNnsL3B9qjt0Zk2CKA+VUbndyPiSDE9Kmnp5BkiYkTSiCTZjurtAahTlRH+UdvLbfdJWifpnZp7AtCQKiP8PZJ+LcmStkbES/W2BKApjmh2j5td+nyefvrpaWvr1q2b0bI/++yz0vr27dunrT3++OOlrx0cHCytP/jgg6X1LtsdEUOtZuJKOyARAg8kQuCBRAg8kAiBBxIh8EAiXAeP2l1zzTXT1mZ6Gnju3Lml9bVr105bO3bsWOlrt27dWqmn2YQRHkiEwAOJEHggEQIPJELggUQIPJAIgQcS4Tw8Ttqtt97a2LK3bNlSWr/tttsqL/vIkSOl9Vbn6U8FjPBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAjn4XGCq666qrS+fv36xtb9+eefl9YPHz7c2LozYIQHEiHwQCIEHkiEwAOJEHggEQIPJELggUQ4D5/QwoULS+v3339/aX3ZsmV1toMOamuEtz1oe2cxfZrt39n+g+2bm20PQJ1aBt52v6QtkuYXT92u8ZvPXyzpWtsLGuwPQI3aGeGPSrpO0ljxeJWkJ4vpVyUN1d8WgCa0/AwfEWOSZPt/T82XtK+Y/kjS4OTX2N4gaUM9LQKoS5Wj9IcknVFMnznVMiJiJCKGIoLRH+ghVQK/W9IlxfRySe/V1g2ARlU5LbdF0vO2vy1pmaTX620JQFNc5X7dts/W+Ci/LSIOtph3ZjcEP0VdeumlpfV58+aV1svOlQ8OnnBY5QvmzCn/f76/v7+03sqE4z0naPV99+Hh4dL6nj17KvWUwO52PkJXuvAmIv6t40fqAcwSXFoLJELggUQIPJAIgQcSIfBAInw9tksOHTpUWr/ooosqL3vRokWl9b6+vsrLbsf+/funrd17772lr+W0W7MY4YFECDyQCIEHEiHwQCIEHkiEwAOJEHggkUpfjz2pFfD12I47cOBAaX1gYKDR9S9evHja2t69extdd2JtfT2WER5IhMADiRB4IBECDyRC4IFECDyQCIEHEuH78Dhpb7/9dmmdc+29ixEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxLhPPwstWLFimlrCxYsKH1t2e2c26lv3LixtI7e1dYIb3vQ9s5i+hzbe23vKP6a/TUFALVpOcLb7pe0RdL84qlvSfppRGxqsjEA9WtnhD8q6TpJY8XjYUm32N5j+77GOgNQu5aBj4ixiDg44akXJK2S9A1JK21fMPk1tjfYHrU9WlunAGasylH6P0bEJxFxVNJbkpZOniEiRiJiqJ0f1QPQOVUCv832WbbnSbpC0p9r7glAQ6qclvuJpFckfSbpoYj4a70tAWhK24GPiFXFv69I+lpTDaE9d9xxx7S1008/vfS1re5F8Prrr5fWt23bVlpH7+JKOyARAg8kQuCBRAg8kAiBBxIh8EAifD22R51//vml9auvvrqxdT/wwAOl9cOHDze2bjSLER5IhMADiRB4IBECDyRC4IFECDyQCIEHEuE8fI8aGCj/MeB58+ZVXvbDDz9cWn/mmWcqLxu9jREeSITAA4kQeCARAg8kQuCBRAg8kAiBBxLhPHyXzJlTvunvvPPOxta9a9eu0vqxY8caWze6ixEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxLhPHyXDA8Pl9ZXr15dedmffvppaf25556rvGzMbi1HeNsLbb9ge7vtZ23Ptb3Z9mu2f9yJJgHUo51d+hsl/TwirpB0QNL1kvoiYqWkJbaXNtkggPq03KWPiF9OeDgg6SZJDxaPt0u6RNLf6m8NQN3aPmhne6Wkfkn/krSvePojSYNTzLvB9qjt0Vq6BFCLtgJve5GkX0i6WdIhSWcUpTOnWkZEjETEUEQM1dUogJlr56DdXElPSborIt6XtFvju/GStFzSe411B6BW7ZyWWy9phaS7bd8t6RFJ37d9tqQ1ksrPL2FKa9asaWzZrb7++sEHHzS2bvS2dg7abZK0aeJztrdKulzSzyLiYEO9AahZpQtvIuJjSU/W3AuAhnFpLZAIgQcSIfBAIgQeSITAA4nw9dguueyyyxpb9saNGxtbNmY3RnggEQIPJELggUQIPJAIgQcSIfBAIgQeSITz8A1p9X33Cy+8cEbL37dv37S1PXv2zGjZOHUxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIpyHb8jOnTtL661+O37JkiWl9SuvvHLa2tjYWOlrkRcjPJAIgQcSIfBAIgQeSITAA4kQeCARAg8k4ogon8FeKOlxSX2S/iPpOkl/l/SPYpbbI+JPJa8vXwGAOuyOiKFWM7UT+B9I+ltEvGh7k6T9kuZHxI/a6YLAAx3RVuBb7tJHxC8j4sXi4YCkzyWttf2G7c22uVoPmCXa/gxve6WkfkkvSlodEd+UdJqkE67xtL3B9qjt0do6BTBjbY3OthdJ+oWk70k6EBFHitKopKWT54+IEUkjxWvZpQd6RMsR3vZcSU9Juisi3pf0qO3ltvskrZP0TsM9AqhJO7v06yWtkHS37R2S/iLpUUlvS3otIl5qrj0AdWp5lH7GK2CXHuiEeo7SAzh1EHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAinfgByg8lvT/h8VeK53oRvVVDbyev7r4WtzNT4z+AccIK7dF2vqjfDfRWDb2dvG71xS49kAiBBxLpRuBHurDOdtFbNfR28rrSV8c/wwPoHnbpgUQIvCTbc2z/0/aO4u/r3e6p19ketL2zmD7H9t4J22+g2/31GtsLbb9ge7vtZ23P7cZ7rqO79LY3S1om6fcRcW/HVtyC7RWSrmv3jridYntQ0m8i4tu2T5P0jKRFkjZHxK+62Fe/pMckfTUiVtj+rqTBiNjUrZ6Kvqa6tfkm9cB7bqZ3Ya5Lx0b44k3RFxErJS2xfcI96bpoWD12R9wiVFskzS+eul3jNxu4WNK1thd0rTnpqMbDNFY8HpZ0i+09tu/rXlu6UdLPI+IKSQckXa8eec/1yl2YO7lLv0rSk8X0dkmXdHDdrbypFnfE7YLJoVql49vvVUldu5gkIsYi4uCEp17QeH/fkLTS9gVd6mtyqG5Sj73nTuYuzE3oZODnS9pXTH8kabCD627l3YjYX0xPeUfcTpsiVL28/f4YEZ9ExFFJb6nL229CqP6lHtpmE+7CfLO69J7rZOAPSTqjmD6zw+tuZTbcEbeXt98222fZnifpCkl/7lYjk0LVM9usV+7C3MkNsFvHd6mWS3qvg+tu5R71/h1xe3n7/UTSK5J2SXooIv7ajSamCFUvbbOeuAtzx47S2/6ypJ2SXpa0RtLwpF1WTMH2johYZXuxpOclvSTpIo1vv6Pd7a632L5V0n06Plo+IumH4j33f50+Ldcv6XJJr0bEgY6t+BRh+2yNj1jbsr9x28V77ou4tBZIpJcO/ABoGIEHEiHwQCIEHkiEwAOJ/Bf4oh7ljN4OWwAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADR9JREFUeJzt3X+o1XWex/HXay0htQ1jXRmnEqQfIJgmarreKYuZfkyDTDZxh8b+qBVhlvpnKCZZKWZqCzaYgiEVwUSKdXGWncXYMS25kq3NjtexZlv6tWypk3MhaVITmm3lvX/c03rVez/neO73/PC+nw+48D3nfb7nvDmcl5+v318fR4QA5PBnnW4AQPsQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiVzQ6g+wzal8QOsdiYgp9V7ECA+MDQcaeVHTgbe9wfYbtlc3+x4A2qupwNteJmlcRCySNMP2VdW2BaAVmh3hl0jaUlveIalnaNH2Stv9tvtH0RuAijUb+ImSPq4tfypp6tBiRKyPiHkRMW80zQGoVrOB/1zSRbXlSaN4HwBt1GxQ9+nUZvxsSR9V0g2Almr2OPy/SNpte5qk2yUtrK4lAK3S1AgfEcc0uOPu15JuioijVTYFoDWaPtMuIv6oU3vqAZwH2NkGJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIucceNsX2D5oe1ftb1YrGgNQvWami75W0uaI+HHVzQBorWY26RdK+o7t39jeYLvpOeYBtFczgd8r6ZsRsUDShZK+feYLbK+03W+7f7QNAqhOM6Pz7yLiT7XlfklXnfmCiFgvab0k2Y7m2wNQpWZG+Bdsz7Y9TtJ3Jb1VcU8AWqSZEf6nkv5BkiVtjYhXq20JQKucc+Aj4m0N7qkHcJ7hxBsgEQIPJELggUQIPJAIgQcSIfBAIpwH3yIXX3xxsf7II48U63fffXexfvz48RFrhw4dKq67bdu2Yn3Tpk3F+hdffFGso3sxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIo5o7Q1pxuodb6688spi/fnnny/WFy9eXKx/+OGHxfrBgweL9ZL58+cX65988kmx/vjjjxfrGzduPOeeMGr7ImJevRcxwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIlwPXzB9+vQRa9u3by+ue9lllxXrK1asKNZffPHFYv3LL78s1ktuuummYv3ee+8t1tetW1esL1u2bMTaPffcU1y3dJ0/Ro8RHkiEwAOJEHggEQIPJELggUQIPJAIgQcSSX09/KRJk4r1l156acTaDTfcUFx3+fLlxfrmzZuL9W5W7zj+zp07R6x99tlnxXWfffbZYv2pp54q1kvnJzzwwAPFdR999NFivbe3t1jv6+sr1lusuuvhbU+1vbu2fKHtl2z/m+37R9slgPapG3jbkyVtkjSx9tSDGvzXZLGk79kuT7ECoGs0MsKflNQr6Vjt8RJJW2rLr0mquxkBoDvUPZc+Io5Jku2vnpoo6ePa8qeSpp65ju2VklZW0yKAqjSzl/5zSRfVlicN9x4RsT4i5jWyEwFA+zQT+H2SemrLsyV9VFk3AFqqmctjN0n6le1vSJop6d+rbQlAqzR1HN72NA2O8tsj4mid13bsOPyQ/Q7Deuyxx4r1VatWjVirN7/7M888U6yPZaV79peO0UvS5ZdfPqrPfvPNN0eszZkzp7ju3r17i/Wbb765WD9x4kSx3mINHYdv6gYYEXFYp/bUAzhPcGotkAiBBxIh8EAiBB5IhMADiYzpy2NLt5mWpHfffbdYf+6550asPfTQQ031lN20adOK9euuu65Yr3c4tDQN94EDB4rr3nrrrcX6+++/X6x3GNNFAzgdgQcSIfBAIgQeSITAA4kQeCARAg8kMqani37iiSeK9YGBgWJ99erVVbYDSYcPHx5V/eqrry7WS8fh16xZU1y3y4+zV4IRHkiEwAOJEHggEQIPJELggUQIPJAIgQcSGdPH4etNB33FFVcU60uXLh2xtmULN+1thQULFhTrTz/9dLFeug125luHf4URHkiEwAOJEHggEQIPJELggUQIPJAIgQcSGdP3pa93THfixInFel9fX5XtQNK4ceOK9XrfeU9PT7E+d+7cEWulqaTHgOruS297qu3dteWv2/697V21vymj7RRAe9Q90872ZEmbJH01HF4v6e8iYm0rGwNQvUZG+JOSeiUdqz1eKGmF7d/afrJlnQGoXN3AR8SxiDg65KltkpZImi9pke1rz1zH9krb/bb7K+sUwKg1s5d+T0Qcj4iTkvZLuurMF0TE+oiY18hOBADt00zgt9v+mu0Jkm6R9HbFPQFokWYuj/2JpD5J/yNpXUS8V21LAFplTB+HR/e54447ivWtW7cW6y+//PKo3n8MY354AKcj8EAiBB5IhMADiRB4IBECDyTCYTlUbsaMGSPW9uzZU1z3xIkTxfr1119frB85cqRYH8M4LAfgdAQeSITAA4kQeCARAg8kQuCBRAg8kMiYni4andHb2ztibcqU8k2O77vvvmI98XH2SjDCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiXA+PypV+U++8805x3ZkzZ1bdThZcDw/gdAQeSITAA4kQeCARAg8kQuCBRAg8kAjXw+Oc3XbbbcV66Tj8tm3bqm4H56DuCG/7EtvbbO+w/Uvb421vsP2G7dXtaBJANRrZpP+BpJ9FxC2SBiR9X9K4iFgkaYbtq1rZIIDq1N2kj4g1Qx5OkbRc0rO1xzsk9Uj6oPrWAFSt4Z12thdJmizpkKSPa09/KmnqMK9dabvfdn8lXQKoREOBt32ppJ9Lul/S55IuqpUmDfceEbE+IuY1cjI/gPZpZKfdeEm/kLQqIg5I2qfBzXhJmi3po5Z1B6BSdS+Ptf1DSU9Keqv21EZJP5K0U9LtkhZGxNHC+lweO8bUm/J51qxZI9auueaa4rqHDx9uqic0dnlsIzvt1kpaO/Q521slfUvS35fCDqC7NHXiTUT8UdKWinsB0GKcWgskQuCBRAg8kAiBBxIh8EAiXB6Ls/T09BTrc+bMKdZff/31EWscZ+8sRnggEQIPJELggUQIPJAIgQcSIfBAIgQeSITponGWvr6+Yv3GG28s1qdNmzZibWBgoKmeUBfTRQM4HYEHEiHwQCIEHkiEwAOJEHggEQIPJML18AktXbq0WJ83r3w494MPylMJcqy9ezHCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAidY/D275E0j9KGifphKReSf8l6b9rL3kwIv6jZR2icg8//HCxPmHChGL9zjvvrLIdtFEjI/wPJP0sIm6RNCDpEUmbI2JJ7Y+wA+eJuoGPiDUR8Urt4RRJ/yvpO7Z/Y3uDbc7WA84TDf8f3vYiSZMlvSLpmxGxQNKFkr49zGtX2u633V9ZpwBGraHR2falkn4u6S5JAxHxp1qpX9JVZ74+ItZLWl9bl3vaAV2i7ghve7ykX0haFREHJL1ge7btcZK+K+mtFvcIoCKNbNL/taS5kv7W9i5J/ynpBUlvSnojIl5tXXsAqsRtqseg8ePHF+v79+8v1t97771i/a677irWW/2bwrC4TTWA0xF4IBECDyRC4IFECDyQCIEHEiHwQCIchwfGBo7DAzgdgQcSIfBAIgQeSITAA4kQeCARAg8k0o4bUB6RdGDI47+oPdeN6K059Hbuqu5reiMvavmJN2d9oN3fyAkCnUBvzaG3c9epvtikBxIh8EAinQj8+g58ZqPorTn0du460lfb/w8PoHPYpAcSIfCSbF9g+6DtXbW/WZ3uqdvZnmp7d23567Z/P+T7m9Lp/rqN7Utsb7O9w/YvbY/vxG+urZv0tjdIminpXyPiibZ9cB2250rqjYgfd7qXoWxPlfRPEfEN2xdK+mdJl0raEBHPd7CvyZI2S/rLiJhre5mkqRGxtlM91foabmrzteqC35ztv5H0QUS8YnutpD9Imtju31zbRvjaj2JcRCySNMP2WXPSddBCddmMuLVQbZI0sfbUgxq8ycFiSd+zfXHHmpNOajBMx2qPF0paYfu3tp/sXFtnTW3+fXXJb65bZmFu5yb9Eklbass7JPW08bPr2as6M+J2wJmhWqJT399rkjp2MklEHIuIo0Oe2qbB/uZLWmT72g71dWaolqvLfnPnMgtzK7Qz8BMlfVxb/lTS1DZ+dj2/i4g/1JaHnRG33YYJVTd/f3si4nhEnJS0Xx3+/oaE6pC66DsbMgvz/erQb66dgf9c0kW15Ult/ux6zocZcbv5+9tu+2u2J0i6RdLbnWrkjFB1zXfWLbMwt/ML2KdTm1SzJX3Uxs+u56fq/hlxu/n7+4mkPkm/lrQuIsqzUbbIMKHqpu+sK2Zhbtteett/Lmm3pJ2Sbpe08IxNVgzD9q6IWGJ7uqRfSXpV0l9p8Ps72dnuuovtH0p6UqdGy42SfiR+c/+v3YflJkv6lqTXImKgbR88RtiepsERa3v2H26j+M2djlNrgUS6accPgBYj8EAiBB5IhMADiRB4IJH/A508nedmP1WRAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAACr1JREFUeJzt3VGoZAd9x/Hvr5sE4saGDd0uxgchsFAEsxBWu1sjewMaiPggVoigT6ksWMiLLyIVdlfaPPRBCoIrC1sJgVpiqcVSQzaRvclSY/WuNpo+iKUkamoeJLJr+mDp8u/DDt2b6+2d2bnnzMzu//uBy56ZOfeeP7Pz5cyde2ZOqgpJPfzOsgeQtDgGLzVi8FIjBi81YvBSIwYvNWLwUiMGLzVi8FIjt4y9gSQeyieN75dVtX/aSu7hpZvDK7OsNHfwSc4meSHJ5+b9GZIWa67gk3wE2FNVR4F7khwcdixJY5h3D78GPDlZPgfcv/nGJMeTbCTZ2MVskgY2b/B7gVcny68DBzbfWFVnqupwVR3ezXCShjVv8G8At0+W79jFz5G0QPOGepFrT+MPAS8PMo2kUc37d/h/AC4kuRt4CDgy3EiSxjLXHr6qLnP1hbvvAA9U1aUhh5I0jrmPtKuqX3HtlXpJNwBfbJMaMXipEYOXGjF4qRGDlxoxeKkRg5caMXipEYOXGjF4qRGDlxoxeKkRg5caMXipEYOXGjF4qRGDlxoxeKkRg5caMXipkdFPF61xHDt27P+97eTJkzt+73PPPbfj7dO+Xzcu9/BSIwYvNWLwUiMGLzVi8FIjBi81YvBSI6mqcTeQjLuBpnb6W/mJEydG3fb6+vqOtz/wwAOjbl/bulhVh6etdN17+CS3JPlpkvXJ17vmm0/Sos1zpN29wFer6jNDDyNpXPP8Dn8E+FCS7yY5m8TDc6UbxDzBfw94f1W9B7gV+ODWFZIcT7KRZGO3A0oazjx75x9W1W8myxvAwa0rVNUZ4Az4op20SubZwz+R5FCSPcCHgRcHnknSSObZw38e+BsgwDeq6tlhR5I0lusOvqpe4uor9WpqbW1tx9t3OkbA99ovl0faSY0YvNSIwUuNGLzUiMFLjRi81IjHwWtwO32EtpbLPbzUiMFLjRi81IjBS40YvNSIwUuNGLzUiH+H1+CmvX1Wy+MeXmrE4KVGDF5qxOClRgxeasTgpUYMXmrE00XfhMb+P92NJMse4WY1zumiJd24DF5qxOClRgxeasTgpUYMXmrE4KVGfD/8DcrPftc8ZtrDJzmQ5MJk+dYk/5jkn5M8Mu54koY0Nfgk+4DHgb2Tqx7l6lE97wU+muStI84naUCz7OGvAA8DlyeX14AnJ8vPA1MP55O0Gqb+Dl9Vl+FNx0DvBV6dLL8OHNj6PUmOA8eHGVHSUOZ5lf4N4PbJ8h3b/YyqOlNVh2c5mF/S4swT/EXg/snyIeDlwaaRNKp5/iz3OPDNJO8D3gn8y7AjSRrLXO+HT3I3V/fyT1fVpSnrru6bs29Svh++pZneDz/XgTdV9Z9ce6Ve0g3CQ2ulRgxeasTgpUYMXmrE4KVGDF5qxOClRgxeasTgpUYMXmrE4KVGDF5qxOClRvyYai3UyZMnd3W7dsc9vNSIwUuNGLzUiMFLjRi81IjBS40YvNSIwUuNGLzUiMFLjRi81IjBS40YvNSIwUuNGLzUiMFLjcwUfJIDSS5Mlt+e5OdJ1idf+8cdUdJQpn7iTZJ9wOPA3slVfwj8RVWdHnMwScObZQ9/BXgYuDy5fAT4ZJLvJ3lstMkkDW5q8FV1uaoubbrqKWANeDdwNMm9W78nyfEkG0k2BptU0q7N86Ldt6vq11V1BfgBcHDrClV1pqoOV9XhXU8oaTDzBP90krcleQvwIPDSwDNJGsk8H1N9CjgP/Dfw5ar68bAjSRrLzMFX1drk3/PAH4w1kKTxeOCN1IjBS40YvNSIwUuNGLzUiMFLjXi66JvQqVOndrz9xIkTC5pEq8Y9vNSIwUuNGLzUiMFLjRi81IjBS40YvNSIwUuNGLzUiMFLjRi81IjBS40YvNSIwUuNGLzUiMFLjRi81IjBS40YvNSIwUuNGLzUiMFLjRi81IifS6+FOnbs2LJHaG3qHj7JnUmeSnIuydeT3JbkbJIXknxuEUNKGsYsT+k/Dnyhqh4EXgM+BuypqqPAPUkOjjmgpOFMfUpfVV/adHE/8AngryaXzwH3Az8ZfjRJQ5v5RbskR4F9wM+AVydXvw4c2Gbd40k2kmwMMqWkQcwUfJK7gC8CjwBvALdPbrpju59RVWeq6nBVHR5qUEm7N8uLdrcBXwM+W1WvABe5+jQe4BDw8mjTSRpUqmrnFZJPAY8BL06u+grwaeBbwEPAkaq6tMP377wBLdy0//MxTTuV9cmTJxczyM3n4izPqGd50e40cHrzdUm+AXwA+MudYpe0WuY68KaqfgU8OfAskkbmobVSIwYvNWLwUiMGLzVi8FIjBi81YvBSIwYvNWLwUiMGLzVi8FIjBi81YvBSI35MdUPr6+s73r62tjbats+fPz/az9Z07uGlRgxeasTgpUYMXmrE4KVGDF5qxOClRqZ+Lv2uN+Dn0kuLMNPn0ruHlxoxeKkRg5caMXipEYOXGjF4qRGDlxqZ+n74JHcCfwvsAf4LeBj4d+A/Jqs8WlU/Gm1CSYOZeuBNkj8FflJVzyQ5DfwC2FtVn5lpAx54Iy3CMAfeVNWXquqZycX9wP8AH0ry3SRnk/ipOdINYubf4ZMcBfYBzwDvr6r3ALcCH9xm3eNJNpJsDDappF2bae+c5C7gi8AfA69V1W8mN20AB7euX1VngDOT7/UpvbQipu7hk9wGfA34bFW9AjyR5FCSPcCHgRdHnlHSQGZ5Sv8nwH3AnyVZB/4NeAL4V+CFqnp2vPEkDcm3x0o3B98eK+nNDF5qxOClRgxeasTgpUYMXmrE4KVGDF5qxOClRgxeasTgpUYMXmrE4KVGDF5qxOClRhbxAZS/BF7ZdPn3JtetImebj7Ndv6HnescsK43+ARi/tcFkY5Y36i+Ds83H2a7fsubyKb3UiMFLjSwj+DNL2OasnG0+znb9ljLXwn+Hl7Q8PqWXGjF4IMktSX6aZH3y9a5lz7TqkhxIcmGy/PYkP990/+1f9nyrJsmdSZ5Kci7J15PctozH3EKf0ic5C7wT+Keq+vOFbXiKJPcBD896RtxFSXIA+Luqel+SW4G/B+4CzlbVXy9xrn3AV4Hfr6r7knwEOFBVp5c102Su7U5tfpoVeMzt9izMQ1nYHn7yoNhTVUeBe5L81jnplugIK3ZG3ElUjwN7J1c9ytWTDbwX+GiSty5tOLjC1ZguTy4fAT6Z5PtJHlveWHwc+EJVPQi8BnyMFXnMrcpZmBf5lH4NeHKyfA64f4HbnuZ7TDkj7hJsjWqNa/ff88DSDiapqstVdWnTVU9xdb53A0eT3LukubZG9QlW7DF3PWdhHsMig98LvDpZfh04sMBtT/PDqvrFZHnbM+Iu2jZRrfL99+2q+nVVXQF+wJLvv01R/YwVus82nYX5EZb0mFtk8G8At0+W71jwtqe5Ec6Iu8r339NJ3pbkLcCDwEvLGmRLVCtzn63KWZgXeQdc5NpTqkPAywvc9jSfZ/XPiLvK998p4DzwHeDLVfXjZQyxTVSrdJ+txFmYF/YqfZLfBS4A3wIeAo5secqqbSRZr6q1JO8Avgk8C/wRV++/K8udbrUk+RTwGNf2ll8BPo2Puf+z6D/L7QM+ADxfVa8tbMM3iSR3c3WP9XT3B+6sfMy9mYfWSo2s0gs/kkZm8FIjBi81YvBSIwYvNfK/3HzQYASBocAAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADUVJREFUeJzt3W+sVPWdx/HPhz8mgKxChAs2ijHiA0IlKu1yLeg1FhNqMchWbdLuAy0hWROfbDTdxmaTVtcHq6mbNAK5EQmYLETMYthYEFgh4tZue4Et6z6oNRtocRElVMFNrCz57oM7Xa4X7m+GuXNmBr7vV3LDmfnOmfPNyXz4nXvOmftzRAhADmM63QCA9iHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSGVf1BmxzKx9QveMRMa3eixjhgUvD4UZe1HTgba+1/bbtHzb7HgDaq6nA214uaWxE9Eq63vbs1rYFoArNjvB9kl6uLe+QtHBo0fZK2wO2B0bRG4AWazbwkyS9X1s+IalnaDEi+iNifkTMH01zAFqr2cB/KmlCbfnyUbwPgDZqNqj7dPYwfp6kQy3pBkClmr0O/6qkvbavlrRE0oLWtQSgKk2N8BFxUoMn7n4h6c6I+KSVTQGoRtN32kXEH3T2TD2AiwAn24BECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkjkggNve5zt39neU/v5chWNAWi9ZqaLvknSxoj4fqubAVCtZg7pF0j6pu1f2l5ru+k55gG0VzOB/5Wkr0fEVyWNl/SN4S+wvdL2gO2B0TYIoHWaGZ0PRsQfa8sDkmYPf0FE9EvqlyTb0Xx7AFqpmRH+JdvzbI+VtEzSr1vcE4CKNDPC/1jSP0qypK0Rsau1LQGoygUHPiLe0eCZenSpMWPKB2633XZbsf7KK68U69OnTy/WbY9Y27hxY3HdF198sVjftYvxZTS48QZIhMADiRB4IBECDyRC4IFECDyQCPfBX4I2b95crC9btmxU7799+/Zi/fTp0yPWlixZUlx31qxZxTqX5UaHER5IhMADiRB4IBECDyRC4IFECDyQCIEHEuE6fJeaO3dusb5169YRa9dcc01x3TNnzhTry5cvL9Zfe+21Yn3GjBkj1latWlVcd9w4PpJVYoQHEiHwQCIEHkiEwAOJEHggEQIPJELggUQcUe3EMMw8c34zZ84s1t94441i/cYbbxyxduTIkeK6d911V7H+3nvvFeujceWVVxbrBw4cKNbvvPPOYv3QoUMX2tKlYl9EzK/3IkZ4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiELx93yJYtW4r10nV2Sdq9e/eItRUrVhTX7eS16g0bNhTr1157bbH+5JNPFuuPPPLIiLVTp04V182goRHedo/tvbXl8bb/2fa/2n642vYAtFLdwNueImm9pEm1px7V4F09X5P0LduTK+wPQAs1MsKfkfSgpJO1x32SXq4tvymp7u18ALpD3d/hI+KkJNn+01OTJL1fWz4hqWf4OrZXSlrZmhYBtEozZ+k/lTShtnz5+d4jIvojYn4jN/MDaJ9mAr9P0sLa8jxJh1rWDYBKNXNZbr2kn9leJGmOpH9rbUsAqtJw4COir/bvYduLNTjK/21ElP/IeVJLly4t1m+99dZi/dixY8X6Cy+8MGKt098Jv++++0as3XzzzaN674ULFxbrEyZMGLHGdfgmb7yJiP/W2TP1AC4S3FoLJELggUQIPJAIgQcSIfBAInw9tklTp04t1p999tlifcyY8v+127dvL9Y3bdpUrFepdElQkh566KHKtt3f31+sf/jhh5Vt+1LACA8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiXAdvkknTpwo1utN2XzDDTcU63fccUexftVVV41Ymz59enHduXPnFusPPPBAsX7vvfcW66Nx8ODBYn3nzp2VbTsDRnggEQIPJELggUQIPJAIgQcSIfBAIgQeSITr8BV5/PHHi/V633e/7rrrivV33313xNr48eOL606cOLFYP336dLH+3HPPFevLli0bsVbv/oONGzcW6wMDA8U6yhjhgUQIPJAIgQcSIfBAIgQeSITAA4kQeCARrsNXZP/+/cX64sWLi/U1a9a0sp0vqPed8nr3CJw8ebJYf+yxx0asHT9+vLju0aNHi3WMTkMjvO0e23try1+yfcT2ntrPtGpbBNAqdUd421MkrZc0qfbUn0v6u4hYXWVjAFqvkRH+jKQHJf3pOG6BpBW299t+urLOALRc3cBHxMmI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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADbFJREFUeJzt3W+MVfWdx/HPR5BIoSsY2ZE2pgaDMdWKMZQdFpqwChhrH0yQhCYlMboVwyY+KWqXLNlYXDWuSbMJSSkT2caYrBtrFtMVVP5YIhFqO/yR2geVDQ4UrQ8QlKJJyZLvPuB2GYH5ncud+2/4vl/JxDP3e889X4/34+/O+Z17jiNCAHK4rNMNAGgfAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IJGxrd6AbU7lA1rvaERMqXoSIzxwaThUz5MaDrzt9bZ32V7V6GsAaK+GAm97kaQxETFb0jTb05vbFoBWaHSEnyfpxdryZklzhxZtL7M9YHtgBL0BaLJGAz9B0ge15WOSeoYWI6I/ImZGxMyRNAeguRoN/ElJ42vLE0fwOgDaqNGg7tbZj/EzJA02pRsALdXoPPzLknbY/oqkuyT1Nq8lAK3S0AgfESd05sDdryT9XUR82symALRGw2faRcRxnT1SD2AU4GAbkAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiVx04G2PtX3Y9vbazzda0RiA5mvkdtG3SHohIn7Y7GYAtFYjH+l7JX3H9q9tr7fd8D3mAbRXI4H/jaT5ETFL0uWSvn3uE2wvsz1ge2CkDQJonkZG5/0R8efa8oCk6ec+ISL6JfVLku1ovD0AzdTICP+87Rm2x0jqk/ROk3sC0CKNjPCrJf2HJEv6RURsbW5LAFrlogMfEe/qzJF6dNC4ceOGrU2cOLG47rFjx5rdziWhr6+vWF+/fn2xvmTJkmJ969bOj42ceAMkQuCBRAg8kAiBBxIh8EAiBB5IhPPgu9ScOXOK9dIU0dVXX11cd+HChcX6nj17ivXRavXq1cX6ypUri/XPPvusWH/vvfcuuqd2Y4QHEiHwQCIEHkiEwAOJEHggEQIPJELggUSYh++Q3t7eYv2NN94o1seOHf4/3fHjx4vrHjx4sFgfzR599NFha6tWrSque+rUqWK96uuxhw8fLta7ASM8kAiBBxIh8EAiBB5IhMADiRB4IBECDyTCPHyLPPDAA8X6gw8+WKyX5tml8lz7DTfcUFz3k08+Kda72eLFi4v1hx9+uOHX3rhxY7G+YsWKhl+7WzDCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAizMMXlG67vGHDhuK6d9xxR7Fuu1g/dOhQsb5o0aJha6P5dtCzZs0q1tetW1esT548edha1X55/PHHi/VLQV0jvO0e2ztqy5fb/m/bb9m+v7XtAWimysDbnizpOUkTag89JGl3RMyRtNj2l1vYH4AmqmeEPy1piaQTtd/nSXqxtvympJnNbwtAK1T+DR8RJ6Qv/M05QdIHteVjknrOXcf2MknLmtMigGZp5Cj9SUnja8sTL/QaEdEfETMjgtEf6CKNBH63pLm15RmSBpvWDYCWamRa7jlJm2x/S9LXJb3d3JYAtErdgY+IebV/HrK9QGdG+X+OiNMt6q3jnnrqqWFrt99+e3HdiCjW3367/P/Je+65p1j/8MMPi/VuVTq3QZKeeeaZYn3SpEnFemm/P/3008V19+3bV6xfCho68SYiPtTZI/UARglOrQUSIfBAIgQeSITAA4kQeCCR1F+Prbpl87JljZ8d/MorrxTr9957b7E+mi8lXfLEE08U63Pnzi3Wq5T2e9WUXwaM8EAiBB5IhMADiRB4IBECDyRC4IFECDyQSOp5+L6+vmK96pbNJXv37i3WL9V5dkm67rrrhq1V3e65yueff16sr1mzZthaqS9JGhwcbKCj0YURHkiEwAOJEHggEQIPJELggUQIPJAIgQcSST0P//777xfrVbd0Llm6dGmxfsUVVxTr27ZtK9Z37do1bO3kyZPFdUdq3LhxxfpLL700bG3q1Kkj2vamTZuK9euvv37Y2tatW0e07UsBIzyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJOKq2xqPeAN2azcwAlXzyfv37x+2Nn369BFtu2qOv+q/y8cffzxs7cCBA8V1d+7cOaJt33zzzcX6nXfeOWxtpP/eVfr7+4etLV++fESv3eV2R8TMqifVNcLb7rG9o7b8VdtHbG+v/UwZaacA2qPyTDvbkyU9J2lC7aG/kfRERKxtZWMAmq+eEf60pCWSTtR+75X0fdt7bD/Zss4ANF1l4CPiRER8OuShVyXNk/RNSbNt33LuOraX2R6wPdC0TgGMWCNH6XdGxJ8i4rSkvZLOO3oVEf0RMbOegwgA2qeRwL9ue6rtL0laKOndJvcEoEUa+XrsjyT9UtIpST+NiN83tyUArZJ6Hr7KNddcM2ztvvvuK65bdf31G2+8sViv+r78SJw6dapYr7oe/2WXNX6+VtU8/IkTJ4r1xx57rFh/9tlnh621+joBHda8eXgAlwYCDyRC4IFECDyQCIEHEiHwQCJMy3XITTfdVKyP9HLOJUePHi3Wqy4F3dPT0/C2S5ewlqRHHnmkWD98+HDD277EMS0H4IsIPJAIgQcSIfBAIgQeSITAA4kQeCAR5uETWrBgQbH+8ssvF+vjx49veNvTpk0r1gcHBxt+7eSYhwfwRQQeSITAA4kQeCARAg8kQuCBRAg8kEgj16VHl5s0aVKxXjXPXnWJ7KrLXPf19Q1bY569sxjhgUQIPJAIgQcSIfBAIgQeSITAA4kQeCAR5uFHqdItm1etWlVct+r77FXXSFi3bl2x/tprrxXr6JzKEd72lbZftb3Z9gbb42yvt73LdvmdBaCr1POR/nuSfhwRCyV9JOm7ksZExGxJ02xPb2WDAJqn8iN9RPxkyK9TJC2V9G+13zdLmivpQPNbA9BsdR+0sz1b0mRJf5D0Qe3hY5LOu9GY7WW2B2wPNKVLAE1RV+BtXyVpjaT7JZ2U9JejPhMv9BoR0R8RM+u5qB6A9qnnoN04ST+XtDIiDknarTMf4yVphqTBlnUHoKkqL1Nte7mkJyW9U3voZ5J+IGmbpLsk9UbEp4X1uUx1C/T29g5be+utt4rr2i7Wjx07VqzfeuutxfqRI0eKdbREXZeprueg3VpJa4c+ZvsXkhZI+tdS2AF0l4ZOvImI45JebHIvAFqMU2uBRAg8kAiBBxIh8EAiBB5IhK/Hdqlrr722WN+4ceOwtZHOs8+fP79YZ5599GKEBxIh8EAiBB5IhMADiRB4IBECDyRC4IFEmIfvUtOmTSvWS7eErrrGwZYtW4r1ffv2FesYvRjhgUQIPJAIgQcSIfBAIgQeSITAA4kQeCAR5uG71N13392y116zZk3LXhvdjREeSITAA4kQeCARAg8kQuCBRAg8kAiBBxKpnIe3faWk/5Q0RtJnkpZI+h9JB2tPeSgiftuyDpPatm1bsb5ixYqGX/vgwYPVT8IlqZ4R/nuSfhwRCyV9JOkfJb0QEfNqP4QdGCUqAx8RP4mIv1wiZYqk/5X0Hdu/tr3eNmfrAaNE3X/D254tabKkLZLmR8QsSZdL+vYFnrvM9oDtgaZ1CmDE6hqdbV8laY2keyR9FBF/rpUGJE0/9/kR0S+pv7Zu+QJrANqmcoS3PU7SzyWtjIhDkp63PcP2GEl9kt5pcY8AmqSej/R/L+k2Sf9ke7uk30l6XtI+SbsiYmvr2gPQTK66pPGIN8BHeqAddkfEzKonceINkAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJNKOC1AelXRoyO9X1x7rRvTWGHq7eM3u62v1PKnlF8A4b4P2QD1f1O8EemsMvV28TvXFR3ogEQIPJNKJwPd3YJv1orfG0NvF60hfbf8bHkDn8JEeSITAS7I91vZh29trP9/odE/dznaP7R215a/aPjJk/03pdH/dxvaVtl+1vdn2BtvjOvGea+tHetvrJX1d0saI+Je2bbiC7dskLYmIH3a6l6Fs90h6KSK+ZftySf8l6SpJ6yPi3zvY12RJL0j664i4zfYiST0RsbZTPdX6utCtzdeqC95ztv9B0oGI2GJ7raQ/SprQ7vdc20b42ptiTETMljTN9nn3pOugXnXZHXFroXpO0oTaQw/pzM0G5khabPvLHWtOOq0zYTpR+71X0vdt77H9ZOfaOu/W5t9Vl7znuuUuzO38SD9P0ou15c2S5rZx21V+o4o74nbAuaGap7P7701JHTuZJCJORMSnQx56VWf6+6ak2bZv6VBf54ZqqbrsPXcxd2FuhXYGfoKkD2rLxyT1tHHbVfZHxB9ryxe8I267XSBU3bz/dkbEnyLitKS96vD+GxKqP6iL9tmQuzDfrw6959oZ+JOSxteWJ7Z521VGwx1xu3n/vW57qu0vSVoo6d1ONXJOqLpmn3XLXZjbuQN26+xHqhmSBtu47Sqr1f13xO3m/fcjSb+U9CtJP42I33eiiQuEqpv2WVfchbltR+lt/5WkHZK2SbpLUu85H1lxAba3R8Q821+TtEnSVkl/qzP773Rnu+sutpdLelJnR8ufSfqBeM/9v3ZPy02WtEDSmxHxUds2fImw/RWdGbFez/7GrRfvuS/i1FogkW468AOgxQg8kAiBBxIh8EAiBB5I5P8AdxKeYL8MiU0AAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADNNJREFUeJzt3W+MXXWdx/HPpwNNapGmxDpYoSUkTTYGOklTdbrSZCRCQjGNUZOa6PIAmyZL6BMDGMEsUXabsASBDGPNhGqAsC7VqMGsDYUNpY0gOrVL1QdGQxi1tQ+EQgskkh2++2But8Mw87u3955z7535vl9J03Pv9557vj29n/mdueefI0IAcljS6wYAdA+BBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQyHl1L8A2h/IB9ftbRKxq9iJGeGBxmGzlRW0H3vYe28/b/nq77wGgu9oKvO3PShqIiE2SLre9rtq2ANSh3RF+RNLexvR+SVfNLNreYXvC9kQHvQGoWLuBXy7pWGP6VUmDM4sRMR4RGyNiYyfNAahWu4F/Q9KyxvQFHbwPgC5qN6iHdXYzfkjSy5V0A6BW7e6H/4mkQ7ZXS7pO0nB1LQGoS1sjfESc0vQXd7+Q9MmIeL3KpgDUo+0j7SLipM5+Uw9gAeDLNiARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kEjbN5NEb42Ojs5bu+mmm4rzPvLII8X6ZZddVqyPjIwU60eOHJm3dvfddxfnffzxx4t1dOacR3jb59n+k+0DjT9X1tEYgOq1M8Kvl/T9iPhq1c0AqFc7v8MPS/q07V/a3mObXwuABaKdwP9K0qci4mOSzpe0ZfYLbO+wPWF7otMGAVSnndH5aET8vTE9IWnd7BdExLikcUmyHe23B6BK7Yzwj9oesj0g6TOSXqy4JwA1aWeE/6ak/5BkSU9ExNPVtgSgLo6od4ubTfr2bN++vVgfGxubtzYwMFB1O+9iu1gvfabefvvt4rxHjx4t1oeHh4v1xA5HxMZmL+JIOyARAg8kQuCBRAg8kAiBBxIh8EAiHAffp6644opive5db3VZunRpsb5mzZpi/eKLLy7WT5w4cc49ZcIIDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJcHpsn5qamirW6/5/Kzl27Fix/s4778xbGxwcLM7bbD/9iy+Wr7fywAMPzFtrdnnuBY7TYwG8G4EHEiHwQCIEHkiEwAOJEHggEQIPJML58D2ydevWni37tddeK9bvuuuuYr20r7uZbdu2Feu33XZbsT40NFSsr1+//px7yoQRHkiEwAOJEHggEQIPJELggUQIPJAIgQcS4Xz4mlxyySXF+uTkZLG+ZEn5Z3HpnPNmHnvssWL9hhtuaPu9O9Xpejt06NC8teuvv74475tvvlms97nqzoe3PWj7UGP6fNs/tf1z2zd22iWA7mkaeNsrJT0saXnjqZ2a/mnyCUmft/3+GvsDUKFWRvgpSdsknWo8HpG0tzF9UFLTzQgA/aHpsfQRcUqSbJ95armkMxc1e1XSey5SZnuHpB3VtAigKu18S/+GpGWN6Qvmeo+IGI+Ija18iQCge9oJ/GFJVzWmhyS9XFk3AGrVzumxD0v6me3Nkj4i6YVqWwJQl5YDHxEjjb8nbV+j6VH+XyKifAF1zKnZ8Q/N9rOX5n/llVeK846NjRXr/azZetu8efO8tRUrVhTnXeD74VvS1gUwIuK4zn5TD2CB4NBaIBECDyRC4IFECDyQCIEHEuEy1YvQ/v37i/UXXuDQiawY4YFECDyQCIEHEiHwQCIEHkiEwAOJEHggEfbDL0Kjo6O9bgF9ihEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEuF8+B6xXawvWVL+WVy6nXSz9+5n9957b7He7N92//33z1s7fvx4Wz0tJi2N8LYHbR9qTH/Y9l9sH2j8WVVviwCq0nSEt71S0sOSljee+rikf4uI3XU2BqB6rYzwU5K2STrVeDwsabvtX9veVVtnACrXNPARcSoiXp/x1D5JI5I+KmmT7fWz57G9w/aE7YnKOgXQsXa+pX8uIk5HxJSkI5LWzX5BRIxHxMaI2NhxhwAq007gn7T9Idvvk3StpN9W3BOAmrSzW+4bkp6R9Lak70TE76ttCUBdWg58RIw0/n5G0j/U1VAWEVGsl/azN5u/2Xv30oYNG4r1LVu2FOvN/m39/G/vBxxpByRC4IFECDyQCIEHEiHwQCIEHkiE02PRVbfcckuxvmzZso7ef3JysqP5FztGeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhP3w6Kq1a9fW+v6jo6O1vv9CxwgPJELggUQIPJAIgQcSIfBAIgQeSITAA4mwHx6VGxoamre2Zs2aLnaC2RjhgUQIPJAIgQcSIfBAIgQeSITAA4kQeCAR9sP3iO1ifcmS8s/i0u2km713p5rd8nnnzp3z1lavXl2c9/Tp08X61q1bi3WUNR3hba+wvc/2fts/tr3U9h7bz9v+ejeaBFCNVjbpvyjpWxFxraQTkr4gaSAiNkm63Pa6OhsEUJ2mm/QR8e0ZD1dJ+pKk+xuP90u6StIfqm8NQNVa/tLO9iZJKyX9WdKxxtOvShqc47U7bE/YnqikSwCVaCnwti+SNCrpRklvSDpzx78L5nqPiBiPiI0RsbGqRgF0rpUv7ZZK+oGkr0XEpKTDmt6Ml6QhSS/X1h2ASrWyW+7LkjZIusP2HZK+J+mfbK+WdJ2k4Rr7W7COHz9erN98883F+oMPPlisR8S8tauvvro476WXXlqsNzM+Pl6sX3jhhfPWSn1L0smTJ4v1gwcPFusoa+VLu92Sds98zvYTkq6R9O8R8XpNvQGoWFsH3kTESUl7K+4FQM04tBZIhMADiRB4IBECDyRC4IFE3Gy/aMcLsOtdwCL10ksvFeu9vNxzs9NvO/lM3XrrrcX6fffd1/Z7L3KHWzmylREeSITAA4kQeCARAg8kQuCBRAg8kAiBBxLhMtV9ateuXcX62NjYvLWBgYGq2zknb7311ry1O++8szjv3r2chFknRnggEQIPJELggUQIPJAIgQcSIfBAIgQeSIT98H3qoYceKtZL55zffvvtxXnXrl3bVk9nPPvss8X6PffcM29t3759HS0bnWGEBxIh8EAiBB5IhMADiRB4IBECDyRC4IFEml6X3vYKSf8paUDSm5K2SfqjpDMXTt8ZEb8pzM916YH6tXRd+lYCf5OkP0TEU7Z3S/qrpOUR8dVWuiDwQFdUcyOKiPh2RDzVeLhK0v9K+rTtX9reY5uj9YAFouXf4W1vkrRS0lOSPhURH5N0vqQtc7x2h+0J2xOVdQqgYy2NzrYvkjQq6XOSTkTE3xulCUnrZr8+IsYljTfmZZMe6BNNR3jbSyX9QNLXImJS0qO2h2wPSPqMpBdr7hFARVrZpP+ypA2S7rB9QNLvJD0q6X8kPR8RT9fXHoAqcbtoYHHgdtEA3o3AA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEunGBSj/JmlyxuMPNJ7rR/TWHno7d1X31dI9wGu/AMZ7FmhPtHKifi/QW3vo7dz1qi826YFECDyQSC8CP96DZbaK3tpDb+euJ311/Xd4AL3DJj2QCIGXZPs823+yfaDx58pe99TvbA/aPtSY/rDtv8xYf6t63V+/sb3C9j7b+23/2PbSXnzmurpJb3uPpI9I+q+I+NeuLbgJ2xskbWv1jrjdYntQ0g8jYrPt8yX9SNJFkvZExHd72NdKSd+X9MGI2GD7s5IGI2J3r3pq9DXXrc13qw8+c53ehbkqXRvhGx+KgYjYJOly2++5J10PDavP7ojbCNXDkpY3ntqp6ZsNfELS522/v2fNSVOaDtOpxuNhSdtt/9r2rt61pS9K+lZEXCvphKQvqE8+c/1yF+ZubtKPSNrbmN4v6aouLruZX6nJHXF7YHaoRnR2/R2U1LODSSLiVES8PuOpfZru76OSNtle36O+ZofqS+qzz9y53IW5Dt0M/HJJxxrTr0oa7OKymzkaEX9tTM95R9xumyNU/bz+nouI0xExJemIerz+ZoTqz+qjdTbjLsw3qkefuW4G/g1JyxrTF3R52c0shDvi9vP6e9L2h2y/T9K1kn7bq0Zmhapv1lm/3IW5myvgsM5uUg1JermLy27mm+r/O+L28/r7hqRnJP1C0nci4ve9aGKOUPXTOuuLuzB37Vt62xdKOiTpvyVdJ2l41iYr5mD7QESM2F4r6WeSnpb0j5pef1O97a6/2P5nSbt0drT8nqSviM/c/+v2brmVkq6RdDAiTnRtwYuE7dWaHrGezP7BbRWfuXfj0FogkX764gdAzQg8kAiBBxIh8EAiBB5I5P8ASoxpWF3pbrEAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADSRJREFUeJzt3W+oVPedx/HPJ9d7g3q7wbDG9AoxCBJoMIZgjXerYEIT0PigdIUIbR7EFkMXJLAEmhJZ0maTB/ugrJSoCG7zB7YhhnVpsKI3oSbSpmvvrVtjH5Quy7XVKEmxaM0Dk5jvPnCyXo1zZjxzzsxcv+8XiOfOd845Xw7z4TczvzPnOCIEIIcbet0AgO4h8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEplR9w5scyofUL8/R8TcVk9ihAeuD8faeVLpwNveafsd25vLbgNAd5UKvO2vSxqIiFFJC20vqrYtAHUoO8KvkvRqY3m/pBVTi7Y32h63Pd5BbwAqVjbwsyWdaCyfljRvajEidkTE0ohY2klzAKpVNvDnJM1sLA93sB0AXVQ2qBO69DZ+iaTJSroBUKuy8/D/Kemg7RFJqyUtr64lAHUpNcJHxFld/OLuV5Lui4gzVTYFoB6lz7SLiL/o0jf1AKYBvmwDEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJlL6ZJOq1YsWKwvro6GjT2tq1awvXXbduXWH9gw8+KKxj+rrmEd72DNt/tH2g8W9xHY0BqF6ZEf4uST+JiO9W3QyAepX5DL9c0lrbh2zvtM3HAmCaKBP4X0v6akQskzQoac2VT7C90fa47fFOGwRQnTKj85GION9YHpe06MonRMQOSTskyXaUbw9AlcqM8C/bXmJ7QNLXJP224p4A1KTMCP8DSf8uyZJ+GhFvVNsSgLpcc+Aj4qguflOPDgwPDxfW9+3bV1ifOXNm6X2PjIwU1vt5Hn7BggWF9fvvv79pbWxsrHDd48ePl+ppOuFMOyARAg8kQuCBRAg8kAiBBxIh8EAinAdfk1mzZhXWn3/++cJ6J9NuJ06cKKyfOXOm9LZ77ZVXXims33vvvU1rb731VuG69913X6mephNGeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhHn4klr9vHXLli2F9UceeaTKdi6zfv36wvrk5GRt++7Upk2bCut333136W3fcsstpde9XjDCA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAizMOXtHLlysL6o48+Wuv+X3rppaa1iYmJWvddp2effbawfuONN5bedtExy4IRHkiEwAOJEHggEQIPJELggUQIPJAIgQcSYR6+wIwZzQ/P5s2ba933rl27CusbNmxoWvv000+rbqcyQ0NDhXXbte37yJEjtW17umhrhLc9z/bBxvKg7ddt/8J281cdgL7TMvC250h6UdLsxkObJE1ExFckrbP9hRr7A1Chdkb4C5IelnS28fcqSa82lt+WtLT6tgDUoeVn+Ig4K1322Wq2pM9uXnZa0rwr17G9UdLGaloEUJUy39Kfk/TZnQ6Hr7aNiNgREUsjgtEf6CNlAj8haUVjeYmkycq6AVCrMtNyL0r6me2Vkr4k6b+qbQlAXdoOfESsavx/zPYDujjK/1NEXKipt567/fbbm9ZGRkY62vZHH31UWN+6dWthvZ/n2ossW7assD44OFjbvlevXl1Y37t3b2377helTryJiPd06Zt6ANMEp9YCiRB4IBECDyRC4IFECDyQCD+PLbBo0aKmtdtuu62jbZ8/f76wfvDgwY62369aTUdGRG37Hhsbq23b0wUjPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kwjx8gYGBgaa1Ti+n/MwzzxTWp+vPXyXphhuajyNPPPFE4bqd3A5akrZv3960tn///o62fT1ghAcSIfBAIgQeSITAA4kQeCARAg8kQuCBRJiHL/DQQw/Vtu358+cX1osukS1Jk5OTTWu33npr4bqLFy8urE9MTBTWT58+XVgvus32unXrCtft1AsvvNC01uoaBBkwwgOJEHggEQIPJELggUQIPJAIgQcSIfBAIszDF9izZ0/T2mOPPdbRth9//PHC+tq1awvrnczD33nnnYX1w4cPF9ZbzcN3eq0A1KetEd72PNsHG8vzbR+3faDxb269LQKoSssR3vYcSS9Kmt146F5Jz0bEtjobA1C9dkb4C5IelnS28fdySd+2/Rvbz9XWGYDKtQx8RJyNiDNTHtoraZWkL0satX3XlevY3mh7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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADahJREFUeJzt3X2IXfWdx/HPJw+CiQ8kbhy1QjEQWKo1oKmbbA1GaIXUoKFWDD6gpHWwK/pHERpdRQwacIWymtA0kWyjgXVJ1K5drcakKMataTrRtZtFxWUZ02oEH0pSN1qd5Lt/5LKZJDO/e+fecx/G7/sFg2fu9557vlzvJ+fM+Z1zf44IAchhQrcbANA5BB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCKT2r0B21zKB7TfhxExo96T2MMDXw7vNPKkpgNve53tV2zf2exrAOispgJv+7uSJkbEPEkzbc+qti0A7dDsHn6BpI215eclXTi8aLvf9oDtgRZ6A1CxZgM/VdK7teWPJfUNL0bE2oiYExFzWmkOQLWaDfwnko6vLZ/QwusA6KBmg7pThw/jZ0sarKQbAG3V7Dj8v0raZvsMSQslza2uJQDt0tQePiL26dCJu+2SLo6IvVU2BaA9mr7SLiL+pMNn6gGMA5xsAxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IJExB972JNu7bb9Y+/l6OxoDUL1mpos+V9JjEfHjqpsB0F7NHNLPlbTI9g7b62w3Pcc8gM5qJvC/k/StiLhA0mRJ3zn6Cbb7bQ/YHmi1QQDVaWbv/PuI+EtteUDSrKOfEBFrJa2VJNvRfHsAqtTMHn6D7dm2J0paLOn1insC0CbN7OGXS/pnSZb0y4jYWm1LANplzIGPiF06dKYeLVi8eHGx/sQTTxTrEyaMfnC2Zs2a4rr33HNPsb5nz55iHeMXF94AiRB4IBECDyRC4IFECDyQCIEHEnFEey+Ey3ql3UknnVSsv/rqq8X6zJkzq2znCG+++WaxfvvttxfrTz31VJXtdMy555ZHky+//PJi/a233irWn3766WJ9//79xXqLdkbEnHpPYg8PJELggUQIPJAIgQcSIfBAIgQeSITAA4kwDt8mZ555ZrG+e/full7/008/HbVW79baCy64oFg/66yzivUNGzYU6zfddNOotaGhoeK6rVq0aNGotYcffri4bl9fX0vbrve+XH/99S29fh2MwwM4EoEHEiHwQCIEHkiEwAOJEHggEQIPJMK8cOPUQw89NGqt3v3sp556arG+atWqYn3p0qXFemksvNV76adNm1asX3HFFaPWbLe07YMHDxbrO3bsaOn1O4E9PJAIgQcSIfBAIgQeSITAA4kQeCARAg8kwv3wbTJpUvkShwceeKBYv/XWW4v19957b9Ta/Pnzi+sODg4W6/XGus8///xifdmyZaPWzj777OK6rd6T3oq33367WL/vvvuK9UcffbTKdsaquvvhbffZ3lZbnmz732z/u+3yFRgAekrdwNueJukRSVNrD92iQ/+afFPS92yf2Mb+AFSokT38AUlXSdpX+32BpI215Zck1T2MANAb6l5LHxH7pCOuQ54q6d3a8seSjvmjy3a/pP5qWgRQlWbO0n8i6fja8gkjvUZErI2IOY2cRADQOc0EfqekC2vLsyUNVtYNgLZq5vbYRyT9yvZ8SV+T9NtqWwLQLk2Nw9s+Q4f28psjYm+d56Ych2/VypUri/Wbb7551NrAwEBx3bvuuqtY37x5c7HeitmzZxfrr732Wtu2vWnTpmL9uuuuK9Y///zzKtupWkPj8E19AUZEvKfDZ+oBjBNcWgskQuCBRAg8kAiBBxIh8EAi3B7bo+p9pfKSJUtGrV155ZXFdU8//fRivd6w3datW4v1iy++eNTac889V1x38uTJxXo9GzeOPnh09dVXF9et9zXUPY7pogEcicADiRB4IBECDyRC4IFECDyQCIEHEmEcHmN26aWXFuvr168ftXbKKae0tO177723WF+xYsWotc8++6ylbfc4xuEBHInAA4kQeCARAg8kQuCBRAg8kAiBBxJhHB7HOOecc4r1LVu2FOutTPlc76ukr7322mL9iy++aHrb4xzj8ACOROCBRAg8kAiBBxIh8EAiBB5IhMADiTQ1eyzGtylTphTrpXvKpdbG2QcHB4v1a665plgfGhpqettocA9vu8/2ttryV2z/0faLtZ8Z7W0RQFXq7uFtT5P0iKSptYf+RtJ9EbG6nY0BqF4je/gDkq6S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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPwAAAD6CAYAAACF8ip6AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvNQv5yAAADPdJREFUeJzt3W+oHfWdx/HPZ29uJE2iXDF7SfKgIAaXQBOMaffGWrhKFYzFhG7BQLpPbImsmCcilJoitOwq7oO4UPCWSLb+Y7uaZbNktWLMmmDYJLY37bZbH8SKmLYxwQSDN1k1wfDdB5ndXK+5c07mzJxzbr7vF1zunPM9M/PlcD785szMmXFECEAOf9brBgB0D4EHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJDIrKZXYJtT+YDmnYiIBa1exAgPXB4Ot/OiyoG3vdX2fts/qLoMAN1VKfC2vylpICJWSbrW9pJ62wLQhKoj/KikF4rpnZJunly0vcH2uO3xDnoDULOqgZ8r6Ugx/YGk4cnFiNgSESsjYmUnzQGoV9XAn5Y0p5ie18FyAHRR1aAe1IXN+OWS3q2lGwCNqnoc/t8k7bW9SNIdkkbqawlAUyqN8BExofM77g5IuiUiPqyzKQDNqHymXUSc1IU99QBmAHa2AYkQeCARAg8kQuCBRAg8kAiBBxIh8EAiBB5IhMADiRB4IBECDyRC4IFECDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCARAg8kQuCBRCrfTBKoYmSk/M7io6OjHS3/gQcemLZ2zTXXdLRs2x3Nf999901bGxsb62jZ7brkEd72LNt/sL2n+PtSE40BqF+VEX6ZpJ9FxPfqbgZAs6p8hx+R9A3bv7C91TZfC4AZokrgfynp6xHxFUmDklZPfYHtDbbHbY932iCA+lQZnX8bEWeK6XFJS6a+ICK2SNoiSbajensA6lRlhH/W9nLbA5LWSvpNzT0BaEiVEf5Hkv5JkiXtiIhd9bYEoCmOaHaLm036/jM0NFRa3759e2n9hhtuqLzuK664orQ+ODhYedn97sSJE9PWhoeHO138wYhY2epFnGkHJELggUQIPJAIgQcSIfBAIgQeSITz4Geol156adrasmXLSudtdehrwYIFlXpqR6ufmDZ9mLiXNm/e3OsWGOGBTAg8kAiBBxIh8EAiBB5IhMADiRB4IBGOw/ephQsXltavu+66aWuLFi2qu53avPXWW6X1N998s6Pllx3rPn78eEfL7tT777/f0/VLjPBAKgQeSITAA4kQeCARAg8kQuCBRAg8kAjH4fvUxx9/XFo/e/Zs5WV/9NFHpfX9+/eX1nfs2FFaP3DgwLS1o0ePls575MiR0jo6wwgPJELggUQIPJAIgQcSIfBAIgQeSITAA4lwHL5Ptbp+eycefvjh0vrjjz/e2LrRW22N8LaHbe8tpgdt/7vt/7R9T7PtAahTy8DbHpL0tKS5xVMbdf7m81+V9C3b8xvsD0CN2hnhz0m6W9JE8XhU0gvF9OuSVtbfFoAmtPwOHxET0me+U86V9H8nPH8gaXjqPLY3SNpQT4sA6lJlL/1pSXOK6XkXW0ZEbImIlRHB6A/0kSqBPyjp5mJ6uaR3a+sGQKOqHJZ7WtLPbX9N0lJJb9TbEoCmtB34iBgt/h+2fZvOj/IPR8S5hnq7rF155ZWl9Weeeaa0vnTp0srrfueddyrPi5mt0ok3EfGeLuypBzBDcGotkAiBBxIh8EAiBB5IhMADifDz2B5Zs2ZNaX316tVd6gSZMMIDiRB4IBECDyRC4IFECDyQCIEHEiHwQCIch2/I4OBgaX3t2rVd6uTz5syZU1qfNav8Y/Hpp5/W2Q66iBEeSITAA4kQeCARAg8kQuCBRAg8kAiBBxJxRDS7ArvZFfSpHTt2lNbvvPPOLnVy6V588cXS+qOPPlpaP3DgQJ3toD0H27nTEyM8kAiBBxIh8EAiBB5IhMADiRB4IBECDyTCcfiKRkZGSut79uwprc+ePbvGbrrrzJkzpfVbbrll2hrH6BtT33F428O29xbTi23/yfae4m9Bp50C6I6WV7yxPSTpaUlzi6f+UtLfRcRYk40BqF87I/w5SXdLmigej0j6ru1f2X6ksc4A1K5l4CNiIiI+nPTUy5JGJX1Z0irby6bOY3uD7XHb47V1CqBjVfbS74uIUxFxTtKvJS2Z+oKI2BIRK9vZiQCge6oE/hXbC21/QdLtkn5Xc08AGlLlMtU/lLRb0llJP4mIQ/W2BKApbQc+IkaL/7sl/UVTDc0Uo6OjpfVW16Vv+vyHJrU6h2D37t3T1sqO0Uscp28aZ9oBiRB4IBECDyRC4IFECDyQCIEHEuHnsRXNmzevtP7ee++V1o8dO1Zaf/LJJ0vrzz333LS1xYsXl8770EMPldbXrFlTWu/EY489Vlpv1RumxWWqAXwWgQcSIfBAIgQeSITAA4kQeCARAg8kUuX38JB0+vTp0vqKFSs6mr/VcfoyR48eLa2vW7eutP7888+X1u+6665L7gn9gREeSITAA4kQeCARAg8kQuCBRAg8kAiBBxLhOHxD3n777V63MK2zZ8+W1icmJkrrmLkY4YFECDyQCIEHEiHwQCIEHkiEwAOJEHggEY7DN+TGG28srW/atKm0Pn/+/NL6E088MW1t165dpfOeOnWqtI7LV8sR3vZVtl+2vdP2dtuzbW+1vd/2D7rRJIB6tLNJv17S5oi4XdIxSeskDUTEKknX2l7SZIMA6tNykz4iJm87LpD0bUn/UDzeKelmSb+vvzUAdWt7p53tVZKGJP1R0pHi6Q8kDV/ktRtsj9ser6VLALVoK/C2r5b0Y0n3SDotaU5RmnexZUTElohY2c7N7QB0Tzs77WZL2ibp+xFxWNJBnd+Ml6Tlkt5trDsAtWrnsNx3JK2QtMn2Jkk/lfTXthdJukPSSIP9zVivvfZaab3V7aZbufXWW6et7du3r3Te48ePl9ZbXWIbM1c7O+3GJI1Nfs72Dkm3Sfr7iPiwod4A1KzSiTcRcVLSCzX3AqBhnFoLJELggUQIPJAIgQcSIfBAIvw8tiFPPfVUaX39+vWl9aGhocrrvummmyrP2zQugd1bjPBAIgQeSITAA4kQeCARAg8kQuCBRAg8kIgjotkV2M2uYIa6/vrrS+v3339/af3ee++dtjYwMFCpp3Z98sknpfUHH3xw2trY2Ni0NXTkYDtXmGKEBxIh8EAiBB5IhMADiRB4IBECDyRC4IFE+D18jxw6dKi0vnHjxtL6yZMnp621uhV1K9u2bSutv/HGG6V1jrX3L0Z4IBECDyRC4IFECDyQCIEHEiHwQCIEHkik5e/hbV8l6Z8lDUj6H0l3S3pb0jvFSzZGxH+XzM/v4YHmtfV7+HYCf5+k30fEq7bHJB2VNDcivtdOFwQe6Ip6LoAREU9ExKvFwwWSPpX0Ddu/sL3VNmfrATNE29/hba+SNCTpVUlfj4ivSBqUtPoir91ge9z2eG2dAuhYW6Oz7asl/VjSX0k6FhFnitK4pCVTXx8RWyRtKeZlkx7oEy1HeNuzJW2T9P2IOCzpWdvLbQ9IWivpNw33CKAm7WzSf0fSCkmbbO+R9KakZyX9l6T9EbGrufYA1InLVAOXBy5TDeCzCDyQCIEHEiHwQCIEHkiEwAOJEHggEQIPJELggUQIPJAIgQcSIfBAIgQeSITAA4kQeCCRblyA8oSkw5MeX1M814/orRp6u3R19/XFdl7U+AUwPrdCe7ydH+r3Ar1VQ2+Xrld9sUkPJELggUR6EfgtPVhnu+itGnq7dD3pq+vf4QH0Dpv0QCIEXpLtWbb/YHtP8felXvfU72wP295bTC+2/adJ79+CXvfXb2xfZftl2zttb7c9uxefua5u0tveKmmppJci4m+7tuIWbK+QdHe7d8TtFtvDkv4lIr5me1DSv0q6WtLWiPjHHvY1JOlnkv48IlbY/qak4YgY61VPRV8Xu7X5mPrgM9fpXZjr0rURvvhQDETEKknX2v7cPel6aER9dkfcIlRPS5pbPLVR52828FVJ37I9v2fNSed0PkwTxeMRSd+1/Svbj/SuLa2XtDkibpd0TNI69clnrl/uwtzNTfpRSS8U0zsl3dzFdbfyS7W4I24PTA3VqC68f69L6tnJJBExEREfTnrqZZ3v78uSVtle1qO+pobq2+qzz9yl3IW5Cd0M/FxJR4rpDyQNd3Hdrfw2Io4W0xe9I263XSRU/fz+7YuIUxFxTtKv1eP3b1Ko/qg+es8m3YX5HvXoM9fNwJ+WNKeYntfldbcyE+6I28/v3yu2F9r+gqTbJf2uV41MCVXfvGf9chfmbr4BB3Vhk2q5pHe7uO5WfqT+vyNuP79/P5S0W9IBST+JiEO9aOIioeqn96wv7sLctb30tq+UtFfSf0i6Q9LIlE1WXITtPRExavuLkn4uaZekm3T+/TvX2+76i+2/kfSILoyWP5X0gPjM/b9uH5YbknSbpNcj4ljXVnyZsL1I50esV7J/cNvFZ+6zOLUWSKSfdvwAaBiBBxIh8EAiBB5IhMADifwv2G1iilUb3I4AAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>


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

Ham/Spam Texts Dataset (UCI)
----------------------------

We will use another UCI ML Repository dataset called the SMS Spam Collection. You can 
read about it `here <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>`. 
As a sidenote about common terms, when predicting if a data point represents 'spam' 
(or unwanted advertisement), the alternative is called 'ham' (or useful information).

This is a great dataset for predicting a binary outcome (spam/ham) from a textual input.
This will be very useful for short text sequences for Natural Language Processing 
(Ch 7) and Recurrent Neural Networks (Ch 9).

.. code:: python

  import requests
  import io
  from zipfile import ZipFile

  # Get/read zip file
  zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
  r = requests.get(zip_url)
  z = ZipFile(io.BytesIO(r.content))
  file = z.read('SMSSpamCollection')
  # Format Data
  text_data = file.decode()
  text_data = text_data.encode('ascii',errors='ignore')
  text_data = text_data.decode().split('\n')
  text_data = [x.split('\t') for x in text_data if len(x)>=1]
  [text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
  print(len(text_data_train))
  print(set(text_data_target))
  print(text_data_train[1])

the output::

  5574
  {'spam', 'ham'}
  Ok lar... Joking wif u oni...
  
  
Movie Review Data (Cornell)
---------------------------
The Movie Review database, collected by Bo Pang and Lillian Lee (researchers at Cornell),
serves as a great dataset to use for predicting a numerical number from textual inputs.


You can read more about the dataset and papers using it `here <https://www.cs.cornell.edu/people/pabo/movie-review-data/>`

.. code:: python

  import requests
  import io
  import tarfile

  movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
  r = requests.get(movie_data_url)
  # Stream data into temp object
  stream_data = io.BytesIO(r.content)
  tmp = io.BytesIO()
  while True:
      s = stream_data.read(16384)
      if not s:  
         break
      tmp.write(s)
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
