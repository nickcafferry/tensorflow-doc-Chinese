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
  
  <img src="data:image/png;base64,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O42HH36YF154gXvvvZfa2lp+97vf8dJLL+U97tlnn+X4448H4IQTTuC5557r0VdVVRVDhw7llltuYc6cObz66qvs3r07ydMTEZFizJkDLS0wblywKHncuOBz1Avmw4yTVCwyYLHcpnT3ZqC5QJtnCH5RWdKeeuoppk6dyqJFi7jwwgs57rjjmDx5MpdeeikPPvggY3OXgw844AB27NgBBLW8jj76aDZs2MD06dPZsGEDRx99dI++1q1bx5o1azjvvPOor6/n5JNPTvNURUQkjDlzkkl4woyTVCwyIBX1OKRDDilufxjjx4/nO9/5DlOmTGHbtm184Qtf4KGHHmLatGncfPPNHH744QCceuqp/OhHP2Lq1KmsW7eOL3zhC/z6179m2rRp/OY3v+GSSy7p0dekSZM49dRTueGGGzjllFMAer3SJiJS0kqpHta8eTBkSHC1aciQ4LPIAJh75tbE92rSpEne0dHRZd9zzz3HRz/60ZQiKl+aVxFJTPd6WBCsbcriLbV586A5z42fhgZYvjz5eCTTzGyju08q1K6iroyJiEgGlVI9rJaW4vaLhKBkTERE0lVK9bDeeae4/SIhKBkTEZF09Vb3Kov1sAYPLm6/SAhKxkREJF2lVA+r89mOYfeLhKBkTERE0lVK9bCWLw8W63deCRs8WIv3ZcAqKhlbsmQJbW1tXfa1tbWxZMmSlCLK74orruj3sbW1tdEFIiKSlDlzYNMmePfd4DWLiVin5cvhz38G9+BViZgMUEUlY9XV1dTX1+9NyNra2qivr6e6ujrlyLq68cYb0w5BRCRZheqMhalDFkUfUYhinKhiLaX6bUnI6ny4e8lsxx13nHf37LPP9tjXl9WrV/uoUaO8sbHRR40a5atXry7q+O4WLlzo9957r7u7L1q0yO+44w4/99xzvaamxufNm7e33cknn+wLFizw0047zd3dd+3a5WeccYbX1NT4Oeec42+//XaXtp12797t559/vk+dOtXPOOMMf+ONN3zPnj1+wQUX+LRp0/zCCy/0N998M++xvbXrHks+xc6riEi/rVjhXlXlHlxrCraqqmB/mO+j6iOJc0mqjyj7KRcpzAfQ4SHym9QTrGK2KJIxd/fGxkYHvLGxsehju3vhhRd87ty57u5++umn+8KFC/2aa65xd/fPfOYz/tRTT7m7+3ve8x6/55579h7X0dHhn/70p93d/f777/fXXntt73f7J1Tf/va3fdGiRe7uftttt/kvf/lLv+mmm3zhwoXu7n7NNdd4c3Nz3mN7a9c9lnyUjIlIYsaN6/ofyM5t3Lhw30fVRxLnklQfUfZTLlKYj7DJWEXdpoTg1mRzczONjY00Nzf3WENWrCOPPJKtW7fy+uuv8/73v5+tW7fmfVD4xIkTmTVr1t7j8j0UPJ/nn3+eyZMnA3DJJZdQXV2d9yHj+fTWrnssIiKpKlRnLEwdsij6iEIU40QVaynVb0tChuejopKxzjVira2tXHvttbS2tnZZQ9ZfkydP5sYbb+Sss85iwoQJXHHFFaxZs4aFCxfufVD4gQce2OWYzoeCP/LII7z22musW7cub99HHXUU7e3tACxatIhbbrll70PGgb0PGc+nt3bdYxERSVWhOmNh6pBF0UcUohgnqlhLqX5bEjI8HxWVjLW3t9Pa2kpdXR0AdXV1tLa27k12+mv27NnceOONnHnmmVx22WV5HxTeXb6Hgudz2WWX8cQTT1BbW8sTTzzBRRddlPch4/mEbScikqpCdcbC1CGLoo8oRDFOVLGWUv22JGR5PsLcy8zKFtWaMSlM8yoiiVqxIli7Yxa8dl9UXej7qPqIQhTjRBVrUudcKhKeD0KuGbOgbWmYNGmSd3R0dNn33HPP8dGPfjSliMqX5lVERGRgzGyju+e/9bWfirpNKSIiIpI1SsZERESKkaWirpWkjOdsSNoBiIiIlIyVK4OHgu/aFXzevHnfQ8LDPsIpij4qTZnPma6MiYiIhHX11fsSgk67dgX7k+yj0pT5nFVeMpbQZc6BPOw77LEDGUNERPohS0VdK0mZz1llJWOdlzk3bw4egtB5mTOGhGwgD/sOe6weKC4ikrAsFXWtJGU+Z5WVjMVwmfP666/nvvvuA+CGG27g7rvvBqC2trZLu9raWv7+7/+eT33qUwDs3r2bGTNmcPzxx3PhhReyaNGiLm07ffOb3+Tqq69m2rRpfOITn2Dbtm152+3Zs4cLLriAk046iTPPPJNdu3axc+dOTj/9dGpqarj00kv7fY4iIpKTpaKulaTM56yykrEYLnPOnj2bhx56CIC1a9cyc+bMvO02bNjAiSeeyMMPPwwEz5w87LDDWL9+Pb/97W/52te+1usYv/3tb1m7di2zZs1i9erVedu0tLRwzDHHsH79es4991yeeeYZXn75ZS6//HJWrVrFpk2beOWVV/p9niIiQrBYvKUFxo0Ds+C1paW4ReRR9FFpynzOKuvXlGPHBrcm8+3vp+4PCh8xYkTedt0fzn3ooYeyceNGpk2bxle+8pU+x7j44otzYY7lrbfeytvm+eef59xzzwXY+9ijzZs3c8stt3D77bfz6quvsnv37mJPT0REupszZ+BJQBR9VJoynrPKujIW02XO/R8U3pvuD+f+6U9/SmNjI4899hhzCvzl6i3B21++B4rfeuutnHfeedx1112h+hARKWthfsAVRRvVIeufSjznnMq6MtaZ9Fx9dXBrcuzYIBEbYKY9e/ZsTjrpJDbnu+rWi2OPPZYZM2Zw00038cEPfpCvf/3rTJw4sd8xXHbZZXzuc5+jtraWkSNHsnLlSh5//HHmzZvHzTffDMBLL73E+PHj+z2GiEjJClOnKoo2qkPWP5V4zvvRsylT8r3vfY+77rqLoUOHMnToUBYsWNBj0X+aSnVeRUTyGj8+/zKVceNg06bo2oTpI4pYy02ZnnPYZ1MqGZO8NK8iUlYGDQpKGnVnBu++G12bMH1EEWu5KdNz1oPCRUREOoWpUxVFG9Uh659KPOf9lEUyVkpX90qB5lNEyk6YH3BF0UZ1yPqnEs95PyWfjA0fPpwdO3YogYiIu7Njxw6GDx+edigiItEJU6cqijaqQ9Y/lXjO+yn5NWNvv/02W7duZc+ePSlFVX6GDx/OYYcdxtChQ9MORUREpGSFXTOGu0e+AQcBDwGPAPcCw3ppdyvwGPD1MP0ed9xxLiIiGbJihfu4ce5mweuKFfG0CdOHpCOpP5sS/DsAdHiYvClMo2I3YB5wau59M3BWnjazgDty728DjijUr5IxEZEMWbHCvaoq+E9J51ZV1fU/klG0CdOHpCOpP5sS/TsQNhmL/Talmf0bsNTdN3Tb/x3gp+7+EzO7ADjA3W/vq698tylFRCQlpVS7S+KR1J9Nif4dyERpCzM7ETi4eyKWMwJ4Kff+VeCQXvqYa2YdZtaxffv2mCIVEZGibdlSeH8UbcL0IelI6s+mzP8OxJaMmdkHgJuAz/fSZCdwQO79gb3F4u4t7j7J3SeNHj06+kBFRKR/Sql2l8QjqT+bMv87EEsyZmbDgLuBq9y9twc2bgROyr0/BtgURywiIhKTUqrdJfFI6s+m3P8OhFlYVuwGNACvAWty2zXAwm5t3gc8BTQBzwEHFepXC/hFRDJGv6YU/ZqyV2RlAX9fzOxg4FRgrbtvK9ReC/hFRESkVGRiAX8h7v6au7eGScRERKRErVwZ/Bpu0KDgdeXKnm3mzYMhQ4Lq60OGBJ/jGKfSaE5KwpC0AxARkTK2ciXMnQu7dgWfN28OPsO+R93MmwfNzfuOeeedfZ+XL49unEqjOSkZJf84JBERybAw9aGGDAkSsO4GD4Y//zm6cSqN5iR1JXGbUkREylyY+lD5ErG+9vd3nEqjOSkZSsZERCQ+YepDDR6cv01v+/s7TqXRnJQMJWMiIhKfMPWhOtcxddfb/v6OU2k0JyVDyZiIiMRnzhxoaQnWKZkFry0tXReQL18ODQ37roQNHhx8Drt4P+w4lUZzUjK0gF9EREQkBlrALyIiIlIClIyJiEheS5Ys4cWzz+5SjPXFs89myZIl+xpFUawVslOcNMz5RBFrVvrI0jiVLMwzk7Ky6dmUIiLJ2XLWWf4uuO+3vQu+5ayzggYNDV2+27s1NBQ30IoV7lVVXfuoqkr+2YNhzieKWLPSR5bGKVOUwrMpi6U1YyIiCSpUjDWKYq2QneKkYc4niliz0keWxilTYdeMKRkTEZH8zHr/zr3w92ENGpS/vRm8+274fgYqzPlEEWtW+sjSOGVKC/hFRGRgChVjjaJYK2SnOGmY84ki1qz0kaVxKpySMRERyevFM86g+zURz+0HoinWCtkpThrmfKKINSt9ZGmcShdmYVlWNi3gFxFJzuLFi4PF+oMHBwu3Bw/2LWed5YsXL97XqKGhy/dFL97vtGKF+7hx7mbBa1oLxMOcTxSxZqWPLI1ThohqAb+ZjQA+AxwLDAdeBB5092dizxS70ZoxERERKRWRrBkzs88CNwPbgeuA+cB9wCwzu9nMDowiWBERKVKJ1H5asmQJbW1tXfa1tbV1rVUmUuF6TcbM7EPAaHe/yN0fdvf/dvc33f15d78WWAZcmFikIiISWLkyWMe0eXPwS7fNm4PPGUzIqqurqa+v35uQtbW1UV9fT3V1dcqRiWRH6NIWZnY9cCLQ+dtfd/dT4gosH92mFBGh5Go/dSZgDQ0NNDc309raSl1dXdphicQu7G3KIUX0ORk409139T8sEREZsC1bitufsrq6OhoaGrjuuutobGxUIibSTTGlLQx4wsxWm1mbma2OKygREelDidV+amtro7m5mcbGRpqbm3usIROpdMUkY9uA09z9FHevS/oWpYiI5JRQ7afOW5Stra1ce+21tLa2dllDJiLFJWOHAnfkroyt1pUxEZGUzJkDLS3BGjGz4LWlJdifMe3t7V3WiNXV1dHa2kp7e3vKkYlkRzEL+A8BjgMeIihz8X/dfUeMsfWgBfwiIiJSKuJ4NuVdEPyEEngWuLOfsYmIyABEUbsrS/W/shRLYkqkTpwko5hkbJi7/wTA3X8AjIgnJBER6UsUtbuyVP8rS7EkooTqxEkyirlN+V1gMPA4UA3g7vPiC60n3aYUEQlEUbsrS/W/shRL7EqsTpz0X+S3Kd39b4CfAKOAh5JOxEREZJ/9a3c1NDT0K3GJoo+oZCmW2JVYnTiJX1+PQxpvZn+7/z53/7G7/6O7/9jMPmJmc+MPUUREuouidleW6n9lKZbYlVidOEmAu/e6ARcAK4HTgIOAocCRwDeAfwYO7Ov4qLfjjjvORUQq3erVq33UqFG+evXqvJ+T6iMqWYolEStWuFdVuQcrxoKtqirYL2UF6PAQ+U2ftynd/YfAl4APAtcA3wZmAfe6+xfdfWdMOaKIiPQiitpdWar/laVYElFCdeIkGaEX8GeBFvCLiIhIqYijzlh/gjjEzNb18f2hZrbVzNbkttFxxiMiEqek6mWNGDGCqVOndtk3depURozYV3GoUCxRxDpz5kyampq67GtqamLmzJldG4apqaW6W1LJwtzL7M8GHAz8FHiijzazgIawfWrNmIhkWVJrn6ZMmeKAT5kyJe/nMLFEEeuyZcvczHzZsmV5P7t7uPVRWkMlZYqQa8biTMbeR7Dof00fbZYAG4EngEWF+lQyJiJZ15nUNDY2xroIvTMBGzZsWI9ELGwsUcTamYDV1NT0TMTc3ceN65pkdW7jxhXXRqQEpZ6M7R2g72SsDngvQTHZNuDjedrMBTqAjrFjx8Y0XSIi0WlsbHTAGxsbYx2nMxEbNmxYv2OJItaamhoHvKampueXZvkTLbPi2oiUoLDJWKxrxkL4D3f/H3d/B/gVcET3Bu7e4u6T3H3S6NFaUiYi2ZZUvaypU6fy1ltvMWzYMN56660ea8jCxBJFrE1NTaxfv56amhrWr1/fYw1ZqJpaqrsllS5MxjaQjb6vjK0B/hdQBTwDTOirL92mFJEs05oxrRkT2R9R36YE/gq4kqDg6zeAb4Q8bk3u9RTgb7p9Vwc8Dzzd/bt8m5IxEcmyxYsX512XtXjx4kjHqaqq6rFGbMqUKV5VVRU6lihinTFjRo81YsuWLfMZM2Z0bbhiRbD+yyx4zZdkhWkjUmLCJmPFPCj8GeBbwIv7XVV7tP/X5IqnOmMiIiJSKuKoM/YKcJe7P9q59T88EZHkJFX/Kwqha3cVUOicw4xTSvOWKNVEk4gVTMbM7GIzuxh4Cmgzs4b99omIZF51dTX19fV7E4u2tjbq6+uprq5OObKepk+fzoIFC/YmSk1NTSxYsIDp06cX1U+hcw4zTinNW2JWroS5c2Hz5mB12+bNwWclZDIQhe5jAp/rZbs4zH3QKDetGROR/kqq/lcUCtbuCqnQOYcZp5TmLRGqiSZFIIYF/CO7fa4Pe2xUm5IxERmIpOp/RaHP2l1FKHTOYcYppXmLnWqiSRHCJmPFrBm7u9vnL/f7cpyISMKSqv8VhYK1u0IqdM5hximleUuEaqJJHApla8DJwDXA79hX1mIxcF+YbC/KTVfGRKQ/kqr/FYVQtbtCKHTOYcYppXlLjGqiSRGI8MrYJoLirH8CHs1t9wCzo04MRUTi0N7eTmtrK3V1dQDU1dXR2tpKe3t7ypH1tGrVKpYuXcr8+fMBmD9/PkuXLmXVqlVF9VPonMOMU0rzlpg5c6ClBcaNA7PgtaUl2C/ST8XUGftbd/9OzPH0SXXGREREpFSErTM2JERHnSUs/rt7OQt3/5d+xiciIiIihCv6arntQuAkYDhwIqA6YyIyYFkpLHrUUUcxb968LvvmzZvHUUcdtfdzoUKpYQqpFmoTZj6iaiMiGRFmYVnuVubPu31eHfbYqDYt4BcpP1lZJN7Q0OCANzQ05P3sXnjRe5hF8YXahJmPqNqISLyIoc7Yj4AmgoKvS4D7wx4b1aZkTKQ8ZaWwaGcCdvjhh/dIxDoVKpQappBqoTZh5iOqNiISnziSsSFAPfBV4LPAsLDHRrUpGRMpX1kpLNqZiB1++OG9tilUKDVMIdVCbcLMR1RtRCQekSdjWdiUjImUp6xcwdGVMRGJkpIxESkJWVnbpDVjIhK1yJIxoCn32gaszm1tWsAvIlFYvHhxjwRh9erVvnjx4kTjmDBhQo8rYQ0NDT5hwoS9n2fMmJH3StiMGTNCfR+mTZj5iKqNiMQrbDIWuuhrFqjoq4iIiJSKsEVfi3lQuIhISUqqLldW+sjSOCJSWOhkzMzazexfzOwrZlZjZgfGGZiISFSqq6upr6/fm3y0tbVRX19PdXV1UW2iGCeJPrI0joiEEOZeZu5W5kiCqvs/BnYBz4U9NqpNa8ZEpL+S+vVhVvrI0jgilYoY6oz9EbgfmAuMC3tclJuSMREZiKTqcmWljyyNI1KJ4kjGDgb+GngAeAv4fdhjo9qUjIlIf+nKWHrjiFSqOJKxR4BvAecDR4Y9LspNyZiI9EdSdbmy0keWxhGpZGGTsdAL+N39NHe/0t3/1d3/a0AL1UREEtTe3k5rayt1dXUA1NXV0draSnt7e1FtohgniT6yNI6IFKY6YyIiIiIxUJ0xkTKl+lA9FZoTzZmIZJmSMZESo/pQPRWaE82ZiGSZblOKlKDOZKKhoYHm5uYua38qVaE50ZyJSNJ0m1KkjNXV1dHQ0MB1111HQ0ODkgoKz4nmTESySsmYSAlqa2ujubmZxsZGmpube6yHqkSF5kRzJiJZNaRQAzNrA7rfyzTA3f2UWKISkV513m7rvM1WV1fX5XMlKjQnmjMRybKCV8bcvc7dT+m21SkRE0mH6kP1VGhONGcikmVawC8iIiISg8gX8JvZYDOrNrNpue2zAwtRRLJs5syZNDU1ddnX1NTEzJkzE+0jqRphYcZRvTIRiUMxC/j/DbgcuB74MnBpoQPM7BAzW9fH90PN7Mdm9gsz+3wRsYhIzKZPn86CBQv2JlNNTU0sWLCA6dOnJ9pHUjXCwoyjemUiEoswD7DM3cpcBwwG7un8XKD9wcBPgSf6aDMf+Gbu/U+A9/bVpx4ULpKsZcuWuZl5TU2Nm5kvW7YslT46H2Ld2NgY68Osw4yTVCwiUvoI+aDwYpKxlcBngR8AVwFPFWj/PuAgYE0fbR4A/ir3/kqgLk+buUAH0DF27NhYJ01EeqqpqXHAa2pqUu2jsbHRAW9sbOx3H1GNk1QsIlLawiZjxdymvAhYBcwDXgHqC1xxe93d/1SgzxHAS7n3rwKH5Omnxd0nufuk0aNHFxGuiAxUU1MT69evp6amhvXr1/dY/5VUH0nVCAszjuqViUjkwmRsQXLHyG6f60Me19eVsfuBMb7vluWFffWl25Qiyem8vdh5W7H756T66Lwt2Hk7sPvnqIQZJ6lYRKQ8EMOVsbu7ff7yAPNAgI3ASbn3xwCbIuhTRCKwatUqli5dyvz58wGYP38+S5cuZdWqVYn2kVSNsDDjqF6ZiMShYJ0xMzsZqAU+B9yR2z0CmODu5xQcwGyNu9ea2SkE68O+u9934wgW7q8CpgAnuPs7vfWlOmMiIiJSKsLWGSv4OCSCq1VrgHOAR3P7dgO/ChOIu9fmXlcDq7t9t9nMTiW4OvaNvhIxERERkXJUMBlz983AZjO73d0fLdS+WO7+e6A16n5FRERESkGYK2OdbjKzMwl+8fgssDmXSImIiIhIPxWzgP9fgTrgi7lrJpPoAAALCklEQVTjVsQSkYiIiEgFKSYZG+3ufwfsdPdfFHmsiIiIiORRTEL1GzO7DfgLM7sG+K+YYhIRERGpGKHXjLn7XDM7G3g+t10bW1QiIiIiFSL0lTEzG5Rr/zbQd3EyEREREQmlmNuUPwROAd4AZhI8OFxEREREBqCY0hYfdPe9Dwc3Mz0dV0RERGSAiknGdpnZlQTPk5wM/MnMprn72nhCExERESl/xdym/CXwHoJnSA4heBxSbQwxiYiIiFSMYn5N+X/iDERERESkEqlwq4iIiEiKlIyJiIiIpEjJmIiIiEiKlIyJiIiIpEjJmIiIiEiKlIyJiIiIpEjJmIiIiEiKlIyJiIiIpEjJmIiIiEiKlIyJiIiIpEjJmIiIiEiKlIyJiIiIpEjJWBkaMwbMem5jxqQdmYiIiHSnZKwMvfJKcftFREQkPUrGRERERFKkZExEREQkRUrGRERERFKkZExEREQkRUrGytAhhxS3X0RERNIzJO0AJHrbtqUdgYiIiISlK2MiIiIiKYotGTOzW83sMTP7ei/fDzGzLWa2Jrd9LK5YpCsVhRUREcmOWJIxM5sFDHb3E4EPm9kReZp9HLjL3Wtz23/GEYv0pKKwIiIi2RHXlbFaoDX3/hHgpDxtTgDONLPHc1fRtH5NREREKk5cydgI4KXc+1eBfL/jawemu/tkYCgwM19HZjbXzDrMrGP79u2xBCsiIiKSlriSsZ3AAbn3B/YyztPu/nLufQeQ71Ym7t7i7pPcfdLo0aOjj1REREQkRXElYxvZd2vyGGBTnjZ3mtkxZjYYOAd4KqZYRERERDIrrmTsPuAiM2sC6oFfm9nCbm2uBe4EngQec/dVMcUi3agorIiISHbEsmje3V83s1rgVGCJu2+j25Uvd3+G4BeVkjAVhRUREcmO2OqMuftr7t6aS8SEaOp75Tu+cws7ThRxqFaZiIhINFSBP0FJ1fcqNE4UcahWmYiISDSUjImIiIikSMmYiIiISIqUjImIiIikSMmYiIiISIqUjCUoqfpehcaJIg7VKhMREYmGHs6doCjqe7kPfJwo4lCtMhERkWjoyliCBg/OX5tr8ODg+yhqiIWhGmEiIiLZoWQsQe++W9z+fFQjTEREpLwoGRMRERFJkZIxERERkRQpGRMRERFJkZIxERERkRQpGUvQoF5mu7f9+ahGmIiISHlRnbEEvfNO399HUUMsDNUIExERyQ5dGSOauluFaohB4TpiYeqMRdFGtcpERESyQ8kY0dTdiqKGWFJUq0xERCQ7lIyJiIiIpEjJmIiIiEiKlIyJiIiIpEjJmIiIiEiKlIwRTd2tKGqIJUW1ykRERLJDdcaIpu5WoRpiULiOWJg6Y1G0Ua0yERGR7MjgdRsRERGRyqFkLKQkipyGGSNM0VcREREpHUrGQkqiyKkKqYqIiFQeJWMiIiIiKVIyJiIiIpIiJWMiIiIiKVIyJiIiIpIiJWMhJVHkVIVURUREKo+KvoaURJHTMGOEKfoqIiIipSO2K2NmdquZPWZmXx9IGxEREZFyFksyZmazgMHufiLwYTM7oj9tRERERMpdXFfGaoHW3PtHgJP62UZERESkrMWVjI0AXsq9fxXItwQ9TBvMbK6ZdZhZx/bt2yMPVERERCRNcS3g3wkckHt/IPmTvjBtcPcWoAXAzLab2eZoQ+1hFPDHmMeoRJrXeGhe46F5jYfmNR6a13hEMa/jwjSKKxnbSHDbcQNwDPBCP9t04e6jI4wxLzPrcPdJcY9TaTSv8dC8xkPzGg/Nazw0r/FIcl7jSsbuA9aZ2V8AM4ALzGyhu3+9jzYnxBSLiIiISGbFsmbM3V8nWKC/Aahz96e6JWL52vwpjlhEREREsiy2oq/u/hr7fi3Z7zYpaEk7gDKleY2H5jUemtd4aF7joXmNR2Lzaq6S7iIiIiKp0bMpRURERFKkZEykRJnZB8zsVDMblXYsIiLSf0rG9mNmh5jZurTjKCdmdpCZPWRmj5jZvWY2LO2YyoGZHQw8CEwG2sws9rIvlST3b8Gv0o6jXJjZEDPbYmZrctvH0o6pnJjZcjP7dNpxlAsza9jv7+qTZvbPcY+pZCwn9x+37xM8GUCiMwdocvfTgG3A6SnHUy4+Dsx39+uBh4FPphxPuVnKvqLUMnAfB+5y99rc9p9pB1QuzKwGGOPuP047lnLh7s2df1eBdcD34h5Tydg+7wDnA6+nHUg5cffl7v6z3MfRwB/SjKdcuPuj7r7BzKYRXB17LO2YyoWZnQK8QfA/DxKNE4AzzexxM7vVzGL7JX8lMbOhBInCJjM7O+14yo2ZHQoc4u4dcY+lZCzH3V9XrbP4mNmJwMHuviHtWMqFmRnB/0C8BrydcjhlIXcbvRG4Mu1Yykw7MN3dJwNDgZkpx1MuLgaeBZYAk83s8pTjKTdfBpqTGEjJmMTOzD4A3AR8Pu1YyokHvgw8DZyVdjxl4kpgubv/d9qBlJmn3f3l3PsO4Ig0gykjxwIt7r4NWAHUpRxP2TCzQQTzuSaJ8ZSMSaxyVxruBq5y97gf8l4xzOyrZnZx7uP7ASUP0ZgOfNnM1gCfMLNbUo6nXNxpZseY2WDgHOCptAMqE78FPpx7PwnQv7HRqQF+6QkVY1XR127MbE1u0Z5EwMwagEXs+8e32d3/NcWQykLuByetwHuAZ4AvJ/WPRqXQvwXRMbOJwA8AAx5w96tTDqksmNl7gduAQwhu/57n7i+lG1V5MLNFQIe7/yiR8fTvt4iIiEh6dJtSREREJEVKxkRERERSpGRMREREJEVKxkREEmJmHyqi7YcLtxKRcqBkTEQkAWb2WaCYKumfNrML44pHRLJDyZiIpM7MxptZbci2a6JsF7KvLvGZ2SVmdkkRx78POMvdbwx7jLt/GzgjV75ARMqYkjERyYLxQG3KMfRlPAOL72zg+/04bgVBkVQRKWN6WKuIxMLMvgkcD1QB24ELgHeAFuDI3L7zgb8BLgXen7v6NBt4FbgTGAf8kaCYZb+fv5l7jmf3cS8CjsltY4B64NcECdBfAi8RVDTfnCc+gGPMbHXnse7+TB8hfBK4dr9Yvgt8guCZohcAPwR+T1C8E+Bxd78K2EDwrMw7+3vuIpJ9ujImInFa5+4nA68QXB06Gxia27cFOCN3O+4K4A53r3X37cBI4N+Bk4HXCZKZgegxbm5/NfAp4FsEz/c8GPigu58AfMjd5/cSX75j+3IAsCv3/tPAEHefCiwFjsvt/xpweC7WE3L7dueOFZEypmRMROK0Mff6NMGtvgnAibn1XNPYdyWou7eBMwmea/phBp6Q9DbuXbkrbluAYQQJ03vM7JfAygJ9dj+2L1uAzl9SHgU8DuDuDwIP5d5vAn7v7jsJHhtE7pgXQ5yfiJQwJWMiEqfJuddjCR5q/ALww9wzH68Ans19v5vgdmbnbbxZBM/cnEVwu3Cgehv3jTzx3uvux7v7sv32d48v37F9eZDg1ijA8wRX1TCzOcB1fRxXnztWRMqYkjERiVN17mrU+wmSigeAvzCzR4GFBOuxAH4FTDCzdQRJyy9yr+uBDwCHDjCO3sbt7nlgvpm1mdk9ZlbTS3xFcfengcPM7Ejgx4Cb2VqCdWt5f2GZa3to7lgRKWN6ULiIxCK3gH+Nu69JOZTQzOwM4B+Atwiuhv3A3X8YUd/vAy5095tDtv9SbvzXoxhfRLJLyZiIiIhIinSbUkRERCRFSsZEREREUqRkTERERCRFSsZEREREUqRkTERERCRFSsZEREREUvT/AVmt+ocTUuaiAAAAAElFTkSuQmCC
  ">


Low Birthrate Dataset (Hosted on Github)
----------------------------------------
The 'Low Birthrate Dataset' is a dataset from a famous study by Hosmer and Lemeshow 
in 1989 called, "Low Infant Birth Weight Risk Factor Study". It is a very commonly 
used academic dataset mostly for logistic regression. We will host this dataset on the
public Github `here <https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat>`_


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


  
Housing Price Dataset (UCI)
-----------------------------
We will also use a housing price dataset from the University of California at Irvine 
(UCI) Machine Learning Database Repository. It is a great regression dataset to use. 
You can read more about it `here <https://archive.ics.uci.edu/ml/datasets/Housing>`_


.. code:: python

  >>> import requests
  >>> housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
  >>> housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
  >>> housing_file = requests.get(housing_url)
  >>> housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
  >>> print(len(housing_data))
  >>> print(len(housing_data[0]))
  
the output::

  506
  14
  
MNIST Handwriting Dataset (Yann LeCun)
-------------------------------------
The MNIST Handwritten digit picture dataset is the Hello World of image recognition. 
The famous scientist and researcher, Yann LeCun, hosts it on his webpage `here <http://yann.lecun.com/exdb/mnist/>`_. 
But because it is so commonly used, many libraries, including TensorFlow, host it 
internally. We will use TensorFlow to access this data as follows.

If you haven't downloaded this before, please wait a bit while it downloads

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
