[![Build Status](https://travis-ci.com/nickcafferry/tensorflow-doc-Chinese.svg?branch=master)](https://travis-ci.com/nickcafferry/tensorflow-doc-Chinese)
[![codecov](https://codecov.io/gh/nickcafferry/tensorflow-doc-Chinese/branch/master/graph/badge.svg)](https://codecov.io/gh/nickcafferry/tensorflow-doc-Chinese)
![deployment](https://github.com/nickcafferry/tensorflow-doc-zh/workflows/deploy/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tensorflow-doc-chinese/badge/?version=latest)](https://tensorflow-doc-chinese.readthedocs.io/zh_CN/latest/?badge=latest)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.2-brightgreen.svg)](https://github.com/tensorflow/tensorflow)
[![Huawei Cloud](https://img.shields.io/badge/platform-huawei%20cloud-blue)](https://auth.huaweicloud.com/authui/login.html?service=https%3A%2F%2Fconsole.huaweicloud.com%2Fconsole%2F%3Flocale%3Dzh-cn#/login)


# [TensorFlow机器学习指南](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-cookbook)

目录
=================

  * [第一章: 如何从TensorFlow开始](#ch-1-getting-started-with-tensorflow)
  * [第二章: TensorFlow方式](#ch-2-the-tensorflow-way)
  * [第三章: 线性回归](#ch-3-linear-regression)
  * [第四章: 支持向量机](#ch-4-support-vector-machines)
  * [第五章: 最近邻法](#ch-5-nearest-neighbor-methods)
  * [第六章: 神经网络](#ch-6-neural-networks)
  * [第七章: 自然语言处理](#ch-7-natural-language-processing)
  * [第八章: 卷积神经网络](#ch-8-convolutional-neural-networks)
  * [第九章: 递归神经网络](#ch-9-recurrent-neural-networks)
  * [第十章: TensorFlow应用技巧](#ch-10-taking-tensorflow-to-production)
  * [第十一章: TensorFlow的更多功能](#ch-11-more-with-tensorflow)

中文版完整内容请查看[TensorFlow机器学习指南](https://tensorflow-doc-chinese.readthedocs.io/zh_CN/latest/?badge=latest)

全球主要材料数据库
================

过去若干年里，全世界范围内材料研究学者们通过实验测量和计算模拟积累了大量的材料数据，由此建立了大量可用于材料研究的涵盖材料结构与性能的数据库。

实验测量作为沿用至今的材料科学研究关键手段之一，对材料的研发起着至关重要的作用。科学工作者们通过对文献中实验测量数据的收集，建立了一些材料数据库，其中包含了化学组成、材料结构、文献引用等基本信息。

CSD（数据来源：实验测量）

网址：https://www.ccdc.cam.ac.uk/

特点：小分子有机物和金属有机化合物晶体结构数据

剑桥结构数据库(cambridge structural database，CSD) 由英国剑桥大学Kennard 等在1965年创建，从文献中收录了115万种小分子有机物和金属有机化合物晶体结构数据，其中包含了晶胞参数、原子坐标和引用文献等。

ICSD（数据来源：实验测量）

网址：https://icsd.products.fiz-karlsruhe.de/

特点：无机晶体结构数据

德国波恩大学Bergerhoff 等在1983年创建了无机晶体结构数据库(inorganic crystalstructure database，ICSD)来作为剑桥结构数据库的补充，收录了1913年以来出版的21万多条实验表征的无机晶体结构详细信息，包含化学名称、化学式、矿物名、晶胞参数、空间群、原子坐标、原子占位及文献引用等。

Pauling file（数据来源：实验测量）

网址：https://www.paulingfile.com/

特点：无机晶体材料、相图和物理性能

1995年，日本科学技术厅等单位合作组建了Paulina Film项目，收集了从1900年至今超过35000种出版物中的无机材料数据，包含了35万个晶体结构、5万个相图和15万条物理性能。

材料学科领域基础科学数据库（数据来源：实验测量）

网址：http://www.matsci.csdb.cn/

特点：金属材料和无机非金属材料

为了有效地应用和积累科学数据，我国在1987年由中国科学院牵头正式启动科学数据资源建设。其中，中国科学院金属研究所承建的“材料学科领域基础科学数据库”，(http://www.matsci.csdb.cn/)拥有金属材料数据6万余条和无机非金属材料数据1万余条，涵盖了材料的热学、力学和电学等各种性能。

国家材料科学数据共享网（数据来源：实验测量/计算模拟）

网址：http://www.materdata.cn/

特点：各类材料体系数据

2001年我国开始逐步启动科学数据共享工程，其中北京科技大学建设的“国家材料科学数据共享网”(http://www.materdata.cn/)汇集了全国30余家科研单位包括有色金属材料、有机高分子材料和能源材料等超过60万条材料科学数据。虽然这些基于实验测量的材料数据库记录的数据可靠且直观，但是获得这些数据的成本高昂。

随着计算机算力的提升，材料研究模式开始以“经验试错法”到基于“材料基因”设计方法转变，期间催生了许多高通量材料计算平台和数据库。

Materials Project（数据来源：ICSD/计算模拟）

网址：https://materialsproject.org/

特点：无机晶体材料、分子、纳米孔隙材料、嵌入型电极材料和转化型电极材料以及材料性能

劳伦斯伯克利国家实验室Ceder等在2011年创立Materials Project 数据库，存储了75 万多种材料，涉及无机化合物、分子、纳米孔隙材料、嵌入型电极材料和转化型电极材料以及包括9万多条能带结构、弹性张量、压电张量等性能的第一性原理计算数据。

AFLOWlib（数据来源：ICSD/计算模拟）

网址：http://aflowlib.org/

特点：无机晶体材料、二元合金、多元合金以及材料性能

2012年，杜克大学Curtarolo等发布了AFLOWlib计算材料数据库，存储了包括无机化合物、二元合金与多元合金等超过356万种材料结构和7亿条第一性原理计算的材料性能数据，是诸多数据库中数据量最大的一个。

OQMD（数据来源：ICSD/计算模拟）

网址：http://www.oqmd.org/

特点：无机晶体材料以及热力学和结构特性

2013 年，西北大学Wolverton等推出了开放量子材料数据库(open quantum materials database，OQMD)，通过DFT计算了102万种材料的热力学和结构特性，其中以钙钛矿数据居多。

以上三个数据库的数据都是从无机晶体结构数据库衍生而来，不同之处在于其所包含的虚拟材料的数量。

相比于国外，国内的材料计算数据库发展较晚。

材料基因工程数据库（数据来源：实验测量/计算模拟）

网址：https://www.mgedata.cn/

特点：各类材料体系数据

2016年，北京科技大学牵头建立的“ 材料基因工程专用数据库” (http://www.mgedata.cn/)，包含超过76万条催化材料、特种合金及其材料热力学和动力学等数据。

Atomly（数据来源：ICSD/计算模拟）

网址：https://atomly.net/#/matdata

特点：无机晶体结构以及材料性能

2020年，中国科学院物理研究所等单位创建的Atomly数据库(http://atomly.net/#matdata)，包含从ICSD数据库和DFT计算得到的18万个无机晶体结构并计算其详细的电子结构信息以及热力学相图。

这些基于计算的数据库拥有着庞大的数据量，使得数据驱动的材料研究得到迅速的发展。

电化学储能材料的研发需要考虑离子输运性质、能量密度、充放电速率等特定的材料性能，上述通用数据库往往不能满足这些需求。因此，专门为电化学储能材料建立的数据库开始被研究与使用。

电池材料离子输运数据库（数据来源：ICSD/计算模拟）

网址：http://e01.iphy.ac.cn/bmd/

特点：无机晶体材料以及离子输运性能

中国科学院物理研究所在2018年推出了电池材料离子输运数据库(http://eol.iphy.ac.cn/bmd/)，采用键价方法计算得到了2万多条无机晶体化合物离子迁移势垒数据，可快速筛选已知结构化合物中离子迁移势垒较低的潜在快离子导体。

电化学储能材料高通量计算平台（数据来源：ICSD/计算模拟）

网址：https://matgen.nscc-gz.cn/solidElectrolyte/

特点：无机晶体材料以及离子输运性能和机器学习描述符

上海大学施思齐课题组于2020年发布了电化学储能材料高通量计算平台(https://matgen.nscc-gz.cn/solidElectrolyte/)，集成了晶体结构几何分析(CAVD)、键价和计算(BVSE)、多精度融合算法和相稳定性计算等程序，并基于CAVD和BVSE构建了包含2.9万条数据的离子输运特性数据库，能够为下游的机器学习任务提供相应的学习样本，如下图所示。

图片
电化学储能材料高通量计算平台总览
来源：施思齐, 涂章伟, 邹欣欣, 等. 数据驱动的机器学习在电化学储能材料研究中的应用[J]. 储能科学与技术, 2022, 11(3): 739.
