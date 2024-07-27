# 蓝山考核
# 第一部分  
## 1.反向传播的推导：  
![AF7660C8D8EE59CB088AE59F22366B8D](https://github.com/user-attachments/assets/089f60de-2e09-42a5-829d-11533da49d22)
![C60BCDEF6F465AE171E9519E7FDE1479](https://github.com/user-attachments/assets/1e76fa3a-68b9-4303-a146-a705b9079ae2)
## 2.神经网络结构自定义：    
我用的是pytorch架构，用简单的乘法数据集构建神经网络，并利用张量对象操作和梯度值计算更新网络权重。  
输入层：  
输入数据维度为2（简单乘法）。  
隐藏层：  
包含一个全连接层（nn.Linear(2, 8)），将输入层的2个特征映射到8个隐藏单元。  
隐藏层使用ReLU（Rectified Linear Unit）激活函数，这是非线性变换，用于引入非线性到模型中，使网络能够学习更复杂的模式。  
输出层：  
包含另一个全连接层（nn.Linear(8, 1)），将隐藏层的8个单元映射到1个输出单元。  
损失函数：  
使用均方误差（Mean Squared Error，MSE）损失函数，适合回归任务，衡量预测值与真实值之间的差异。  
优化器：  
使用随机梯度下降（Stochastic Gradient Descent，SGD）优化器，学习率我设置为0.001，用于更新网络的权重以最小化损失函数。  
通过生成器循环获取的网络参数   
![通过生成器循环获取的网络参数  ](https://github.com/user-attachments/assets/18914834-86a9-44ec-a47e-f174bc36e849)
损失随 epoch 的变化情况可视化  
![损失随 epoch 的变化情况可视化](https://github.com/user-attachments/assets/d9602c05-5d62-41d7-b2ea-a16f077a9d43)
# 第二部分  
数据集我选的是讯飞AI开发者大赛的‘车载相机图像的目标检测挑战赛’数据集  
![F65853094CF69F65057FE72760347341](https://github.com/user-attachments/assets/c73591c8-1bdc-45b2-bb64-b81f9fd05eb0)
[车载相机图像的目标检测挑战赛]（https://challenge.xfyun.cn/topic/info?type=vehicle-mounted-camera）  
运行终端截图如下：    
![148D558337CE953DD8A345C8C400CF18](https://github.com/user-attachments/assets/784ec3f0-8330-41a0-903c-124d65c6c5a2)
输出结果如下：  
![923299096ECE73C7B87D352FF412AB6D](https://github.com/user-attachments/assets/ee1cbc55-1061-478a-842c-2bcc3685b067)  
分数提交效果初步如下：  
![9DFC78E558D3472F0257BCF4D9F02130](https://github.com/user-attachments/assets/ceed9590-917f-4e7a-a787-de3a245fb3fd)
在小目标检测上，YOLOv10表现逊于YOLOv8，并且对于复杂场景，YOLOV10处理效果不是很好（唯一的优点恐怕就是速度快）（爆改后可能有不错反响）  
# 第三部分  
对于推荐系统，经过初步学习，以电影数据为例的话可以使用两种方式来实现：  
1.基于文本CNN的推荐。  
2.基于矩阵分解的协同过滤的推荐。  
我选的数据集是：（因为这个数据集比较小） 
ez_douban数据集[ez_douban]（https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ez_douban/intro.ipynb）  
而这个数据集的rating并没有具体的评论，所以用文本CNN可能不是很好，我选择的是矩阵分解的协同过滤算法来实现（同样采用pytorch框架）  
## 数据预处理：  
从CSV文件中读取用户评分数据和电影信息。  
将电影ID映射到连续的整数索引，便于后续处理。  
合并数据框，只保留用户ID、电影索引和评分。  
创建用户和电影数量的统计变量。  
构建用户-电影评分矩阵，其中每一行代表一部电影，每一列代表一个用户，值为用户对电影的评分。  
## 归一化处理：  
对评分矩阵进行归一化，去除用户评分偏好的平均影响，使得模型更关注于评分差异而非绝对值。  
## 模型构建：  
使用PyTorch定义两个参数矩阵：X_parameters 和 Theta_parameters 分别代表电影特征和用户偏好，初始化为随机值。  
定义损失函数，包括预测误差的平方损失和正则化项，以防止过拟合。  
使用Adam优化器更新参数矩阵，最小化损失函数。  
## 训练过程：  
进行多次迭代，每次迭代中更新参数矩阵，直到收敛或达到预定的迭代次数。  
输出训练损失和预测误差，以便监控模型训练情况。  
## 推荐生成：  
在训练结束后，根据最终的参数矩阵生成预测评分矩阵。  
用户输入其ID后，系统将输出预测评分最高的前20部电影及其名称。  



不过由于特征比较缺乏，无法做到有效的矩阵分解，再加上数据的庞大，目前进度困难有待后续改进
![8EDBE85437A5AE4C1E3FD59543534759](https://github.com/user-attachments/assets/60ca9a74-440c-4da8-ad66-7b137920ac59)




