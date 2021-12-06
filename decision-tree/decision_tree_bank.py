# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# CART on the Bank Note data_set
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	data_set = list(lines)
	return data_set

# Convert string column to float
def str_column_to_float(data_set, column):
	for row in data_set:
		row[column] = float(row[column].strip())

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(tree, test_set):
	correct = 0
	for row in test_set:
		if predict(tree, row) == row[-1]:
			correct += 1
	return correct / float(len(test_set)) * 100.0

# Split a data_set based on an attribute and an attribute value
def test_split(index, value, data_set):
	left, right = list(), list()
	for row in data_set:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split data_set
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a data_set
# 找到最佳分割点，即按某点分割后，基尼系数最小的点
def get_split(data_set):
	class_values = list(set(row[-1] for row in data_set))
	# 这4个参数记录了最佳分割点信息，b_index：数据集中的列，b_value：该列对应的值，b_score：基尼系数，b_groups：把数据集按基尼系数最小得到的最佳分割点，分割后的左右两组数据
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	# 按每一列遍历
	for index in range(len(data_set[0])-1):
		# 对于每一列都需要遍历数据集的所有行
		for row in data_set:
			# 取出遍历到的该行该列的值作为参考点，再遍历该列的所有数据集，比参考点小的放左边，否则放右边，
			groups = test_split(index, row[index], data_set)
			# 计算分割后的基尼系数
			gini = gini_index(groups, class_values)
			if gini < b_score:
				# 如果比之前的小，就更新
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	# 由以上实现逻辑可知，遍历完后，最终将得到基尼系数最小的分割点
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
# 创建终止节点
def to_terminal(group):
	# 取出指定组中的最后一列的值组成新的list
	outcomes = [row[-1] for row in group]
	# 在该组中每个样本都有对应的类别，数量最多的类别就作为决策树判定该组的类别
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		# 如果下面不需要再进一步分组了（在本次分组中至少有1组为None），就可以生成终止节点判定类别了
		# 易知，当前分的组全部都属于同一类
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		# 如果决策树的深度大于指定深度，终止分组迭代，提前判定类别是过拟合的解决方法之一
		# 左边的组判定左边的类别，右边的组判定右边的类别
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		# 如果左边的组样本数量太少，就没有必要再分组了，这也是过拟合的解决方法之一
		node['left'] = to_terminal(left)
	else:
		# 如果左边这组剩余数量还比较多，继续迭代分组
		# 可见，决策树的结构和二叉树一样，有左右孩子节点
		# 在决策树中，影响力越大的特征离根节点越近
		# 每一个节点的数据结构为：
		# {'index': 根据训练集基于该列分割得到的基尼系数最小,
		#  'value': 在训练集中的分割样本的分割列对应的值,
		#  'groups': 分割的左右两组,
		#  'left': 继续分割迭代的左子树,
		#  'right': 继续分割迭代的右子树}
		# 根据不同的列（即特征）其影响力的大小（基尼系数）生成的决策树，就可以做预测功能了
		# 从根节点开始比对（根节点对分类结果影响最大，层数越高对分类的影响力越小），然后往子树走，一直走到终止节点
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
# 构建决策树
def build_tree(train, max_depth, min_size):
	# 先根据最小基尼系数查找到首个最佳分割点，作为决策树的根节点
	root = get_split(train)
	# 然后迭代分割，最终将得到决策树模型
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	# 对于决策树的当前节点，index记录了决策树判定的最佳分割特征，value记录了决策树在训练集中分割样本对应分割列的值
	# 对于该特征的值，测试样本如果比训练集分割样本的值小，就往左子树走，否则往右子树走
	if row[node['index']] < node['value']:
		# 左子树如果是dict，说明还要继续迭代以确定最终的类别
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			# 如果不是dict，那就是终止节点，是该样本最终的类别
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
# 对传入的train训练集生成决策树模型，并根据传入的test测试集返回预测结果
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return (predictions)

# Test CART on Bank Note data_set
seed(1)
# load and prepare data
train_filename = '../data/bank_train.csv'
train_set = load_csv(train_filename)
test_filename = '../data/bank_test.csv'
test_set = load_csv(test_filename)
# convert string attributes to integers
for i in range(len(train_set[0])):
	str_column_to_float(train_set, i)
for i in range(len(test_set[0])):
	str_column_to_float(test_set, i)

# evaluate algorithm
max_depth = 6
min_size = 3

tree = build_tree(train_set, max_depth, min_size)

# 评估正确率
scores = evaluate_algorithm(tree, test_set)
print('Accuracy: %s' % scores)
