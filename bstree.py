# DEBUG = True
import math, os

class Node:
	def __init__(self, value = None, left = None, right = None):
		self.left = left
		self.right = right
		self.parent = None
		self.ind = value
		self.x = None
		self.y = None
		self.width = None
		self.height = None

	def set_node_value(self, value, x = None, y = None, width = None, height = None):
		self.ind = value
		self.x = x
		self.y = y
		self.width = width
		self.height = height

class Bstree:
	def __init__(self, root = None):
		self.root = root
		self.xpoint = set()
		self.ypoint = set()
		self.hct = []
		self.vct = []
		self.ind_arr = []
		self.x_arr = []
		self.y_arr = []
		self.width_arr = []
		self.height_arr = []
		self.path = ''

	def set_path(self, path):
		self.path = path
		os.system('mkdir -p ' + path)

	def find_node(self, node, ind):
		if node == None:
			return None
		if node.ind == ind:
			return node
		res = self.find_node(node.left, ind)
		if res:
			return res
		return self.find_node(node.right, ind)

	def addnode(self, node, ind, x, y, width, height):
		# adding a new node (provided ind, x, y, width, height) to the left or right child of current node
		# the placement should be admissible, otherwise there will be node missing
		if node == None:
			return False
		if self.root.ind == None:
			self.root.set_node_value(ind, x, y, width, height)
			try:
				DEBUG
				print ('Node ', ind, 'add to the root')
			except NameError:
				pass
			return True
		elif x == node.x + node.width and y <= node.y + node.height and y + height >= node.y:
			if node.left == None:
				node.left = Node()
				node.left.set_node_value(ind, x, y, width, height)
				node.left.parent = node
				try:
					DEBUG
					print ('Node ', ind, 'add to the left of ', node.ind)
				except NameError:
					pass
				return True
		elif x == node.x:
			if node.right == None:
				node.right = Node()
				node.right.set_node_value(ind, x, y, width, height)
				node.right.parent = node
				try:
					DEBUG
					print ('Node ', ind, 'add to the right of ', node.ind)
				except NameError:
					pass
				return True
		try_right = self.addnode(node.right, ind, x, y, width, height)
		if not try_right:
			return self.addnode(node.left, ind, x, y, width, height)
		else:
			return True

	def flp2bstree(self, oind=None, ox=None, oy=None, owidth=None, oheight=None):
		# x, y, width, height describes an admissible floorplan generated by a tight placement (or a naive side-by-side placement)
		# x, y are bottom-left coordinates of chiplet
		try:
			DEBUG
			print ('flp2bstree')
		except NameError:
			pass		
		if oind == None:
			ind = self.ind_arr[:]
			x, y, width, height = self.x_arr[:], self.y_arr[:], self.width_arr[:], self.height_arr[:]
		else:
			ind = oind[:]
			x, y, width, height = ox[:], oy[:], owidth[:], oheight[:]
		chiplet_count = len(x)
		# ind = [i for i in range(chiplet_count)]
		x, y, width, height, ind = list(map(list, zip(*sorted(zip(x,y,width,height, ind), key=lambda pair: pair[0:2]))))
		self.root = Node()
		for i in range(chiplet_count):
			if not self.addnode(self.root, ind[i], x[i], y[i], width[i], height[i]):
				print ('something missing')
		# self.printTree(self.root)
		try:
			DEBUG
			global count
			print (count)
			print (ind, x, y, width, height,'', sep='\n')
			self.gen_flp(str(count))
			count += 1
		except NameError:
			pass
		return self.root

	def set_flp(self, oind, ox, oy, owidth, oheight):
		ind, x, y, width, height = oind[:], ox[:], oy[:], owidth[:], oheight[:]
		x, y, width, height, ind = list(map(list, zip(*sorted(zip(x,y,width,height, ind), key=lambda pair: pair[4]))))
		self.ind_arr = ind[:]
		self.x_arr = x[:]
		self.y_arr = y[:]
		self.width_arr = width[:]
		self.height_arr = height[:]					

	def parsenode(self, node, ind, x, y, width, height):
		if node == None:
			return
		ind.append(node.ind)
		x.append(node.x)
		y.append(node.y)
		width.append(node.width)
		height.append(node.height)
		self.parsenode(node.left, ind, x, y, width, height)
		self.parsenode(node.right, ind, x, y, width, height)

	def bstree2flp(self):
		try:
			DEBUG
			print ('bstree2flp')
		except NameError:
			pass
		ind, x, y, width, height = [], [], [], [], []
		self.parsenode(self.root, ind, x, y, width, height)
		x, y, width, height, ind = list(map(list, zip(*sorted(zip(x,y,width,height, ind), key=lambda pair: pair[4]))))
		self.ind_arr = ind[:]
		self.x_arr = x[:]
		self.y_arr = y[:]
		self.width_arr = width[:]
		self.height_arr = height[:]
		try:
			DEBUG
			global count
			self.gen_flp(str(count))
			print (count)
			print (ind, x, y, width, height,'', sep='\n')
			count += 1
		except NameError:
			pass

	def resetloc(self, node):
		if node == None:
			return
		node.x, node.y = None, None
		self.resetloc(node.left)
		self.resetloc(node.right)

	def computex(self, node):
		try:
			DEBUG
			if node == self.root:
				print ('compute x')
		except NameError:
			pass
		if node == None:
			return
		self.xpoint.add(node.x)
		self.xpoint.add(node.x + node.width)
		if node.left:
			node.left.x = node.x + node.width
			self.computex(node.left)
		if node.right:
			node.right.x = node.x
			self.computex(node.right)

	def relax_x(self, node, granularity):
		# left-bottom coordinate
		if node == None:
			return
		# left edge
		self.xpoint.add(node.x)
		# center point
		cx = (node.x + node.width + 1) / 2
		cx_grid = math.ceil(cx / granularity) * granularity
		# right edge
		rt = node.x + node.width + 1 + cx_grid - cx
		rt_grid = math.ceil(rt / granularity) * granularity
		self.xpoint.add(rt_grid)
		if node.left:
			node.left.x  = rt_grid
			self.relax_x(node.left, granularity)
		if node.right:
			node.right.x = node.x
			self.relax_x(node.right, granularity)

	def computey(self, node):
		try:
			DEBUG
			if node == self.root:
				print ('compute y')
		except NameError:
			pass
		if node == None:
			return
		y = 0
		for i in range(len(self.xpoint)):
			if node.x <= self.xpoint[i] < node.x + node.width:
				y = max(y, self.hct[i])
		node.y = y
		for i in range(len(self.xpoint)):
			if node.x <= self.xpoint[i] < node.x + node.width:
				self.hct[i] = y + node.height
		self.ypoint.add(node.y)
		self.ypoint.add(node.y + node.height)
		self.computey(node.left)
		self.computey(node.right)

	def relax_y(self, node, granularity):
		if node == None:
			return
		y = 0
		# x center point
		cx = (node.x + node.width + 1) / 2
		cx_grid = math.ceil(cx / granularity) * granularity
		# x right edge
		rt = node.x + node.width + 1 + cx_grid - cx
		rt_grid = math.ceil(rt / granularity) * granularity		
		for i in range(len(self.xpoint)):
			if node.x <= self.xpoint[i] < rt_grid:
				y = max(y, self.hct[i])
		node.y = y
		# y center point
		cy = (node.y + node.height + 1) / 2
		cy_grid = math.ceil(cy / granularity) * granularity
		# y top edge
		top = node.y + node.height + 1 + cy_grid - cy
		top_grid = math.ceil(top / granularity) * granularity
		for i in range(len(self.xpoint)):
			if node.x <= self.xpoint[i] < rt_grid:
					self.hct[i] = top_grid
		self.ypoint.add(node.y)
		self.ypoint.add(top_grid)
		self.relax_y(node.left, granularity)
		self.relax_y(node.right, granularity)

	def compacty(self):
		try:
			DEBUG
			print ('compact y')
		except NameError:
			pass	
		ind, x, y, width, height = self.ind_arr[:], self.x_arr[:], self.y_arr[:], self.width_arr[:], self.height_arr[:]
		y, x, width, height, ind = list(map(list, zip(*sorted(zip(y, x, width,height, ind), key=lambda pair: pair[0:1]))))
		# print (ind, x, y, width, height,'', sep = '\n')
		for i in range(len(ind)):
			yy = 0
			for j in range(len(self.xpoint)):
				if x[i] <= self.xpoint[j] < x[i] + width[i]:
					yy = max(yy, self.hct[j])
			y[i] = yy
			for j in range(len(self.xpoint)):
				if x[i] <= self.xpoint[j] < x[i] + width[i]:
					self.hct[j] = yy + height[i]
			self.ypoint.add(y[i])
			self.ypoint.add(y[i] + height[i])
		x, y, width, height, ind = list(map(list, zip(*sorted(zip(x,y,width,height, ind), key=lambda pair: pair[4]))))
		self.y_arr = y[:]
		try:
			DEBUG
			global count
			print (count)
			print (ind, x, y, width, height,'', sep = '\n')
			self.gen_flp(str(count))
			count += 1
		except NameError:
			pass

	def compactx(self):
		try:
			DEBUG
			print ('compact x')
		except NameError:
			pass
		ind, x, y, width, height = self.ind_arr[:], self.x_arr[:], self.y_arr[:], self.width_arr[:], self.height_arr[:]
		x, y, width, height, ind = list(map(list, zip(*sorted(zip(x,y,width,height, ind), key=lambda pair: pair[0:2]))))
		# print (ind, x, y, width, height,'', sep = '\n')
		for i in range(len(ind)):
			xx = 0
			for j in range(len(self.ypoint)):
				if y[i] <= self.ypoint[j] < y[i] + height[i]:
					xx = max(xx, self.vct[j])
			x[i] = xx
			for j in range(len(self.ypoint)):
				if y[i] <= self.ypoint[j] < y[i] + height[i]:
					self.vct[j] = xx + width[i]
			self.xpoint.add(x[i])
			self.xpoint.add(x[i] + width[i])
		x, y, width, height, ind = list(map(list, zip(*sorted(zip(x,y,width,height, ind), key=lambda pair: pair[4]))))
		self.x_arr = x[:]
		try:
			DEBUG
			global count
			print (count)
			print (ind, x, y, width, height,'', sep = '\n')
			self.gen_flp(str(count))
			count += 1
		except NameError:
			pass

	def reconstruct(self):
		# recompute the x, y location. since the tree may have rotate/swap/move node
		try:
			DEBUG
			print ('reconstruct')
		except NameError:
			pass
		self.resetloc(self.root)
		self.root.x = 0
		self.root.y = 0
		self.xpoint = set([0])
		self.computex(self.root)
		self.xpoint = sorted(list(self.xpoint))
		self.bstree2flp()
		# self.printTree(self.root)
		self.hct = [0] * len(self.xpoint) # hct for horizontal contour line
		self.ypoint = set([0])
		self.computey(self.root)
		self.ypoint = sorted(list(self.ypoint))
		self.vct = [0] * len(self.ypoint) # vct for vertical contour line
		self.bstree2flp() 
		# self.printTree(self.root)

		# make the flp compact
		while True:
			oind, ox, oy, owidth, oheight = self.ind_arr[:], self.x_arr[:], self.y_arr[:], self.width_arr[:], self.height_arr[:]
			self.xpoint = set([0])
			self.compactx()
			self.xpoint = sorted(list(self.xpoint))
			self.hct = [0] * len(self.xpoint) # hct for horizontal contour line
			# self.bstree2flp()
			# self.flp2bstree()
			
			self.ypoint = set([0])
			self.compacty()
			self.ypoint = sorted(list(self.ypoint))
			self.vct = [0] * len(self.ypoint) # vct for vertical contour line
			# self.bstree2flp()
			# self.flp2bstree()
			if (oind, ox, oy, owidth, oheight) == (self.ind_arr, self.x_arr, self.y_arr, self.width_arr, self.height_arr):
				break
		# reconstruct the flp using the admissible flp
		self.flp2bstree()

	def rotate(self, node):
		# print ('rotate', node.ind)
		# rotate do not change B*-tree structure, but will impact the flp
		node.width, node.height = node.height, node.width
		self.bstree2flp()
		self.reconstruct()

	def swap(self, node1, node2):
		# print ('swap', node1.ind, node2.ind)
		# instead of applying insert and delete operations, we use an alternative by swapping the index, width and height, but maintain the tree relationship and update the xy coordinates.
		node1.width, node2.width = node2.width, node1.width
		node1.height, node2.height = node2.height, node1.height
		node1.ind, node2.ind = node2.ind, node1.ind
		# self.printTree(self.root)
		self.bstree2flp()
		self.reconstruct()
		# self.printTree(self.root)

	def delete(self, node):
		# print ('delete', node.ind)
		if node.left and node.right:
			# the node has two children
			while (node.left and node.right):
				# self.swap(node, node.left)
				node.width, node.left.width = node.left.width, node.width
				node.height, node.left.height = node.left.height, node.height
				node.ind, node.left.ind = node.left.ind, node.ind
				node = node.left
			if node.left:
				node.parent.left = node.left
				node.left.parent = node.parent
			elif node.right:
				node.parent.left = node.right
				node.right.parent = node.parent
			else:
				node.parent.left = None
				node.parent = None
		elif node.left:
			# the node has only left child
			if node.parent == None:
				# delete the root node
				self.root = node.left
			elif node.parent.left == node:
				node.parent.left = node.left
			elif node.parent.right == node:
				node.parent.right = node.left
			node.left.parent = node.parent
		elif node.right:
			# the node has only right child
			if node.parent == None:
				# delete the root node
				self.root = node.right
			elif node.parent.left == node:
				node.parent.left = node.right
			elif node.parent.right == node:
				node.parent.right = node.right
			node.right.parent = node.parent
		else:
			# the node is a leaf, just delete it
			if node.parent.left == node:
				node.parent.left = None
			elif node.parent.right == node:
				node.parent.right = None
			node.parent = None
		node.left = None
		node.right = None
		return node

	def insert(self, node, parent, direction):
		# add the node to the leaf of the parent, direction indicates left or right child
		# print ('insert', node.ind, parent if parent == None else parent.ind, direction)
		if parent == None:
			# the only case the parent is none is to insert the node to the root
			if direction == 'left':
				node.left = self.root
				self.root.parent = node
				node.right = None
				node.parent = None
				self.root = node
			elif direction == 'right':
				node.right = self.root
				self.root.parent = node
				node.left = None
				node.parent = None
				self.root = node
		elif direction == 'left':
			if parent.left:
				node.left = parent.left
				parent.left.parent = node
				node.right = None
			parent.left = node
			node.parent = parent
		elif direction == 'right':
			if parent.right:
				node.right = parent.right
				parent.right.parent = node
				node.left = None
			parent.right = node
			node.parent = parent

	def move(self, node1, node2, direction):
		n2 = node2 if node2 == None else node2.ind
		# print ('move', node1.ind, node2 if node2 == None else node2.ind, direction)
		node = self.delete(node1)
		try:
			DEBUG
			global count
			print (count)
			self.printTree(self.root)
			self.bstree2flp()
			self.gen_flp(str(count))
		except NameError:
			pass		
		if n2 == None:
			self.insert(node, None, direction)
		else:
			self.insert(node, self.find_node(self.root, n2), direction)
		# self.insert(node, node2, direction)
		try:
			DEBUG
			print (count)
			self.printTree(self.root)
			self.bstree2flp()
			self.gen_flp(str(count))
			count += 1
		except NameError:
			pass		
		self.bstree2flp()
		self.reconstruct()

	def printTree(self, tree):
		# print the tree in preorder
		if tree == self.root:
			print ('ind', 'x', 'y', 'width', 'height', 'parent','left','right', sep = '\t')
		if tree != None:
			print (tree.ind, tree.x, tree.y, tree.width, tree.height, tree.parent.ind if tree.parent else None, tree.left.ind if tree.left else None, tree.right.ind if tree.right else None, sep='\t')
			self.printTree(tree.left)
			self.printTree(tree.right)

	def gen_flp(self, filename):
		import os
		# path = 'outputs/bstree/'
		with open(self.path+filename + 'sim.flp','w') as SIMP:
			for i in range(len(self.ind_arr)):
				SIMP.write("node_"+str(self.ind_arr[i])+"\t"+str(self.width_arr[i])+"\t"+str(self.height_arr[i])+"\t"+str(self.x_arr[i])+"\t"+str(self.y_arr[i])+"\n")
			SIMP.write(' \t50\t50\t0\t0\n')
		# os.system("perl util/tofig.pl -f 20 "+self.path+filename+"sim.flp | fig2dev -L ps | ps2pdf - "+self.path+filename+"sim.pdf")


if __name__ == "__main__":
	DEBUG = True
	global count
	count = 1
	# example 1
	# x = [0, 2, 2, 0]
	# y = [0, 0, 1, 3]
	# width = [2, 1, 2, 3]
	# height = [2, 1, 2, 1]

	# example 2
	# node   0  1    2  3    4    5  6  7
	# ind = 	[0, 1,   2, 3, 	 4,   5, 6, 7]
	# x = 	[0, 3, 	 0, 3, 	 5,   2, 0, 3]
	# y = 	[0, 0, 	 2, 1.5, 1.5, 3, 5, 4]
	# width = [3, 4, 	 2, 2, 	 1,   4, 3, 4]
	# height =[2, 1.5, 3, 1.5, 1,   1, 2, 2]

	# example 3
	# x = [0, 0, 1, 1]
	# y = [0, 1, 0, 1]
	# width = [1,1,1,1]
	# height = [1,1,1,1]

	# example 4
	ind = 	[0, 1,   2, 3, 	 4,   5,   6, 7]
	x = 	[0, 3,   0, 2,   3,   4,   2, 0]
	y = 	[0, 0,   2, 2.5, 1.5, 1.5, 4, 6]
	width = [3, 4,   2, 2,   1,   4,   3, 4]
	height =[2, 1.5, 3, 1.5, 1,   1,   2, 2]	

	# example 5
	ind = 	[0, 1,   2,   3,   4, 5, 6,   7]
	x = 	[6, 2,   0,   0,   2, 2, 0,   3]
	y = 	[0, 2,   1.5, 0,   0, 1, 4.5, 3.5]
	width = [3, 4,   2,   2,   1, 4, 3,   4]
	height =[2, 1.5, 3,   1.5, 1, 1, 2,   2]

	# example 6
	ind = 	[0,   1,   2,   3,   4, 5, 6, 7]
	x = 	[0,   2,   0,   0,   6, 6, 7, 2]
	y = 	[4.5, 0,   1.5, 0,   0, 2, 0, 1.5]
	width = [3,   4,   2,   2,   1, 4, 3, 4]
	height =[2,   1.5, 3,   1.5, 1, 1, 2, 2]

	# example 7
	ind = 	[0,   1,   2,   3,   4, 5, 6, 7]
	x = 	[0,   0,   0,   0,   0, 0, 0, 0]
	y = 	[0,   0,   0,   0,   0, 0, 0, 0]
	width = [3,   4,   2,   2,   1, 4, 3, 4]
	height =[2,   1.5, 3,   1.5, 1, 1, 2, 2]

	tree = Bstree()
	tree.set_path('outputs/bstree/')
	tree.set_flp(ind, x, y, width, height)
	tree.flp2bstree(ind, x, y, width, height)
	tree.reconstruct()
	# print ('rotate 1')
	# tree.rotate(tree.find_node(tree.root, 1))
	# print ('swap 1 and 2')
	# tree.swap(tree.find_node(tree.root, 1), tree.find_node(tree.root, 2))
	# print ('move 2 to the root')
	# tree.move(tree.find_node(tree.root, 2), tree.root.parent, 'left')
	print ('move 2 to the right child of node 7')
	tree.move(tree.find_node(tree.root, 2), tree.find_node(tree.root, 7), 'right')
	# del_node = tree.delete(tree.find_node(tree.root, 1))
	# tree.reconstruct()
	# print ('after delete node 1')
	# tree.printTree(tree.root)
	# tree.insert(del_node, tree.root.parent, 'left')
	print ('\n after insert node 1 to the root')
	tree.printTree(tree.root)
	print (tree.root.ind)

	tree.bstree2flp()

