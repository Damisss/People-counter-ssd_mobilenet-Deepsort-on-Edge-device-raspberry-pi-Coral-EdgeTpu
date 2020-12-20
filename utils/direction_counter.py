import numpy as np

class DirectionCounter:
	def __init__(self, direction, H, W):
		# initialize the height and width of the input image
		self.H = H
		self.W = W

		# initialize variables holding the direction of movement,
		# along with counters for each respective movement (i.e.,
		# left-to-right and top-to-bottom)
		self.direction = direction
		self.totalUp = 0
		self.totalDown = 0
		self.totalRight = 0
		self.totalLeft = 0

		# initialize the mean between the *current* centroid and the
		# mean of *previous* centroids to zero
		self.mean = 0

	def find_direction(self, to, centroid):
		# check to see if we are tracking horizontal movements
		if self.direction == "horizontal":
			# the difference between the x-coordinate of the
			# *current* centroid and the mean of *previous* centroids
			# will tell us in which direction the object is moving
			# (negative for 'left' and positive for 'right')
			x = [c[0] for c in to.centroids]
			self.mean = centroid[0] - np.mean(x)

		# otherwise we are tracking vertical movements
		elif self.direction == "vertical":
			# the difference between the y-coordinate of the
			# *current* centroid and the mean of *previous* centroids
			# will tell us in which direction the object is moving
			# (negative for 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			self.mean = centroid[1] - np.mean(y)

	def count_object(self, to, centroid):
		# initialize the output list
		output = []

		# check if the direction of the movement is horizontal
		if self.direction == "horizontal":
			# if the mean is negative (indicating the object
			# is moving left) AND the centroid is above the center
			# line, count the object
			if self.mean < 0 and centroid[0] < self.W // 2:
				self.totalLeft += 1
				to.counted = True

			# if the mean is positive (indicating the object
			# is moving right) AND the centroid is below the
			# center line, count the object
			elif self.mean > 0 and centroid[0] > self.W // 2:
				self.totalRight += 1
				to.counted = True

			# construct a list of tuples with the count of objects
			# that have passed in the left and right direction
			output = [("Left", self.totalLeft),
				("Right", self.totalRight)]

		# otherwise the direction of movement is vertical
		elif self.direction == "vertical":
			# if the mean is negative (indicating the object
			# is moving up) AND the centroid is above the center
			# line, count the object
			if self.mean < 0 and centroid[1] < self.H // 2:
				self.totalUp += 1
				to.counted = True

			# if the mean is positive (indicating the object
			# is moving down) AND the centroid is below the
			# center line, count the object
			elif self.mean > 0 and centroid[1] > self.H // 2:
				self.totalDown += 1
				to.counted = True

			# return a list of tuples with the count of objects that
			# have passed in the up and down direction
			output = [("Up", self.totalUp), ("Down", self.totalDown)]

		# return the output list
		return output