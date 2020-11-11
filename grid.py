import pygame

pygame.init()

HIGHT,WIDTH=600,600
BLACK,WHITE,GRAY=(0,0,0),(255,255,255),(125,125,125)

screen = pygame.display.set_mode((HIGHT,WIDTH))


RUN=True

class Block(object):
	"""docstring for block"""
	def __init__(self, x,y , size):
		super(Block, self).__init__()
		self.x = int(x)
		self.y = int(y)
		self.size=int(size)
	def draw(self,color=WHITE):
		pygame.draw.rect(screen,color,(self.x,self.y,self.size, self.size))
	



class Grid(object):
	"""docstring for Grid"""
	def __init__(self,row,cols):
		super(Grid, self).__init__()
		self.row = row
		self.cols = cols
		self.grid=self.makeGrid()

	def makeGrid(self):
		grid = [[] for i in range(self.row)]
		for i in range(self.row):
			for j in range(self.cols):
				g=Block( int(i*WIDTH/self.row) , int(j*HIGHT/self.cols) , int(WIDTH/self.row) )
				grid[i].append(g)
		return grid


	def draw(self,):
		for i in range(self.row):
			for j in range(self.cols):
				# if i==0 or j==0 or i==self.row-1 or j==self.cols-1:
					self.grid[i][j].draw()

		for i in range(1,self.row):
			pygame.draw.line(screen,GRAY,(0,int(HIGHT*i/self.row)),(WIDTH,int(HIGHT*i/self.row)),1)
		for i in range(1,self.cols):
			pygame.draw.line(screen,GRAY,(int(WIDTH*i/self.cols),0),(int(WIDTH*i/self.cols),HIGHT),1)
		
g = Grid(50,50)
while RUN:
	screen.fill(BLACK)
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			exit()
	g.draw()

	pygame.display.update()
