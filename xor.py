import pygame
import numpy as np
import neural_network as nn
import time
pygame.init()

HIGHT, WIDTH = 600, 600
BLACK, WHITE, GRAY = np.array([0, 0, 0]), np.array(
    [255, 255, 255]), np.array([100, 100, 100])

screen = pygame.display.set_mode((HIGHT, WIDTH))


RUN = True


class Block(object):
    """docstring for block"""

    def __init__(self, x, y, size):
        super(Block, self).__init__()
        self.x = int(x)
        self.y = int(y)
        self.size = int(size)

    def draw(self, color=WHITE):
        pygame.draw.rect(screen, color, (self.x, self.y, self.size, self.size))


class Grid(object):
    """docstring for Grid"""

    def __init__(self, row, cols):
        super(Grid, self).__init__()
        self.row = row
        self.cols = cols
        self.grid = self.makeGrid()
        self.brain = nn.NeuralNetwork([2, 5, 1], 0.1)
        self.data_x = [[0, 0], [1, 1], [1, 0], [0, 1]]
        self.data_y = [[0], [0], [1], [1]]

    def makeGrid(self):
        grid = [[] for i in range(self.row)]
        for i in range(self.row):
            for j in range(self.cols):
                g = Block(int(i*WIDTH/self.row),
                          int(j*HIGHT/self.cols), int(WIDTH/self.row))
                grid[i].append(g)
        return grid

    def draw(self,):
        self.brain.train(self.data_x, self.data_y, 100)
        for i in range(self.row):
            for j in range(self.cols):
                # if i==0 or j==0 or i==self.row-1 or j==self.cols-1:
                ans = self.brain.predict([(i+1)/self.row, (j+1)/self.cols])
                # print(ans,[(i+1)/self.row , (j+1)/self.cols])
                self.grid[i][j].draw(WHITE*ans)
        # exit()

        for i in range(1, self.row):
            pygame.draw.line(screen, GRAY, (0, int(
                HIGHT*i/self.row)), (WIDTH, int(HIGHT*i/self.row)), 1)
        for i in range(1, self.cols):
            pygame.draw.line(screen, GRAY, (int(WIDTH*i/self.cols),
                                            0), (int(WIDTH*i/self.cols), HIGHT), 1)


g = Grid(15, 15)
while RUN:
    screen.fill(BLACK)
    pygame.time.delay(10)
    # time.sleep(0.1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    g.draw()

    pygame.display.update()
