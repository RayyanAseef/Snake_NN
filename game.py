from snake import Snake
import pygame, numpy, random

class SnakeGameAI:

    def __init__(self, gridSize = 10, gridLineWidth = 4, snake_size=3):
        self.gridSize = gridSize
        self.gridLineWidth = gridLineWidth

        self.snake_start_size = snake_size
        self.reset(self.snake_start_size)

    def reset(self, snake_size):
        self.score = 0

        self.grid = numpy.zeros((self.gridSize, self.gridSize), dtype=int)
        self.snake = Snake(self.gridSize//2, self.gridSize//2, min(snake_size, self.gridSize-1), 1, 0)
        self._placeApple()

    def input(self, turn):
        self.snake.changeDirection(turn)
        deadState = self.snake.move(self.gridSize, self.applePos)

        self.score = self.snake.bodySize - self.snake_start_size

        if not numpy.any(self.grid == -1):
            self.grid = self._placeApple()
        self.grid = self._placeSnake()

        return self.score, deadState

    def draw(self, surface, surfaceSize):
        surface.fill((0, 0, 0))

        gap = surfaceSize / self.gridSize
        halfLineWidth = self.gridLineWidth / 2

        for i in range(self.gridSize + 1):
            pygame.draw.line(surface, (255, 255, 255), [0 - halfLineWidth, gap * i], [surfaceSize - halfLineWidth, gap * i], self.gridLineWidth)
            pygame.draw.line(surface, (255, 255, 255), [gap * i, 0 - halfLineWidth], [gap * i, surfaceSize - halfLineWidth], self.gridLineWidth)

        snakeBodyColors = {i + 1: (0, 155 * (i / self.snake.bodySize) + 100, 0) for i in range(self.snake.bodySize)}

        for row in range(self.gridSize):
            for col in range(self.gridSize):
                if self.grid[row, col] > 0:
                    pygame.draw.rect(surface, snakeBodyColors[self.grid[row, col]], [col * gap + halfLineWidth + 1, row * gap + halfLineWidth + 1, gap - self.gridLineWidth, gap - self.gridLineWidth])
                elif self.grid[row, col] == -1:
                    pygame.draw.rect(surface, (255, 0, 0), [col * gap + halfLineWidth + 1, row * gap + halfLineWidth + 1, gap - self.gridLineWidth, gap - self.gridLineWidth])
    
    def _placeSnake(self):  
        self.grid[self.grid == 1] = 0

        for i in range(self.snake.bodySize):
            if self.snake.body[i].x <= self.gridSize-1 and self.snake.body[i].y <= self.gridSize-1 and self.snake.body[i].x >= 0 and self.snake.body[i].y >= 0:
                self.grid[self.snake.body[i].y][self.snake.body[i].x] = self.snake.bodySize-i

        return self.grid

    def _placeApple(self):
        availablePlaces = numpy.argwhere(self.grid == 0)
        if len(availablePlaces) > 0:
            self.applePos = availablePlaces[random.randint(0, len(availablePlaces) - 1)]
            self.grid[self.applePos[1], self.applePos[0]] = -1
        return self.grid
