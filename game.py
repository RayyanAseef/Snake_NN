from Game_Parts.snake import Snake
import pygame, numpy, random, math
import copy

class SnakeGame:
    def __init__(self, gridSize = 10, gridLineWidth = 4, snake_size=3):
        self.gridSize = gridSize
        self.gridLineWidth = gridLineWidth

        self.snake_start_size = snake_size
        self.reset()

    def reset(self):
        self.score = 0

        self.grid = numpy.zeros((self.gridSize, self.gridSize), dtype=int)
        self.snake = Snake(self.gridSize//2, self.gridSize//2, min(self.snake_start_size, self.gridSize-1), 1, 0)
        self._placeApple()

    def get_state(self):
        left_rotation = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}
        right_rotation = {(1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1), (0, 1): (1, 0)}
        directions = [[self.snake.xDir, self.snake.yDir], [*right_rotation[(self.snake.xDir, self.snake.yDir)]], [*left_rotation[(self.snake.xDir, self.snake.yDir)]]]

        dangers = [self.snake.isDead(self.gridSize, body=self.snake.body, pt=[self.snake.body[0].x + direction[0], self.snake.body[0].y - direction[1]]) for direction in directions]

        directions = { (1, 0): (0, 1, 0, 0), (-1, 0): (1, 0, 0, 0), (0,  1): (0, 0, 1, 0), (0, -1): (0, 0, 0, 1)  }
        currentDirection = self.snake.xDir, self.snake.yDir

        dirLeft, dirRight, dirUp, dirDown = directions[currentDirection]

        head = self.snake.body[0]
        apple = self.applePos

        foodRight = 1 if head.x < apple[0] else 0
        foodLeft = 0 if head.x < apple[0] else 1

        foodUp = 1 if head.y > apple[1] else 0
        foodDown = 0 if head.y > apple[1] else 1
        
        return [ dangers[0], dangers[1], dangers[2],
                    dirLeft, dirRight, dirUp, dirDown,
                    foodLeft, foodRight, foodUp, foodDown ]

    def input(self, turn):
        self.snake.changeDirection(turn)
        deadState, ateApple = self.snake.move(self.gridSize, self.applePos)

        self.score = self.snake.bodySize - self.snake_start_size

        if not numpy.any(self.grid == -1):
            self.grid = self._placeApple()
        self.grid = self._placeSnake()

        reward = 10 if ateApple else -10 if deadState else 0

        return deadState, reward
    
    def simulate_move(self, turns):
        snake = copy.deepcopy(self.snake)

        for turn in turns:
            snake.changeDirection(turn)
            deadState, _ = snake.move(self.gridSize, self.applePos)

        score = snake.bodySize - self.snake_start_size

        appleDistance = math.sqrt((self.applePos[0] - snake.body[0].x)**2 + (self.applePos[1] - snake.body[0].y)**2)
        return appleDistance, score, deadState

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
