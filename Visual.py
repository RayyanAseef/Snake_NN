from game import SnakeGameAI
import pygame
from snake import Snake
import copy

screenSize = 500
screen = pygame.display.set_mode([screenSize, screenSize])

clock = pygame.time.Clock()
game = SnakeGameAI(10, 4)

def get_state(game):
    left_rotation = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}
    right_rotation = {(1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1), (0, 1): (1, 0)}
    directions = [[game.snake.xDir, game.snake.yDir], [*right_rotation[(game.snake.xDir, game.snake.yDir)]], [*left_rotation[(game.snake.xDir, game.snake.yDir)]]]

    dangers = [game.snake.isDead(game.gridSize, body=game.snake.body, pt=[game.snake.body[0].x + direction[0], game.snake.body[0].y - direction[1]]) for direction in directions]

    directions = { (1, 0): (0, 1, 0, 0), (-1, 0): (1, 0, 0, 0), (0,  1): (0, 0, 1, 0), (0, -1): (0, 0, 0, 1)  }
    currentDirection = game.snake.xDir, game.snake.yDir

    dirLeft, dirRight, dirUp, dirDown = directions[currentDirection]

    head = game.snake.body[0]
    apple = game.applePos

    foodRight = 1 if head.x < apple[0] else 0
    foodLeft = 0 if head.x < apple[0] else 1

    foodUp = 1 if head.y > apple[1] else 0
    foodDown = 0 if head.y > apple[1] else 1

    return [ dangers[0], dangers[1], dangers[2], dirLeft, dirRight, dirUp, dirDown, foodLeft, foodRight, foodUp, foodDown ]

deadState = 0
running = True
while running:
    clock.tick(8)

    turn = [1, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                turn = [0, 0, 1]
            elif event.key == pygame.K_RIGHT:
                turn = [0, 1, 0]
            elif event.key == pygame.K_BACKSPACE:
                running = False
        
    if not deadState:
        # turn = model.predict(get_state(game))
        score, deadState = game.input(turn)

        game.draw(screen, screenSize)
        pygame.display.update()

pygame.quit()

# Loss A =  (1−0.5^appleDistance) − (1−0.5^(maxScore−currentScore)) + deathPenalty×deadState)
# Loss B = ( (1−0.5^appleDistance) + deathPenalty×deadState) ) * (1−0.5^(maxScore−currentScore))