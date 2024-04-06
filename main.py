from game import SnakeGameAI
import pygame
from snake import Snake

screenSize = 500
screen = pygame.display.set_mode([screenSize, screenSize])

clock = pygame.time.Clock()
game = SnakeGameAI(10, 4)

def test_direction(game, direction_input):
    head = game.snake.body[0]
    temp_snake = Snake(startX=head.x, startY=head.y, bodySize=1, xDir=game.snake.xDir, yDir=game.snake.yDir)
    temp_snake.changeDirection(direction_input)
    _, game_over = temp_snake.move(game.gridSize, game.applePos)
    return 1 if game_over else 0

def get_state(game):
    # Inputs for straight, right, and left movements
    directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Test each direction
    dangers = [test_direction(game, direction) for direction in directions]

    directions = { (1, 0): (0, 1, 0, 0), (-1, 0): (1, 0, 0, 0), (0,  1): (0, 0, 1, 0), (0, -1): (0, 0, 0, 1)  }
    currentDirection = game.snake.xDir, game.snake.yDir

    dirLeft, dirRight, dirUp, dirDown = directions[currentDirection]

    head = game.snake.body[0]
    apple = game.applePos

    foodRight = 1 if head.x < apple[0] else 0
    foodLeft = 0 if head.x < apple[0] else 1

    foodUp = 1 if head.y > apple[1] else 0
    foodDown = 0 if head.y > apple[1] else 1

    print(
        dangers[0], dangers[1], dangers[2],
        dirLeft, dirRight, dirUp, dirDown,
        foodLeft, foodRight, foodUp, foodDown
    )

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
    
    reward, game_over, score = game.input(turn)
    get_state(game)

    game.draw(screen, screenSize)
    pygame.display.update()

    running = not game_over

pygame.quit()