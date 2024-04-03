from game import SnakeGameAI
import pygame

screenSize = 500
screen = pygame.display.set_mode([screenSize, screenSize])

clock = pygame.time.Clock()
game = SnakeGameAI(10, 4)

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

    game.draw(screen, screenSize)
    pygame.display.update()

    running = not game_over

pygame.quit()