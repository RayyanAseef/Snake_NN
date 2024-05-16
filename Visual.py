from Game import SnakeGame
from Model import Model
import pygame, numpy

screenSize = 500
screen = pygame.display.set_mode([screenSize, screenSize])

clock = pygame.time.Clock()
game = SnakeGame(10, 4)

model = Model.load('Model.model')

deadState = 0
running = True
while running:
    clock.tick(8)

    turn = [1, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                running = False
        
    if not deadState:
        turn = model.predict(game.get_state())
        index = numpy.argmax(turn)
        turn = [0, 0, 0]
        turn[index] = 1
        deadState = game.input(turn)

        game.draw(screen, screenSize)
        pygame.display.update()

    else:
        game.reset()
        deadState = not deadState

pygame.quit()

# Loss A =  (1−0.5^appleDistance) − (1−0.5^(maxScore−currentScore)) + deathPenalty×deadState)
# Loss B = ( (1−0.5^appleDistance) + deathPenalty×deadState) ) * (1−0.5^(maxScore−currentScore))