from Model import Model
from Game import SnakeGame
import pygame, numpy

class Agent:
    def __init__(self, gridsize=10, snakesize=4):
        self.model = Model()
        self.game = SnakeGame(gridsize, snakesize)
        self.X = []
        self.y = []

    def create_model(self, layers, loss, optimizer, accuracy):
        self.model.layers = layers
        self.model.set(loss=loss, optimizer=optimizer, accuracy=accuracy)
        self.model.finalize()

    def get_state(self, game):
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
        
        return [ dangers[0], dangers[1], dangers[2],
                    dirLeft, dirRight, dirUp, dirDown,
                    foodLeft, foodRight, foodUp, foodDown ]
    
    def remember(self, state, y_true):
        self.X.append(state)
        self.y.append(y_true)

    def train_long_memory(self, batch_size, displayResult=False):
        X = numpy.array(self.X)
        y = numpy.array(self.y)

        keys = numpy.array(range(X.shape[0]))
        numpy.random.shuffle(keys)

        X = X[keys]
        y = y[keys]

        if len(X) > batch_size:
            X = X[:batch_size]
            y = y[:batch_size]
        
        printEvery = 100 if displayResult else None
        self.model.train(X, y, epochs=100, print_every_epoch=printEvery)

    def train_short_memory(self, state, y_true):
        move = self.model.predict(state)

        turn = [0, 0, 0]
        turn[numpy.argmax(move)] = 1

        deadState = self.game.input(turn)
        self.model.train(numpy.array([state]), numpy.array([y_true]), epochs=1)

        return deadState

    def calculate_best_move(self):
        best_loss = float('inf')
        best_move = [0, 0, 0]
        for i in range(3):
            turn = [0, 0, 0]
            turn[i] = 1

            appleDistance, currentScore, deadState = self.game.simulate_move([turn])
            current_loss = ((1-0.5**appleDistance) + 20*deadState) * (1-0.5**(100-currentScore))

            if current_loss < best_loss:
                best_loss = current_loss
                best_move = turn

        return best_move

    def train(self, n):
        best_score = -1
        update_percent = 0
        for i in range(n):
            state = self.get_state(self.game)
            y_true = self.calculate_best_move()

            deadState = self.train_short_memory(state, y_true)
            self.remember(state, y_true)

            if deadState:
                if self.game.score > best_score:
                    best_score = self.game.score
                    print(f"Best Score: {best_score}")
                
                self.game.reset()
                self.train_long_memory(1000)

            if i / n > update_percent:
                print(f"----------------- Progress: {update_percent*100}% --------------------")
                update_percent += 0.2
                self.train_long_memory(min(10000, len(self.X)), displayResult=True)
        
        print(f"----------------- Progress: {update_percent*100}% --------------------")