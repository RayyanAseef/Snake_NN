from Model import Model
from Game import SnakeGame
import pygame, numpy

class Agent:
    def __init__(self, gridsize=10, snakesize=4):
        self.model = Model()
        self.game = SnakeGame(gridsize, snakesize)
        # self.X = numpy.empty((0, 11))
        # self.y = numpy.empty((0,3))
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
    
    def remember(self, state):
        self.X.append(state)

        y_true = self.calculate_best_move()
        self.y.append(y_true)

    def train_long_memory(self, batch_size):
        X = numpy.array(self.X)
        y = numpy.array(self.y)

        keys = numpy.array(range(X.shape[0]))
        numpy.random.shuffle(keys)

        X = X[keys]
        y = y[keys]

        if len(X) > batch_size:
            X = X[:batch_size]
            y = y[:batch_size]
        
        self.model.train(X, y, epochs=1000)

    def train_short_memory(self, state):
        y_true = self.calculate_best_move()
        move = self.model.predict(state)

        turn = [0, 0, 0]
        turn[move[0][0]] = 1

        deadState = self.game.input(turn)
        if deadState:
            self.game.reset()

        self.model.train(numpy.array([state]), numpy.array([y_true]), epochs=1)
        return deadState

    def calculate_best_move(self):
        best_loss = float('inf')
        best_move = [0, 0, 0]
        for i in range(3):
            turn = [0, 0, 0]
            turn[i] = 1

            appleDistance, currentScore, deadState = self.game.simulate_move([turn])
            eatenApple = appleDistance == 1
            current_loss = ((1-0.5**appleDistance) + 20*deadState + turn[0]*20*deadState - turn[0]*(appleDistance < 2)) * (1-0.5**(100-currentScore))

            if current_loss < best_loss:
                best_loss = current_loss
                best_move = turn

        return best_move

    def train(self, n):
        best_score = 0
        update_percent = 0
        for i in range(n):
            state = self.get_state(self.game)
            deadState = self.train_short_memory(state)
            self.remember(state)

            if deadState:
                self.game.reset()
                self.train_long_memory(1000)

                if self.game.score > best_score:
                    best_score = self.game.score
                    print(f"Best Score: {best_score}")

            if i / n > update_percent:
                print(f"Progress: {update_percent*100}%")
                update_percent += 0.2

    # def simulate_game(self, n, m, epoch=1, batch_size=None, Visual_Learning=False):
    #     best_score = 0
    #     for i in range(n):
    #         state = self.get_state(self.game)
    #         y_true = self.calculate_best_move()

    #         self.X = numpy.vstack([self.X, state])
    #         self.y = numpy.vstack([self.y, y_true])
            
    #         turn = self.model.predict(state)
    #         index = numpy.argmax(turn)
    #         turn = [0, 0, 0]
    #         turn[index] = 1

    #         deadState = self.game.input(turn)

    #         if self.game.score > best_score:
    #             best_score = self.game.score
    #             print(f"Best Score: {best_score}")

    #         if deadState:
    #             self.game.reset()

    #         self.model.train(numpy.array([state]), numpy.array([y_true]), epochs=1)

    #         if i % m == 0 and i > m*5:
    #             print(f"Training: {i}")

    #             keys = numpy.array(range(self.X.shape[0]))
    #             numpy.random.shuffle(keys)

    #             self.X = self.X[keys]
    #             self.y = self.y[keys]

    #             X = self.X[:m]
    #             y = self.y[:m]

    #             self.model.train(X, y, epochs=min(m, i//100), batch_size=batch_size, print_every_epoch=min(m, i//100))