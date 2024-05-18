from Model import Model
from Game import SnakeGame
import pygame, numpy, random
import copy

from Model_Parts.Layers import Layer_Dense
from Model_Parts.Activations import ReLU, Softmax, Linear
from Model_Parts.Loss import CategoricalCrossentropy, MSError
from Model_Parts.Optimizers import Adam
from Model_Parts.Accuracy import Accuracy_Categorical, Accuracy_Regression

class Agent:
    def __init__(self, gridsize=10, snakesize=4):
        self.model = Model()
        self.game = SnakeGame(gridsize, snakesize)

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
    
class QLearningAgent(Agent):
    def init(self, gamma=0.9, lr=0.001):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = gamma
        self.lr = lr
        self.memory = []

        self._create_model()

    def _create_model(self):
        self.model.layers=[
            Layer_Dense(11, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
            ReLU(),
            Layer_Dense(256, 3),
            Linear()
        ]
        self.model.set(loss=MSError(), accuracy=Accuracy_Regression(), optimizer=Adam(learning_rate=0.005, decay=5e-7))
        self.model.finalize()

    def remember(self, state, action, reward, next_state, dead_state):
        self.memory.append((state, action, reward, next_state, dead_state))

    def train_long_memory(self, batch_size):
        random.shuffle(self.memory)

        if batch_size < len(self.memory):
            batch = self.memory[:batch_size]
        else:
            batch = self.memory
        
        states, actions, rewards, next_states, dead_states = zip(*batch)
        self.train_step(states, actions, rewards, next_states, dead_states)
            

    def train_short_memory(self, state, action, reward, next_state, dead_state):
        self.train_step(state, action, reward, next_state, dead_state)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games

        turn = [0, 0, 0]
        if numpy.random.randint(0, 200) < self.epsilon:
            idx = numpy.random.randint(0, 3)
        else:
            prediction = self.model.predict(state)
            idx = numpy.argmax(prediction)
        
        turn[idx] = 1

        return turn
    
    def train_step(self, states, actions, rewards, next_states, dead_states):
        states = numpy.array(states)
        actions = numpy.array(actions)
        rewards = numpy.array(rewards)
        next_states = numpy.array(next_states)
        dead_states = numpy.array(dead_states)

        if states.ndim == 1:
            states = states.reshape((1, -1))
            actions = actions.reshape((1, -1))
            rewards = rewards.reshape((1, -1))
            next_states = next_states.reshape((1, -1))
            dead_states = dead_states.reshape((1, -1))

        q_val = self.model.forward(states)

        targets = copy.deepcopy(q_val)
        temp_model = copy.deepcopy(self.model)

        for action, reward, next_state, dead_state, target in zip(actions, rewards, next_states, dead_states, targets):
            q_new = reward
            if not dead_state:
                q_new = self.lr*(reward + self.gamma*numpy.max(temp_model.predict(next_state)))

            target[numpy.argmax(action)] = q_new

        self.model.calulate_loss(q_val, targets, new_pass=True)
        self.model.backward(q_val, targets)
        self.model.optimize()

    def train(self, n):
        best_score = 0
        update_percent = 0.0
        for i in range(n):
            state = self.get_state(self.game)
            action = self.get_action(state)

            deadState, reward = self.game.input(action)
            next_state = self.get_state(self.game)

            self.train_short_memory(state, action, reward, next_state, deadState)
            self.remember(state, action, reward, next_state, deadState)

            if deadState:
                if self.game.score > best_score:
                    best_score = self.game.score
                    print(f"Best Score: {best_score}")
                
                self.game.reset()
                self.train_long_memory(1000)

            if i / n > update_percent:
                print(f"----------------- Progress: {update_percent*100}% --------------------")
                update_percent += 0.2
                self.train_long_memory(min(10000, len(self.memory)))
        
        print(f"----------------- Progress: {update_percent*100}% --------------------")
                

class Custom_Agent(Agent):
    def init(self):
        self.X = []
        self.y = []

        self._create_model()

    def _create_model(self):
        self.model.layers=[
            Layer_Dense(11, 32, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
            ReLU(),
            Layer_Dense(32, 3),
            Softmax()
        ]
        self.model.set(loss=CategoricalCrossentropy(),accuracy=Accuracy_Categorical(), optimizer=Adam(learning_rate=0.005, decay=5e-7))
        self.model.finalize()
        
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

        deadState, _ = self.game.input(turn)
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