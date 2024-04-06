import torch, random, numpy

from game import SnakeGameAI   
from snake import Snake
from snakeBody import Snake_Body 
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

LR = 0.001

class Agent:
    def __init__(self):
        self_n_games = 0
        self.eplison = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def _test_direction(self, game, direction_input):
        head = game.snake.body[0]
        temp_snake = Snake(startX=head.x, startY=head.y, bodySize=1, xDir=game.snake.xDir, yDir=game.snake.yDir)
        temp_snake.changeDirection(direction_input)
        _, game_over = temp_snake.move(game.gridSize, game.applePos)
        return 1 if game_over else 0

    def get_state(self, game):
        # Inputs for straight, right, and left movements
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Test each direction
        dangers = [self.test_direction(game, direction) for direction in directions]

        # Get current snake direction
        directions = { (1, 0): (0, 1, 0, 0), (-1, 0): (1, 0, 0, 0), (0,  1): (0, 0, 1, 0), (0, -1): (0, 0, 0, 1)  }
        currentDirection = game.snake.xDir, game.snake.yDir

        dirLeft, dirRight, dirUp, dirDown = directions[currentDirection]

        # Get food postion relative to snake
        head = game.snake.body[0]
        apple = game.applePos

        foodRight = 1 if head.x < apple[0] else 0
        foodLeft = 0 if head.x < apple[0] else 1

        foodUp = 1 if head.y < apple[1] else 0
        foodDown = 0 if head.y < apple[1] else 1

        return [
            dangers[0], dangers[1], dangers[2],
            dirLeft, dirRight, dirUp, dirDown,
            foodLeft, foodRight, foodUp, foodDown
        ]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done) # popleft if max memroy is reached

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_short_memory(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.eplison = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.eplison:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move 
        final_move = agent.get_action(state_old)

        # Perform move and get new State
        reward, done, score = game.input()
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot game
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # TODO: plot


if __name__ == '__main__':
    train()
