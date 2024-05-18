from Model import Model
from Agent import Custom_Agent, QLearningAgent

agent = QLearningAgent()

# agent.model = Model.load('Model1.model')
agent.init()

agent.train(100_000)
agent.model.save_model('Model1.model')