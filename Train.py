from Model import Model
from Agent import Agent

from Model_Parts.Layers import Layer_Dense
from Model_Parts.Activations import ReLU, Softmax
from Model_Parts.Loss import CategoricalCrossentropy
from Model_Parts.Optimizers import Adam
from Model_Parts.Accuracy import Accuracy_Categorical

agent = Agent()
agent.model = Model.load('Model.model')
# agent.create_model(
#     layers=[
#         Layer_Dense(11, 32, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
#         ReLU(),
#         Layer_Dense(32, 3),
#         Softmax()
#     ],
#     loss=CategoricalCrossentropy(),
#     accuracy=Accuracy_Categorical(),
#     optimizer=Adam(learning_rate=0.005, decay=5e-7)
# )

agent.train(10_000)
agent.model.save_model('Model1.model')