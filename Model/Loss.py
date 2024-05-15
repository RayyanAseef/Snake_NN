import numpy 
from Model.Activations import Softmax

class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output, y, *,  include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = numpy.mean(sample_losses)

        self.accumulated_sum += numpy.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
        
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * numpy.sum(numpy.abs(layer.weights))
                
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * numpy.sum(layer.weights * layer.weights)
            
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * numpy.sum(numpy.abs(layer.biases))
            
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * numpy.sum(layer.biases * layer.biases)
        
        return regularization_loss

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = numpy.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = numpy.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -numpy.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = numpy.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Softmax_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = numpy.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()

        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples