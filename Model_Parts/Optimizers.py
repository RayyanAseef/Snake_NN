import numpy

class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2

        self.iterations = 0
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1 + self.decay*self.iterations) )
    
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = numpy.zeros_like(layer.weights)
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_momentums = numpy.zeros_like(layer.biases)
            layer.bias_cache = numpy.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1)) 
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (numpy.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (numpy.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        self.iterations += 1