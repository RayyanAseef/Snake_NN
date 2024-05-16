import numpy

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = numpy.maximum(inputs, 0)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax:
    def forward(self, inputs):
        exp_values = numpy.exp(inputs - numpy.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / numpy.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = numpy.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = numpy.diagflat(single_output) - numpy.dot(single_output, single_output.T)

            self.dinputs[index] = numpy.dot(jacobian_matrix, single_dvalues)
        
    def predictions(self, outputs):
        return numpy.argmax(outputs, axis=1)