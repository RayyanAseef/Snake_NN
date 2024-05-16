from Model_Parts.Layers import Layer_Input, Layer_Dense
from Model_Parts.Activations import Softmax
from Model_Parts.Loss import CategoricalCrossentropy, Softmax_CategoricalCrossentropy
import numpy
import pickle
import copy

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, accuracy=None, optimizer=None):
        if loss != None:
            self.loss = loss
        if accuracy != None:
            self.accuracy = accuracy
        if optimizer != None:
            self.optimizer = optimizer

    def finalize(self):
        self.input_layer = Layer_Input()
        self.trainable_layers = []

        for i in range(len(self.layers)):
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < len(self.layers) -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if isinstance(self.layers[i], Layer_Dense):
                self.trainable_layers.append(self.layers[i])
            
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossentropy):
            self.softmax_classifier_output = Softmax_CategoricalCrossentropy()

        self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output
    
    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X)
            self.loss.calculate(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}')

    def train(self, X, y, *, epochs=1, print_every_epoch=None, print_per_epoch=None,  batch_size=None, validation_data=None):
        self.accuracy.init(y)

        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps*batch_size < len(X):
                train_steps += 1
            
        if print_per_epoch != None:
            if train_steps > print_per_epoch:
                print_per_epoch = train_steps // print_per_epoch
            else:
                print_per_epoch = 1
            
        for epoch in range(1, epochs+1):
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X)

                data_loss, reg_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + reg_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if print_every_epoch != None and print_per_epoch != None:
                    if not step % print_per_epoch and not epoch % print_every_epoch:
                        print(f'step: {step}, ' +
                            f'acc: {accuracy:.3f}, ' +
                            f'loss: {loss:.3f} (' +
                            f'data_loss: {data_loss:.3f}, ' +
                            f'reg_loss: {reg_loss:.3f}), ' +
                            f'lr: {self.optimizer.current_learning_rate}')
                    
            epoch_data_loss, epoch_reg_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_reg_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if print_every_epoch != None:
                if not epoch % print_every_epoch:
                    print(f'training(Epoch: {epoch}), ' +
                            f'acc: {epoch_accuracy:.3f}, ' +
                            f'loss: {epoch_loss:.3f} (' +
                            f'data_loss: {epoch_data_loss:.3f}, ' +
                            f'reg_loss: {epoch_reg_loss:.3f}), ' +
                            f'lr: {self.optimizer.current_learning_rate}\n')

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // prediction_steps
            if prediction_steps*batch_size < len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch = X
            else:
                batch = X[step*batch_size:(step+1)*batch_size]
            
            output.append(self.forward(batch))
        
        return numpy.vstack(output)

    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        return parameters
    
    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save_model(self, path):
        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model

