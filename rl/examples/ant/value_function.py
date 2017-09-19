from rl.core.value_function import NeuralNetStateMachineActionValueFunction
from keras.layers import Input, Dense
from keras.models import Model


class AntActionValueFunction(NeuralNetStateMachineActionValueFunction):

    def __init__(self):
        super(AntActionValueFunction, self).__init__()

        input_size = 11

        # This returns a tensor
        inputs = Input(shape=(input_size,))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(inputs)
        x = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)

        model_kwargs = dict(
            optimizer='rmsprop',
            loss='mean_squared_error'
        )

        output1 = Dense(2, activation='linear', kernel_initializer='lecun_uniform')(x)
        model1 = Model(inputs=inputs, outputs=output1)
        model1.compile(**model_kwargs)

        output2 = Dense(2, activation='linear', kernel_initializer='lecun_uniform')(x)
        model2 = Model(inputs=inputs, outputs=output2)
        model2.compile(**model_kwargs)

        model = Model(inputs=inputs, outputs=[output1, output2])
        model.compile(**model_kwargs)

        self.state_models = [model1, model2]
        self.model = model

    def evaluate(self, states, targets, **kwargs):
        return self.model.evaluate(states.as_array(), targets, **kwargs)

    def vectorized_fit(self, states, targets, **kwargs):
        x = states.as_array()
        self.model.fit(x, targets, **kwargs)

    def scalar_fit(self, states, actions, rewards, **kwargs):
        pass


if __name__ == '__main__':
    from rl.examples.ant.state import AntState
    value_function = AntActionValueFunction()

    state = AntState(internal_state=0)

    print(value_function(state))

    state = AntState(internal_state=1)

    print(value_function(state))

    print(value_function.model.predict(state.external_state.as_array().reshape((1, 11))))



