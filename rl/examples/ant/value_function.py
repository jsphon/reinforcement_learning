from rl.core.value_function import NeuralNetStateMachineActionValueFunction
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model


class AntActionValueFunction(NeuralNetStateMachineActionValueFunction):

    def __init__(self):
        super(AntActionValueFunction, self).__init__()

        input_size = 10

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

        combined_model = Model(inputs=inputs, outputs=[output1, output2])
        combined_model.compile(**model_kwargs)

        self.state_models = [model1, model2]
        self.combined_model = combined_model

    def combined_fit(self, states, targets, **kwargs):
        """
        A vectorized fit of all states at the same time.
        Args:
            states:
            targets:
            **kwargs:

        Returns:

        """

    def vectorized_fit(self, states, targets, **kwargs):
        x = states.as_array()
        self.model.fit(x, targets, **kwargs)

    def scalar_fit(self, states, actions, rewards, **kwargs):
        pass


if __name__ == '__main__':
    from rl.core.state import IntExtState, State

    class ExternalState(State):
        def __init__(self):
            self.values = np.ones(10)

        def as_array(self):
            return self.values

    value_function = AntActionValueFunction()

    external_state = ExternalState()
    int_ext_state = IntExtState(0, external_state)

    print(value_function(int_ext_state))

    int_ext_state = IntExtState(1, external_state)

    print(value_function(int_ext_state))