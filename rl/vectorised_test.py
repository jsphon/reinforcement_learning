"""
KERAS_BACKEND=theano THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python vectorized_test.py

"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

INPUT_SIZE = 640


def get_example_model():
    result = Sequential()
    result.add(Dense(164, init='lecun_uniform', input_shape=(INPUT_SIZE,)))
    result.add(Activation('relu'))
    # result.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    result.add(Dense(150, init='lecun_uniform'))
    result.add(Activation('relu'))
    # model.add(Dropout(0.2))

    result.add(Dense(15000, init='lecun_uniform'))
    result.add(Activation('relu'))

    result.add(Dense(15000, init='lecun_uniform'))
    result.add(Activation('relu'))

    result.add(Dense(4, init='lecun_uniform'))
    result.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    rms = RMSprop()
    result.compile(loss='mse', optimizer=rms)
    return result


if __name__=='__main__':

    from timer import Timer
    import numpy as np

    model = get_example_model()

    num_samples = 10
    #Ns = np.ndarray(num_samples)
    batch_times = np.ndarray(num_samples)
    loop_times = np.ndarray(num_samples)

    Ns = [int(x) for x in np.logspace(1, 3, num_samples)]

    for i, N in enumerate(Ns):

        state = np.random.randn(N, INPUT_SIZE)

        with Timer('%i in 1 go'%N) as t0:
            prediction = model.predict(state)

        batch_times[i] = t0.timetaken

        with Timer('%s separately'%N) as t1:
            prediction2 = np.ndarray((N, 4), dtype=prediction.dtype)
            for j in range(N):
                prediction2[j] = model.predict(state[j, :].reshape(1, INPUT_SIZE))

        loop_times[i] = t1.timetaken

        print('Running %i iterations is %0.2f times faster when done as a single batch'%(N, t1.timetaken/t0.timetaken))

        np.testing.assert_array_almost_equal(prediction, prediction2, decimal=4)

    ratios = loop_times/batch_times
    print('ratios are:')
    print(ratios)
    # plt.figure()
    # plt.plot(Ns, batch_times, legend='batch')
    # plt.plot(Ns, loop_times, legend='loop')
    # plt.show()