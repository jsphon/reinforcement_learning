from rl.core.rl_system import RLSystem


from rl.examples.ant.reward_function import AntRewardFunction
from rl.examples.ant.model import AntModel
from rl.examples.ant.value_function import AntActionValueFunction
from rl.core.policy import SoftmaxPolicy

class AntWorld(RLSystem):

    def __init__(self):
        super(RLSystem, self).__init__()

        self.reward_function = AntRewardFunction()
        self.model = AntModel()
        self.action_value_function = AntActionValueFunction()
        self.policy = SoftmaxPolicy()

        #self.state_class = SimpleGridState
        #self.shape = (4, 4)

        #self.action_value_function = TabularGridActionValueFunction(self.num_actions)


if __name__=='__main__':

    from rl.examples.ant.state import AntState

    state = AntState()
    world = AntWorld()

    world.choose_action(state)