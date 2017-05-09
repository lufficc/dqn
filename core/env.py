# Env-related abstractions
class Env():
    def step(self, action_index):
        '''
        Return:
              A tuple: (state, reward, terminal, info)
        '''
        raise NotImplementedError

    def reset(self):
        '''
        Return:
              state
        '''
        raise NotImplementedError

    def render(self):
        '''render env, like show game screen
        '''
        raise NotImplementedError
