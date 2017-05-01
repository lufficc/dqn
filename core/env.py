# Env-related abstractions
class Env():
    def step(self, action_index):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError