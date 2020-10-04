class Visualization(object):
    def __init__(self):
        self.real_position_history = []
        self.estimated_position_history = []
        
    def add_real_position(self, pos):
        self.real_position_history.append(pos)

    def add_estimated_position(self, pos):
        self.estimated_position_history.append(pos)