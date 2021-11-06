class PenOrRing:

    def __init__(self, data: list, params: list, learn_const: float):
        self.x_weight = params[0]
        self.y_weight = params[1]
        self.z_weight = params[2]
        self.learn_const = learn_const
        self.data = data

    def network_cell(self, item: list) -> str:
        _ratio = (self.x_weight * item[0] + self.y_weight * item[1]) + self.z_weight
        if _ratio > 0:
            return "ring"
        else:
            return "pen"

    def learn(self, item: list):
        if item[2] == "ring":
            self.x_weight += item[0] * self.learn_const
            self.y_weight += item[1] * self.learn_const
            self.z_weight += 1 * self.learn_const
        else:
            self.x_weight += -1 * item[0] * self.learn_const
            self.y_weight += -1 * item[1] * self.learn_const
            self.z_weight += -1 * self.learn_const

    def evaluate(self):
        results = [self.network_cell(self.data[i]) for i in range(len(self.data))]
        return [results[i] == self.data[i][2] for i in range(len(self.data))]

    def ratio(self):
        correct_counter = 0
        for guess in self.evaluate():
            if guess:
                correct_counter += 1
        return correct_counter

    def __get_index(self, logic_array):
        index = 0
        for value in logic_array:
            if not value:
                return index
            else:
                index += 1
        if index > len(logic_array):
            return -1
        else:
            return index

    def machine_learning(self, ceiling):
        loop_counter = 0
        while self.ratio() < ceiling and loop_counter < 10:
            results = self.evaluate()
            next_index = self.__get_index(results)
            if next_index >= 0:
                self.learn(self.data[next_index])
            else:
                break
            loop_counter += 1

    def get_params(self):
        return [self.x_weight, self.y_weight, self.z_weight]

    def get_learn_const(self):
        return self.learn_const
