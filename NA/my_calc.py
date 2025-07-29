import numpy as np

class calculator:
    def __init__(self):
        self.memory = 0 # initialize value

    # def add(self, a, b):
    #     return a + b
    # def sub(self, a, b):
    #     return a - b
    # def mul(self, a, b):
    #     return a*b
    # def div(self, a, b):
    #     if b != 0:
    #         return a/b
    #     else:
    #         raise ZeroDivisionError("Cannot divide by zero!")
    
    def add(a, b):
        return a + b
    def sub(a, b):
        return a - b
    def mul(a, b):
        return a*b
    def div(a, b):
        if b != 0:
            return a/b
        else:
            raise ZeroDivisionError("Cannot divide by zero!")
    

    def store(self, value):
        self.memory = value
    def recall(self):
        return self.memory
    def clear(self):
        self.memory = 0
