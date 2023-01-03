import numpy as np
import matplotlib.pyplot as plt
from autograd import grad, jacobian


func1 = lambda x: x[0] - 2*x[1] + 3*x[2] - 7
func2 = lambda x: 2*x[0] - x[1] + x[2] - 4
func3 = lambda x: -3*x[0] - 2*x[1] + 2*x[2] + 10

jac_func1 = jacobian(func1)
jac_func2 = jacobian(func2)
jac_func3 = jacobian(func3)

 