import numpy as np
import sympy as sp
from autograd import grad, jacobian
import matplotlib.pyplot as plt

class my_lorentz_data_fitting():

    def __init__(self):
        super().__init__()
        self.my_data()
        self.jac_func()
        self.my_fit()

    def my_data(self):

        self.x_data =np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        self.y_data =np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
        self.n = len(self.x_data)

    def func1(self,x):

        sr_diff = 0
        for i in range(self.n):
            sr_diff += (self.y_data[i] - x[0]/((self.x_data[i]-x[1])**2+x[2]))/((self.x_data[i]-x[1]**2)+x[2])
        return sr_diff

    def func2(self,x):

        sr_diff = 0
        for i in range(self.n):
            sr_diff += (self.y_data[i] - x[0]/((self.x_data[i]-x[1])**2+x[2]))*(self.x_data[i]-x[1])/((self.x_data[i]-x[1]**2)+x[2])**2
        return sr_diff

    def func3(self,x):

        sr_diff = 0
        for i in range(self.n):
            sr_diff += (self.y_data[i] - x[0]/((self.x_data[i]-x[1])**2+x[2]))/((self.x_data[i]-x[1]**2)+x[2])**2
        return sr_diff

    def jac_func(self):

        self.jac_func1 = jacobian(self.func1)
        self.jac_func2 = jacobian(self.func2)
        self.jac_func3 = jacobian(self.func3)
     
    def my_fit(self):

        i = 0
        error = 100
        tol = 1e-8
        maxiter = 1000
        M = 3
        N = 3

        x_0 = np.array([8.90920857e+04,7.67704384e+01,1.26408441e+03], dtype=float).reshape(N,1)
        fun_evaluate = np.array([self.func1(x_0),self.func2(x_0),self.func3(x_0)]).reshape(M,1)

        er = []
        iter = []
        while np.any(error > tol) and i < maxiter:
            func_evaluate = np.array([self.func1(x_0),self.func2(x_0),self.func3(x_0)]).reshape(M,1)
            flat_x_0 = x_0.flatten()
            jac = np.array([self.jac_func1(flat_x_0),self.jac_func2(flat_x_0),self.jac_func3(flat_x_0)])
            jac.reshape(M,N)

            self.x_new = x_0 - np.linalg.inv(jac)@func_evaluate
            error = (abs(self.x_new - x_0)/x_0)
            
            x_0 = self.x_new

            print(i)
            print(error)
            print('-'*10)

            er.append(error)
            iter.append(i)
            
            i += 1
            
        print('The Solution is: {}'.format(self.x_new))
        print("LHS of function 1: {}".format(np.around(self.func1(self.x_new),12)))
        print("LHS of function 2: {}".format(np.around(self.func2(self.x_new),12)))
        print("LHS of function 3: {}".format(np.around(self.func3(self.x_new),12)))

if __name__ == '__main__':

    simulation = my_lorentz_data_fitting()
    x_data = simulation.x_data
    y_data = simulation.y_data
    plt.scatter(x_data,y_data)
    x = np.linspace(-100,200,600)
    y = simulation.x_new[0]/((x-simulation.x_new[1])**2+simulation.x_new[2])
    plt.plot(x,y)
    plt.show()