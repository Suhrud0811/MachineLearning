import numpy as np
import math
# Gradient Descent Function
def gradient_descent(x,y,iterations,learning_rate):
    m = b = 0
    n = len(x)
    for i in range(iterations):
        y_preds = m*x+b
        
        #MSE
        cost = sum((y-y_preds)**2)/len(x)
        #Partial Derivatie of m
        md = -(2/n)   * sum(x*(y-y_preds))
        bd = -(2/n) * sum((y-y_preds))
        m = m - learning_rate * md
        b = b - learning_rate * bd
        print("Iteration:{}, m:{}, b:{}, cost:{}".format(i,m,b,cost))


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
# gradient_descent(x,y,100,0.01)


# Gradient Descent with when to stop
def gradient_descent_stop(x,y,iterations,learning_rate):
    m = b = 0
    n = len(x)
    curr_cost = math.inf
    for i in range(iterations):
        y_preds = m*x+b
        
        #MSE
        cost = sum((y-y_preds)**2)/len(x)
        #Partial Derivatie of m
        md = -(2/n)   * sum(x*(y-y_preds))
        bd = -(2/n) * sum((y-y_preds))
        m = m - learning_rate * md
        b = b - learning_rate * bd
        print("Iteration:{}, m:{}, b:{}, cost:{}".format(i,m,b,cost))
        if math.isclose(curr_cost,cost,rel_tol=1e-09):
            print("Breaking")
            break
        curr_cost = cost
        


x = np.array([92,56,88,70,80,49,65,35,66,67])
y = np.array([98,68,81,80,83,52,66,30,68,73])
gradient_descent_stop(x,y,100,0.01)

        

        


    