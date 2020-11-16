import numpy as np

def f1(x):
    return (np.cos(x) + 0.05 * x**3 + np.log(2 * x**2))
  
def derivation(x, f):
    eps = 0.0000001
    return((f(x+eps) - f(x))/eps)
  
def f2(x1, x2):
    return (x1**2/np.cos(x2) + 0.05 * x2**3 + 3 * x1**3 / np.log(2 * x2**2))
  
def grad_func_2(x1, x2):
    dx1 = 2 * x1/np.cos(x2) + 9 * x1**2/np.log(2 * x2**2)
    dx2 = (np.sin(x2) * x1**2)/((np.cos(x2))**2) + 0.15 * x2**2 - 6 * x1**3/(x2 * (np.log(2 * x2**2))**2)
    return [dx1, dx2]
  
def gradient(xar, f):
    eps = 0.00000001
    dx1 = (f(xar[0] + eps, xar[1]) - f(xar[0],xar[1]))/eps
    dx2 = (f(xar[0], xar[1] + eps) - f(xar[0],xar[1]))/eps
    return round(np.sqrt(dx1**2 + dx2**2), 2)
  
def gradient_optimisation_one_dim(f):
    eps = 0.001
    x = 10
    der = derivation(x, f)
    step = 0
    while (der >= eps and step < 50):
        x = x - eps * der
        der = derivation(x, f)
        step += 1
    print('Точка минимума х = ', x, ' Значение функции = ', f1(x), ' Количество шагов = ', step)
    return round(f1(x), 2)

def gradient_optimisation_two_dim(f):
    eps = 0.01 
    xar = [4 , 10]
    derar = grad_func_2(xar[0], xar[1])
    step = 0
    while((derar[0] >= eps or derar[1] >= eps) and step < 50):
        for i in range(len(xar)):
            xar[i] = xar[i] - eps * derar[i]
        derar = grad_func_2(xar[0], xar[1])
        step += 1
    print('Точка минимума х = ', xar, ' Значение функции = ', f2(xar[0], xar[1]), ' Количество шагов = ', step)
    return round(f2(xar[0], xar[1]), 2)
