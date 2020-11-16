import numpy as np

def f1(x):
    return (np.cos(x) + 0.05 * x**3 + np.log2(x**2))
  
def derivation(x, f):
    eps = 0.00000001
    return round(((f(x+eps)- f(x))/eps), 2)
  
def f2(xar):
    return ((xar[0]**2)*np.cos(xar[1]) + 0.05 * xar[1]**3 + 3 * xar[0]**3 * np.log2(xar[1]**2))
  
def grad_func_2(xar, f):
    eps = 0.00000001
    dx1 = (f([xar[0] + eps, xar[1]]) - f(xar))/eps
    dx2 = (f([xar[0], xar[1] + eps]) - f(xar))/eps
    return [dx1, dx2]
  
def gradient(xar, f):
    eps = 0.00000001
    dx1 = (f([xar[0] + eps, xar[1]]) - f(xar))/eps
    dx2 = (f([xar[0], xar[1] + eps]) - f(xar))/eps
    return round(np.sqrt(dx1**2 + dx2**2), 2)
  
def gradient_optimization_one_dim(f):
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

def gradient_optimization_multi_dim(f):
    eps = 0.01 
    xar = [4 , 10]
    derar = grad_func_2(xar, f2)
    step = 0
    while((derar[0] >= eps or derar[1] >= eps) and step < 50):
        for i in range(len(xar)):
            xar[i] = xar[i] - eps * derar[i]
        derar = grad_func_2(xar, f2)
        step += 1
    print('Точка минимума х = ', xar, ' Значение функции = ', f2(xar), ' Количество шагов = ', step)
    return round(f2(xar), 2)
