import numpy as np

def func_1(x):
    return (np.cos(x) + 0.05 * x**3 + np.log(2 * x**2))
  
def deriv_func_1(x):
    return(-np.sin(x) + 3 * 0.05 * x**2 + 2/x)
  
def func_2(x1, x2):
    return (x1**2/np.cos(x2) + 0.05 * x2**3 + 3 * x1**3 / np.log(2 * x2**2))
  
def grad_func_2(x1, x2):
    dx1 = 2 * x1/np.cos(x2) + 9 * x1**2/np.log(2 * x2**2)
    dx2 = (np.sin(x2) * x1**2)/((np.cos(x2))**2) + 0.15 * x2**2 - 6 * x1**3/(x2 * (np.log(2 * x2**2))**2)
    return [dx1, dx2]
  
def grad2_func_2(x1, x2):
    eps = 0.00000001
    dx1 = (func_2(x1 + eps, x2) - func_2(x1,x2))/eps
    dx2 = (func_2(x1, x2 + eps) - func_2(x1,x2))/eps
    return [dx1, dx2]
  
def graddes_f1(x , eps):
  der = deriv_func_1(x)
  step = 0
  while (der >= eps and step <= 10000):
      x = x - eps * der
      der = deriv_func_1(x)
      step += 1
  print('Точка минимума х = ', x, ' Значение функции = ', func_1(x), ' Количество шагов = ', step)
  return(x)

def graddes_f2(xar, eps):
  derar = grad2_func_2(xar[0], xar[1])
  step = 0
  while((derar[0] >= eps or derar[1] >= eps) and step <= 10000):
      for i in range(len(xar)):
          xar[i] = xar[i] - eps * derar[i]
      derar = grad2_func_2(xar[0], xar[1])
      step += 1
  print('Точка минимума х = ', xar, ' Значение функции = ', func_2(xar[0], xar[1]), ' Количество шагов = ', step)
  return(xar)

print('Производная первой функции равна = ', deriv_func_1(10))
print('Значение вектора градиента функции = ', grad_func_2(10,1), '-------', grad2_func_2(10,1))
graddes_f1(10, 0.001)
graddes_f2([4, 10], 0.001)
