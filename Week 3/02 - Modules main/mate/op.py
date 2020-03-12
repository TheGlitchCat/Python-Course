## OPERATIONS


def prom(array):
    result = 0
    for i in array:
        result += i
        
    return result / len(array)


def sumatoria(array):
    result = 0
    for i in array:
        result += i
    
    return result


def potencias_cuadrada(array):
    result = []
    for i in array:
        result.append(i**2)
    
    return result