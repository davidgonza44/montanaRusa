# Paso 4: Resolución de Ecuaciones

# resolucion_ecuaciones.py
import numpy as np

def verificar_matriz(A):
    """
    Verifica si la matriz A es cuadrada y tiene determinante no nulo.

    Parámetros:
        A: matriz numpy

    Retorna:
        bool: True si la matriz es válida
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")
    if np.linalg.det(A) == 0:
        raise ValueError("La matriz A debe ser no singular")
    return True

def verificar_dimensiones(A, b):
    """
    Verifica si las dimensiones de A y b son compatibles.

    Parámetros:
        A: matriz numpy
        b: vector numpy

    Retorna:
        bool: True si las dimensiones son compatibles
    """
    if A.shape[0] != len(b):
        raise ValueError("Las dimensiones de A y b deben ser compatibles")
    return True

def resolver_sistema(A, b):
    """
    Resuelve el sistema de ecuaciones Ax = b.

    Parámetros:
        A: matriz de coeficientes
        b: vector de términos independientes

    Retorna:
        numpy.array: vector solución
    """
    verificar_matriz(A)
    verificar_dimensiones(A, b)
    return np.linalg.solve(A, b)

def formatear_resultados(solucion):
    """
    Formatea los resultados de la solución.

    Parámetros:
        solucion: vector solución

    Retorna:
        list: lista de strings con los resultados formateados
    """
    resultados = []
    for i, fuerza in enumerate(solucion, 1):
        resultados.append(f"Fuerza en el punto {i}: {fuerza:.4f}")
    return resultados

def main():
    # Definir el sistema
    A = np.array([[1, 2, 1],
                  [2, -1, 1],
                  [3, 1, -1]])
    b = np.array([4, 1, -2])

    # Resolver el sistema
    try:
        solucion = resolver_sistema(A, b)

        # Mostrar resultados
        print("Resolución de Ecuaciones:")
        for resultado in formatear_resultados(solucion):
            print(resultado)

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

