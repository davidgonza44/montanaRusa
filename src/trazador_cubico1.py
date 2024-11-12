# Paso 1: Método de Trazador Cúbico Sujeto

# trazador_cubico.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def crear_trazador_cubico(x_data, y_data, fpa=0, fpb=0):
    """
    Crea un trazador cúbico con condiciones de frontera específicas.

    Parámetros:
        x_data: array de valores x
        y_data: array de valores y
        fpa: derivada en el punto inicial (default 0)
        fpb: derivada en el punto final (default 0)

    Retorna:
        CubicSpline: objeto trazador cúbico
    """
    if len(x_data) != len(y_data):
        raise ValueError("Los arrays x_data y y_data deben tener la misma longitud")
    if len(x_data) < 2:
        raise ValueError("Se necesitan al menos dos puntos para crear el trazador")

    return CubicSpline(x_data, y_data, bc_type=((1, fpa), (1, fpb)))

def generar_puntos_interpolacion(cs, x_inicial, x_final, num_puntos=100):
    """
    Genera puntos interpolados usando el trazador cúbico.

    Parámetros:
        cs: objeto CubicSpline
        x_inicial: valor inicial de x
        x_final: valor final de x
        num_puntos: número de puntos a generar

    Retorna:
        tuple: (x_vals, y_vals) puntos interpolados
    """
    x_vals = np.linspace(x_inicial, x_final, num_puntos)
    return x_vals, cs(x_vals)

def graficar_trazador(x_data, y_data, x_vals, y_vals, mostrar=True):
    """
    Grafica los datos originales y la curva interpolada.

    Parámetros:
        x_data: array de valores x originales
        y_data: array de valores y originales
        x_vals: array de valores x interpolados
        y_vals: array de valores y interpolados
        mostrar: bool, si True muestra el gráfico
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, 'o', label='Datos de Control')
    plt.plot(x_vals, y_vals, '-', label='Trazador Cúbico Sujeto')
    plt.title('Interpolación por Trazador Cúbico Sujeto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    if mostrar:
        plt.show()
    return plt.gcf()

def main():
    # Datos de ejemplo
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0.5, 0.8, 1.0, 0.9, 1.2, 0.7])

    # Crear trazador
    cs = crear_trazador_cubico(x_data, y_data)

    # Generar puntos
    x_vals, y_vals = generar_puntos_interpolacion(cs, x_data[0], x_data[-1])

    # Graficar
    graficar_trazador(x_data, y_data, x_vals, y_vals)

if __name__ == "__main__":
    main()
