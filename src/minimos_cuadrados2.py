# Paso 2: Polinomio de Mínimos Cuadrados

# ajuste_minimos_cuadrados.py
import numpy as np
import matplotlib.pyplot as plt

def ajustar_polinomio(x_data, y_data, grado):
    """
    Ajusta un polinomio de grado especificado a los datos usando mínimos cuadrados.

    Parámetros:
        x_data: array de valores x
        y_data: array de valores y
        grado: grado del polinomio

    Retorna:
        np.poly1d: objeto polinomio ajustado
    """
    if len(x_data) != len(y_data):
        raise ValueError("Los arrays x_data y y_data deben tener la misma longitud")
    if len(x_data) <= grado:
        raise ValueError("El número de puntos debe ser mayor que el grado del polinomio")

    coeficientes = np.polyfit(x_data, y_data, grado)
    return np.poly1d(coeficientes)

def generar_puntos_ajuste(polinomio, x_inicial, x_final, num_puntos=100):
    """
    Genera puntos para graficar la curva ajustada.

    Parámetros:
        polinomio: objeto np.poly1d
        x_inicial: valor inicial de x
        x_final: valor final de x
        num_puntos: número de puntos a generar

    Retorna:
        tuple: (x_vals, y_vals) puntos de la curva ajustada
    """
    x_vals = np.linspace(x_inicial, x_final, num_puntos)
    return x_vals, polinomio(x_vals)

def calcular_error_rms(y_real, y_predicho):
    """
    Calcula el error cuadrático medio.

    Parámetros:
        y_real: valores reales
        y_predicho: valores predichos

    Retorna:
        float: error RMS
    """
    return np.sqrt(np.mean((y_real - y_predicho)**2))

def graficar_ajuste(x_data, y_data, x_vals, y_vals, grado, error, mostrar=True):
    """
    Grafica los datos originales y la curva ajustada.

    Parámetros:
        x_data: array de valores x originales
        y_data: array de valores y originales
        x_vals: array de valores x de la curva
        y_vals: array de valores y de la curva
        grado: grado del polinomio
        error: error RMS del ajuste
        mostrar: bool, si True muestra el gráfico
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o', color='blue', markersize=8,
             label='Datos Experimentales')
    plt.plot(x_vals, y_vals, '-', color='red', linewidth=2,
             label=f'Polinomio de Grado {grado}')

    plt.title(f'Ajuste por Mínimos Cuadrados (Error RMS: {error:.6f})',
             fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('y', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    if mostrar:
        plt.show()
    return plt.gcf()

def main():
    # Datos de ejemplo
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([1.1, 3.5, 2.8, 4.2, 5.0])
    grado = 4

    # Ajustar polinomio
    polinomio = ajustar_polinomio(x_data, y_data, grado)

    # Generar puntos para la curva
    x_vals, y_vals = generar_puntos_ajuste(polinomio, x_data[0], x_data[-1])

    # Calcular error
    y_pred = polinomio(x_data)
    error = calcular_error_rms(y_data, y_pred)

    # Graficar
    graficar_ajuste(x_data, y_data, x_vals, y_vals, grado, error)

if __name__ == "__main__":
    main()
