# Paso 3: Polinomios Ortogonales

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def ajuste_chebyshev_mejorado(x_data, y_data, grado=3):
    """
    Realiza el ajuste de datos usando una combinación de polinomios de Chebyshev
    y splines para mejor precisión

    Parámetros:
    x_data: array de valores x
    y_data: array de valores y
    grado: grado del polinomio (default: 3)
    """

    # Validación de entrada
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("Los arrays de datos no pueden estar vacíos")
    if len(x_data) != len(y_data):
        raise ValueError("Los arrays x_data y y_data deben tener la misma longitud")

    # Normalizar los datos al intervalo [-1, 1] para mejor ajuste
    x_norm = 2 * (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) - 1

    # Calcular coeficientes de Chebyshev
    coeffs = np.polynomial.chebyshev.chebfit(x_norm, y_data, grado)

    # Generar valores para la curva suave
    x_smooth = np.linspace(min(x_data), max(x_data), 200)
    x_smooth_norm = 2 * (x_smooth - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) - 1
    y_smooth = np.polynomial.chebyshev.chebval(x_smooth_norm, coeffs)

    # Usar spline para suavizar aún más la curva
    spl = make_interp_spline(x_smooth, y_smooth, k=3)
    x_final = np.linspace(min(x_data), max(x_data), 300)
    y_final = spl(x_final)

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o', color='blue', markersize=8,
             label='Datos Originales')
    plt.plot(x_final, y_final, '-', color='red', linewidth=2,
             label=f'Ajuste Polinomial (Grado {grado})')

    plt.title('Ajuste Optimizado de la Trayectoria')
    plt.xlabel('Posición (x)')
    plt.ylabel('Altura (y)')
    plt.legend()
    plt.grid(True)

    # Calcular error
    y_pred = np.polynomial.chebyshev.chebval(x_norm, coeffs)
    error = np.sqrt(np.mean((y_data - y_pred)**2))

    return x_final, y_final, error

def calcular_error(x_data, y_data, grado):
    """Calcula solo el error para un grado dado sin graficar"""
    x_norm = 2 * (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) - 1
    coeffs = np.polynomial.chebyshev.chebfit(x_norm, y_data, grado)
    y_pred = np.polynomial.chebyshev.chebval(x_norm, coeffs)
    return np.sqrt(np.mean((y_data - y_pred)**2))


def main():
    # Datos de prueba
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0.5, 0.8, 1.0, 0.9, 1.2, 0.7])

    # Calcular errores para diferentes grados
    grados = [3,4,5]
    for grado in grados:
        error = calcular_error(x_data, y_data, grado)
        print(f"Error RMS para grado {grado}: {error:.6f}")

    # Realizar el ajuste final y graficar solo una vez
    x_final, y_final, error = ajuste_chebyshev_mejorado(x_data, y_data, grado=3)

    plt.show()

if __name__ == "__main__":
    main()
