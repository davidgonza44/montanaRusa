# test_ajuste_minimos_cuadrados.py
import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.minimos_cuadrados2 import (ajustar_polinomio,
                                    generar_puntos_ajuste,
                                    calcular_error_rms,
                                    graficar_ajuste)

class TestAjusteMinimoCuadrados(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.x_data = np.array([0, 1, 2, 3, 4])
        self.y_data = np.array([1.1, 3.5, 2.8, 4.2, 5.0])
        self.grado = 2

    def test_ajustar_polinomio(self):
        """Prueba el ajuste del polinomio"""
        # Prueba ajuste normal
        polinomio = ajustar_polinomio(self.x_data, self.y_data, self.grado)
        self.assertIsNotNone(polinomio)
        self.assertEqual(polinomio.order, self.grado)

        # Prueba con arrays de diferente longitud
        with self.assertRaises(ValueError):
            ajustar_polinomio(self.x_data, self.y_data[:-1], self.grado)

        # Prueba con grado mayor que número de puntos
        with self.assertRaises(ValueError):
            ajustar_polinomio(self.x_data, self.y_data, len(self.x_data))

    def test_generar_puntos_ajuste(self):
        """Prueba la generación de puntos para la curva"""
        polinomio = ajustar_polinomio(self.x_data, self.y_data, self.grado)

        # Prueba con número de puntos por defecto
        x_vals, y_vals = generar_puntos_ajuste(polinomio, self.x_data[0], self.x_data[-1])
        self.assertEqual(len(x_vals), 100)
        self.assertEqual(len(y_vals), 100)

        # Prueba con número específico de puntos
        x_vals, y_vals = generar_puntos_ajuste(polinomio, self.x_data[0], self.x_data[-1], 50)
        self.assertEqual(len(x_vals), 50)
        self.assertEqual(len(y_vals), 50)

    def test_calcular_error_rms(self):
        """Prueba el cálculo del error RMS"""
        # Prueba con predicción perfecta
        error = calcular_error_rms(self.y_data, self.y_data)
        self.assertEqual(error, 0)

        # Prueba con predicción conocida
        y_pred = np.array([1.0, 3.4, 2.7, 4.1, 4.9])  # Valores cercanos
        error = calcular_error_rms(self.y_data, y_pred)
        self.assertGreater(error, 0)
        self.assertLess(error, 1)  # El error debe ser pequeño

    def test_graficar_ajuste(self):
        """Prueba la función de graficación"""
        polinomio = ajustar_polinomio(self.x_data, self.y_data, self.grado)
        x_vals, y_vals = generar_puntos_ajuste(polinomio, self.x_data[0], self.x_data[-1])
        y_pred = polinomio(self.x_data)
        error = calcular_error_rms(self.y_data, y_pred)

        # Prueba sin mostrar el gráfico
        fig = graficar_ajuste(self.x_data, self.y_data, x_vals, y_vals,
                            self.grado, error, mostrar=False)

        # Verificar que se creó una figura
        self.assertIsNotNone(fig)

        # Verificar elementos del gráfico
        ax = fig.gca()
        self.assertEqual(len(ax.lines), 2)  # Debe haber dos líneas
        self.assertTrue(ax.get_xlabel())
        self.assertTrue(ax.get_ylabel())

        plt.close(fig)

    def test_ajuste_casos_especiales(self):
        """Prueba casos especiales de ajuste"""
        # Datos lineales perfectos
        x_linear = np.array([0, 1, 2])
        y_linear = np.array([0, 1, 2])
        polinomio = ajustar_polinomio(x_linear, y_linear, 1)
        y_pred = polinomio(x_linear)
        error = calcular_error_rms(y_linear, y_pred)
        self.assertAlmostEqual(error, 0, places=10)

def run_tests():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()
