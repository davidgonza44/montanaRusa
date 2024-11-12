# test_trazador_cubico.py
import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.trazador_cubico1 import (crear_trazador_cubico,
                           generar_puntos_interpolacion,
                           graficar_trazador)

class TestTrazadorCubico(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.x_data = np.array([0, 1, 2, 3, 4, 5])
        self.y_data = np.array([0.5, 0.8, 1.0, 0.9, 1.2, 0.7])

    def test_crear_trazador_cubico(self):
        """Prueba la creación del trazador cúbico"""
        # Prueba creación normal
        cs = crear_trazador_cubico(self.x_data, self.y_data)
        self.assertIsNotNone(cs)

        # Prueba con arrays de diferente longitud
        with self.assertRaises(ValueError):
            crear_trazador_cubico(self.x_data, self.y_data[:-1])

        # Prueba con arrays vacíos
        with self.assertRaises(ValueError):
            crear_trazador_cubico(np.array([]), np.array([]))

        # Prueba con un solo punto
        with self.assertRaises(ValueError):
            crear_trazador_cubico(np.array([1]), np.array([1]))

    def test_generar_puntos_interpolacion(self):
        """Prueba la generación de puntos interpolados"""
        cs = crear_trazador_cubico(self.x_data, self.y_data)

        # Prueba con número de puntos por defecto
        x_vals, y_vals = generar_puntos_interpolacion(cs, self.x_data[0], self.x_data[-1])
        self.assertEqual(len(x_vals), 100)
        self.assertEqual(len(y_vals), 100)

        # Prueba con número específico de puntos
        x_vals, y_vals = generar_puntos_interpolacion(cs, self.x_data[0], self.x_data[-1], 50)
        self.assertEqual(len(x_vals), 50)
        self.assertEqual(len(y_vals), 50)

        # Verificar que los puntos están en el rango correcto
        self.assertTrue(np.all(x_vals >= self.x_data[0]))
        self.assertTrue(np.all(x_vals <= self.x_data[-1]))

    def test_graficar_trazador(self):
        """Prueba la función de graficación"""
        cs = crear_trazador_cubico(self.x_data, self.y_data)
        x_vals, y_vals = generar_puntos_interpolacion(cs, self.x_data[0], self.x_data[-1])

        # Prueba sin mostrar el gráfico
        fig = graficar_trazador(self.x_data, self.y_data, x_vals, y_vals, mostrar=False)

        # Verificar que se creó una figura
        self.assertIsNotNone(fig)

        # Verificar que la figura tiene los elementos correctos
        ax = fig.gca()
        self.assertEqual(len(ax.lines), 2)  # Debe haber dos líneas (datos y curva)
        self.assertTrue(ax.get_xlabel())  # Debe tener etiqueta x
        self.assertTrue(ax.get_ylabel())  # Debe tener etiqueta y

        plt.close(fig)  # Limpiar la figura

    def test_interpolacion_valores(self):
        """Prueba que la interpolación pasa por los puntos originales"""
        cs = crear_trazador_cubico(self.x_data, self.y_data)
        y_interpolados = cs(self.x_data)
        np.testing.assert_array_almost_equal(y_interpolados, self.y_data)

def run_tests():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()
