# test_ajuste_chebyshev.py
import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.ajuste_chebyshev3 import ajuste_chebyshev_mejorado, calcular_error, main

class TestAjusteChebyshevMejorado(unittest.TestCase):
    """Prueba unitaria para la función ajuste_chebyshev_mejorado"""

    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.x_data = np.array([0, 1, 2, 3, 4, 5])
        self.y_data = np.array([0.5, 0.8, 1.0, 0.9, 1.2, 0.7])

    def test_ajuste_chebyshev_mejorado(self):
        """Prueba la función ajuste_chebyshev_mejorado"""
        # Ejecutar la función
        x_final, y_final, error = ajuste_chebyshev_mejorado(self.x_data, self.y_data, grado=3)

        # Verificar que los arrays de salida tienen el tamaño correcto
        self.assertEqual(len(x_final), 300)
        self.assertEqual(len(y_final), 300)

        # Verificar que el error es un número positivo
        self.assertGreaterEqual(error, 0)

        # Verificar que los valores x_final están en el rango correcto
        self.assertGreaterEqual(min(x_final), min(self.x_data))
        self.assertLessEqual(max(x_final), max(self.x_data))

    def test_valores_invalidos(self):
        """Prueba el manejo de valores inválidos"""
        with self.assertRaises(ValueError):
            ajuste_chebyshev_mejorado(np.array([]), np.array([]))
        with self.assertRaises(ValueError):
            ajuste_chebyshev_mejorado(np.array([1]), np.array([1, 2]))

class TestCalcularError(unittest.TestCase):
    """Prueba unitaria para la función calcular_error"""

    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.x_data = np.array([0, 1, 2, 3, 4, 5])
        self.y_data = np.array([0.5, 0.8, 1.0, 0.9, 1.2, 0.7])

    def test_calcular_error(self):
        """Prueba la función calcular_error"""
        # Probar con diferentes grados
        error_grado3 = calcular_error(self.x_data, self.y_data, grado=3)
        error_grado4 = calcular_error(self.x_data, self.y_data, grado=4)
        error_grado5 = calcular_error(self.x_data, self.y_data, grado=5)

        # Verificar que todos los errores son números positivos
        self.assertGreaterEqual(error_grado3, 0)
        self.assertGreaterEqual(error_grado4, 0)
        self.assertGreaterEqual(error_grado5, 0)

        # Verificar que los errores son diferentes para diferentes grados
        self.assertNotEqual(error_grado3, error_grado4)
        self.assertNotEqual(error_grado4, error_grado5)

class TestMain(unittest.TestCase):
    """Prueba unitaria para la función main"""

    def test_main_execution(self):
        """Prueba la ejecución de la función main"""
        try:
            # Desactivar la visualización de plt para las pruebas
            plt.ioff()
            main()
            plt.close('all')
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"main() raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
