# test_resolucion_ecuaciones.py
import unittest
import numpy as np
from src.resolucion_ecuaciones4 import (verificar_matriz,
                                 verificar_dimensiones,
                                 resolver_sistema,
                                 formatear_resultados)

class TestResolucionEcuaciones(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.A = np.array([[1, 2, 1],
                          [2, -1, 1],
                          [3, 1, -1]])
        self.b = np.array([4, 1, -2])
        self.solucion_esperada = np.array([1, 1, 1])  # Solución conocida

    def test_verificar_matriz(self):
        """Prueba la verificación de la matriz"""
        # Prueba matriz válida
        self.assertTrue(verificar_matriz(self.A))

        # Prueba matriz no cuadrada
        A_no_cuadrada = np.array([[1, 2], [3, 4], [5, 6]])
        with self.assertRaises(ValueError):
            verificar_matriz(A_no_cuadrada)

        # Prueba matriz singular
        A_singular = np.array([[1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            verificar_matriz(A_singular)

    def test_verificar_dimensiones(self):
        """Prueba la verificación de dimensiones"""
        # Prueba dimensiones compatibles
        self.assertTrue(verificar_dimensiones(self.A, self.b))

        # Prueba dimensiones incompatibles
        b_incompatible = np.array([1, 2])
        with self.assertRaises(ValueError):
            verificar_dimensiones(self.A, b_incompatible)

    def test_resolver_sistema(self):
        """Prueba la resolución del sistema"""
        # Prueba sistema con solución conocida
        A = np.array([[1, 0], [0, 1]])
        b = np.array([1, 1])
        solucion = resolver_sistema(A, b)
        np.testing.assert_array_almost_equal(solucion, np.array([1, 1]))

        # Prueba con el sistema del ejemplo
        solucion = resolver_sistema(self.A, self.b)
        self.assertEqual(len(solucion), 3)

        # Verificar que la solución satisface el sistema
        np.testing.assert_array_almost_equal(np.dot(self.A, solucion), self.b)

    def test_formatear_resultados(self):
        """Prueba el formateo de resultados"""
        solucion = np.array([1.2345, -2.3456, 3.4567])
        resultados = formatear_resultados(solucion)

        # Verificar número de resultados
        self.assertEqual(len(resultados), 3)

        # Verificar formato
        self.assertTrue(all(isinstance(r, str) for r in resultados))
        self.assertTrue(all("Fuerza en el punto" in r for r in resultados))
        self.assertTrue(all(len(r.split(":")[1].strip().split(".")[1]) == 4
                          for r in resultados))

    def test_sistema_completo(self):
        """Prueba integrada del sistema completo"""
        # Resolver sistema
        solucion = resolver_sistema(self.A, self.b)

        # Formatear resultados
        resultados = formatear_resultados(solucion)

        # Verificaciones
        self.assertEqual(len(resultados), 3)
        self.assertTrue(isinstance(resultados, list))
        self.assertTrue(all(isinstance(r, str) for r in resultados))

class TestCasosEspeciales(unittest.TestCase):
    def test_sistema_grande(self):
        """Prueba con un sistema de ecuaciones más grande"""
        n = 10
        A = np.eye(n) + np.ones((n, n)) * 0.1  # Matriz diagonal dominante
        b = np.ones(n)

        solucion = resolver_sistema(A, b)
        self.assertEqual(len(solucion), n)

    def test_precision_numerica(self):
        """Prueba la precisión numérica de la solución"""
        A = np.array([[1e-10, 1], [1, 1]])
        b = np.array([1, 2])

        solucion = resolver_sistema(A, b)
        np.testing.assert_array_almost_equal(np.dot(A, solucion), b)

def run_tests():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()
