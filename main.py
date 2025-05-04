import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from QTS import QTS
from QEA import QEA
from multiprocessing import Pool, cpu_count


# Parámetros
num_runs = 100
num_generaciones = 1000
#instancia_mochila = Path('./data/toyProblemInstance_100.csv')
instancia_mochila = Path('./data/toyProblemInstance_250.csv')
#instancia_mochila = Path('./data/toyProblemInstance_500.csv')


# Función que ejecuta una corrida completa
def run_algorithms(_):
    qt = QTS(num_generaciones, 0.01 * math.pi, 10, 2)
    qea = QEA(num_generaciones, 0.01 * math.pi, 10, 100, 3)
    _, _, historial_qea = qea.run(instancia_mochila)
    _, _, historial_qts = qt.run(instancia_mochila)
    
    return historial_qts, historial_qea


if __name__ == '__main__':
    # Usar tantos procesos como núcleos disponibles
    with Pool(processes=min(cpu_count(), num_runs)) as pool:
        resultados = pool.map(run_algorithms, range(num_runs))

    # Separar los historiales en listas distintas
    historiales_qts, historiales_qea = zip(*resultados)

    # Convertir a arrays
    historiales_qts_np = np.array(historiales_qts)
    historiales_qea_np = np.array(historiales_qea)

    # Calcular medias
    media_qts = np.mean(historiales_qts_np, axis=0)
    media_qea = np.mean(historiales_qea_np, axis=0)

    # Graficar
    plt.plot(media_qts, marker='o', linestyle='-', color='b', label='QTS')
    plt.plot(media_qea, marker='s', linestyle='--', color='r', label='QEA')

    plt.title(f'Fitness promedio durante {num_runs} ejecuciones')
    plt.xlabel('Generación')
    plt.ylabel('Fitness promedio')
    plt.grid(True)
    plt.legend()
    plt.show()
