import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from QTS import QTS
from QEA import QEA

# Parámetros
num_runs = 100
num_generaciones = 1000  # debe coincidir con el número de iteraciones en QTS

# Instancia del problema
instancia_mochila = Path('./data/knapPI_1_5000_1000000_1.csv')

# Crear objeto QTS una vez (o dentro del bucle si no se puede reutilizar)
historiales_qts = []
historiales_qea = []

for _ in range(num_runs):
    qt = QTS(num_generaciones, 0.01 * math.pi, 10, 2)
    qea = QEA(num_generaciones,0.01*math.pi,10,10,3)
    _,_, historial_qea = qea.run(instancia_mochila)
    _, _, historial_qts = qt.run(instancia_mochila)
    historiales_qts.append(historial_qts)
    historiales_qea.append(historial_qea)

# Convertir a array para poder hacer operaciones
#print(historiales[0])
historiales_qts_np = np.array(historiales_qts)  # shape: (num_runs, num_generaciones)
historiales_qea_np = np.array(historiales_qea)

# Calcular la media por generación
media_qts = np.mean(historiales_qts_np, axis=0)
media_qea = np.mean(historiales_qea_np, axis=0)

# Graficar
plt.plot(media_qts, marker='o', linestyle='-', color='b', label='QTS')
plt.plot(media_qea, marker='s', linestyle='--', color='r', label='QEA')

plt.title('Fitness promedio durante %s ejecuciones' % num_runs)
plt.xlabel('Generación')
plt.ylabel('Fitness promedio')
plt.grid(True)
plt.legend()
plt.show()
