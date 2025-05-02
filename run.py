import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
from QTS import QTS

qt = QTS(1000,0.01*math.pi,10,2)
instancia_mochila = Path('./data/toyProblemInstance_250.csv')
mejor_sol, mejor_iter,historial_soluciones = qt.run(instancia_mochila)

print(mejor_sol[1], mejor_iter)

# Crear gráfico
plt.plot(historial_soluciones, marker='o', linestyle='-', color='b', label='Fitness')

# Opcional: personalizar
plt.title('Evolución del fitness')
plt.xlabel('Generación')
plt.ylabel('Valor de fitness')
plt.grid(True)
plt.legend()

# Mostrar gráfico
plt.show()