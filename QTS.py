import numpy as np
import math
from pathlib import Path
import matplotlib as plt

class QTS:
    class QObjeto:
        """Abstracción de la representación de un qubit como un "objeto cuántico" del problema de la mochila.
        
        Atributos
        ----------
        valor : int
            Valor del objeto representado por el qubit.
        peso : int
            Peso del objeto representado por el qubit.
        alpha : float, opcional
            Valor alpha de la representación del qubit (por defecto math.sqrt(1/2)).
        beta : float, opcional
            Valor beta de la representación del qubit (por defecto math.sqrt(1/2)).
        """
        
        def __init__(self, valor=None, peso=None, alpha=math.sqrt(1/2), beta=math.sqrt(1/2)):
            self.alpha = alpha
            self.beta = beta
            self.valor = valor
            self.peso = peso
        
        def medir(self):
            """Mide el valor del qubit comparando con un número aleatorio entre [0,1).
            
            Devuelve
            -------
            medicion : int
                0 o 1 dependiendo del qubit (alpha y beta) y el número aleatorio generado.
            """
            return 1 if np.random.random_sample() < self.beta**2 else 0
        
        def actualizar(self, matriz):
            """Actualiza los valores alpha y beta del qubit aplicando una matriz.
            
            Parámetros
            ----------
            matriz :
                Matriz bidimensional (compuerta cuántica) para aplicar al qubit.
            """
            alpha_old = self.alpha
            beta_old = self.beta
            self.alpha = matriz[0][0] * alpha_old + matriz[0][1] * beta_old
            self.beta = matriz[1][0] * alpha_old + matriz[1][1] * beta_old

    def crear_matriz_rotacion(self,angulo):
        """Genera una matriz de rotación para operar en un QObjeto con el ángulo dado."""
        return [[math.cos(angulo), -math.sin(angulo)], [math.sin(angulo), math.cos(angulo)]]

    def medir_poblacion(self,poblacion_q):
        """Mide cada qubit de la población de ObjetosCuanticos y devuelve el resultado."""
        return [q.medir() for q in poblacion_q]


    def evaluar_solucion(self,poblacion_q, solucion):
        """Evalúa el valor y peso de la solución dada.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        solucion : [int]
            Solución obtenida de una medición de la población.
            
        Devuelve
        -------
        valor : int
            Valor total de la solución evaluada.
        peso : int
            Peso total de la solución evaluada.
        """
        valor_total = 0
        peso_total = 0

        for i, medida in enumerate(solucion):
            valor_total += medida*(poblacion_q[i].valor)
            peso_total += medida*(poblacion_q[i].peso)
        
        return valor_total, peso_total

    def reparar_solucion(self,poblacion_q, solucion, capacidad_max, valor_actual, peso_actual):
        """Repara la solución para hacerla válida.
        Si la suma de los pesos excede el límite, 
        elimina objetos al azar hasta satisfacer la restricción.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        solucion : [int]
            Solución obtenida de una medición de la población.
        capacidad_max : int
            Capacidad máxima de peso de la mochila.
        valor_actual : int
            Valor total de la solución a reparar.
        peso_actual : int
            Peso total de la solución a reparar.
        
        Devuelve
        -------
        valor_actual : int
            Valor total de la solución reparada.
        peso_actual : int
            Peso total de la solución reparada.
        """
        while peso_actual > capacidad_max:
            indice = np.random.randint(0, len(solucion))
            if solucion[indice]:
                solucion[indice] = 0
                valor_actual -= poblacion_q[indice].valor
                peso_actual -= poblacion_q[indice].peso
        
            # Luego intenta rellenar objetos que quepan, de forma codiciosa
        #capacidad_restante = capacidad_max - peso_actual

        # Buscar candidatos no incluidos que quepan
        #candidatos = [
            #(i, q.valor / q.peso if q.peso > 0 else 0)
            #for i, (q, sel) in enumerate(zip(poblacion_q, solucion))
            #if sel == 0 and q.peso <= capacidad_restante
        #]

        # Ordenar por valor/peso descendente
        #candidatos.sort(key=lambda x: x[1], reverse=True)

        #for i, _ in candidatos:
            #if poblacion_q[i].peso <= capacidad_restante:
                #solucion[i] = 1
                #valor_actual += poblacion_q[i].valor
                #peso_actual += poblacion_q[i].peso
                #capacidad_restante -= poblacion_q[i].peso
        
        anyadido = True
        while anyadido:
            anyadido = False
            for i in range(len(solucion)):
                if solucion[i] == 0 and peso_actual + poblacion_q[i].peso <= capacidad_max:
                    solucion[i] = 1
                    valor_actual += poblacion_q[i].valor
                    peso_actual += poblacion_q[i].peso
                    anyadido = True
                    break

        return valor_actual, peso_actual

    def evaluar_y_reparar(self,poblacion_q, solucion, capacidad_max):
        """Evalúa (valor y peso) y repara una solución.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        solucion : [int]
            Solución obtenida de una medición de la población.
        capacidad_max : int
            Capacidad máxima de peso de la mochila.
        
        Devuelve
        -------
        valor_total : int
            Valor total de la solución evaluada y reparada.
        peso_total : int
            Peso total de la solución evaluada y reparada.
        """
        valor_total, peso_total = self.evaluar_solucion(poblacion_q, solucion)
        if peso_total > capacidad_max:
            valor_total, peso_total = self.reparar_solucion(poblacion_q, solucion, capacidad_max, valor_total, peso_total)
        return valor_total, peso_total

    def obtener_vecindario(self,poblacion_q, tamano_poblacion):
        """Mide la población de ObjetosCuanticos para generar un vecindario de soluciones.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        tamano_poblacion : int
            Número de vecindarios de soluciones a generar.
        
        Devuelve
        -------
        [solucion : [int]]
            Lista de soluciones vecinas.
        """
        return [[q.medir() for q in poblacion_q] for _ in range(tamano_poblacion)]

    def evaluar_y_reparar_vecindario(self,poblacion_q, vecindario, capacidad_max):
        """Evalúa (valor y peso) y repara todas las soluciones vecinas.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        vecindario : [[int]]
            Lista de soluciones vecinas.
        capacidad_max : int
            Capacidad máxima de peso de la mochila.
        
        Devuelve
        -------
        [[solucion : [int], valor : int, peso : int]]
            Lista de soluciones reparadas y su evaluación.
        """
        soluciones = []
        for solucion in vecindario:       
            soluciones.append([solucion, *self.evaluar_y_reparar(poblacion_q, solucion, capacidad_max)])
        return soluciones

    def actualizar_estado(self,poblacion_q, angulo, sol_actual, solucion_comparacion, es_mejor,lista_tabu,tabu_itt):
        """Actualiza cada qubit de la población aplicando la matriz 
        según la solución actual y la solución de comparación.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        angulo : float
            Ángulo usado para construir la matriz de rotación.
        lista_tabu : dict {int : int}
            Lista tabú (implementada como un diccionario).
        tabu_itt : int
            Número de iteraciones que un ítem debe permanecer en la lista tabú.
        solucion_actual : [int]
            Solución actual de una medición de la población.
        solucion_comparacion : [int]
            Solución para comparar con la actual.
        es_mejor : bool
            True si la solución de comparación fue la mejor encontrada.
        """
        for i, q in enumerate(poblacion_q):
            
            if lista_tabu.setdefault(i, 0) == 0:
                continue
            diferencia = solucion_comparacion[i] - sol_actual[i]
            if diferencia == 0:
                lista_tabu[i] = tabu_itt
                continue
            if not es_mejor: 
                diferencia *= -1
            if q.alpha * q.beta < 0:
                diferencia *= -1
            
            q.actualizar(self.crear_matriz_rotacion(angulo*diferencia))
            
            


    def busqueda_tabu_cuantica(self,iteraciones, angulo, tamano_poblacion,itt_tabu,archivo):
        """Ejecuta el algoritmo de búsqueda tabú inspirada en cuántica (QTS).
        
        Parámetros
        ----------
        iteraciones : int
            Número de iteraciones para ejecutar el algoritmo.
        angulo : float
            Ángulo usado para construir la matriz de rotación.
        tamano_poblacion : int
            Tamaño de la población de vecindarios a generar.
        iteraciones_tabu : int
            Número de iteraciones que un ítem debe permanecer en la lista tabú.
        archivo: Path
            Archivo de instancia del problema de la mochila.
        
        Devuelve
        -------
        mejor_sol : [solucion : [int], valor : int, peso : int]
            Mejor solución encontrada.
        mejor_iter : int
            Iteración donde se encontró la mejor solución.
        """
        
        poblacion_q = []
        lista_tabu = dict()
        capacidad_max = 0
        num_items = 0
        optimo = 0
        solucion_actual = None
        
        mejor_sol = []
        mejor_iter = -1

        with open(archivo) as f:
            num_items = int(f.readline().split()[1])
            capacidad_max = int(f.readline().split()[1])
            optimo = int(f.readline().split()[1])
            f.readline()
            for linea in f:
                _, valor, peso, _ = list(map(int, linea.split(',')))
                poblacion_q.append(self.QObjeto(valor, peso))
        
        solucion_actual = self.medir_poblacion(poblacion_q)
        valor_actual, peso_actual = self.evaluar_y_reparar(poblacion_q, solucion_actual, capacidad_max)
        
        mejor_sol = [solucion_actual, valor_actual, peso_actual]
        #historial de soluciones para hacer la comparativa entre algoritmos
        historial_soluciones = [mejor_sol[1]]
        contador_iter = 0
        iter_sin_cambio = 0
        while contador_iter < iteraciones:
            contador_iter += 1
            vecindario_poblacion = self.obtener_vecindario(poblacion_q, tamano_poblacion)
            vecindario = self.evaluar_y_reparar_vecindario(poblacion_q, vecindario_poblacion, capacidad_max)
            mejor_vecino = max(vecindario, key=lambda x: x[1])
            peor_vecino = min(vecindario, key=lambda x: x[1])
            
            encontro_mejor = (
                mejor_vecino[1] > mejor_sol[1] or 
                (mejor_vecino[1] == mejor_sol[1] and mejor_vecino[2] < mejor_sol[2])
            )
            
            if encontro_mejor:
                mejor_sol = mejor_vecino[:]
                mejor_iter = contador_iter
                iter_sin_cambio = 0
            else: 
                iter_sin_cambio +=1

            for key, value in list(lista_tabu.items()):
                lista_tabu[key] -= 1
                if lista_tabu[key]==0:
                    del lista_tabu[key]
            
            historial_soluciones.append(mejor_sol[1])
            self.actualizar_estado(poblacion_q, angulo, solucion_actual, mejor_sol[0], True,lista_tabu,self.itt_tabu)
            solucion_actual = self.medir_poblacion(poblacion_q)
            
            self.actualizar_estado(poblacion_q, angulo/3, solucion_actual, peor_vecino[0], False,lista_tabu,self.itt_tabu)
            solucion_actual = self.medir_poblacion(poblacion_q)
            

        return mejor_sol, mejor_iter, historial_soluciones
    

    def __init__(self,iteraciones,theta,tamano_poblacion,itt_tabu):
        self.iteraciones = iteraciones
        self.theta = theta
        self.tamano_poblacion = tamano_poblacion
        self.itt_tabu = itt_tabu


    def run(self,instancia_mochila):
        return self.busqueda_tabu_cuantica(self.iteraciones,self.theta,self.tamano_poblacion,self.itt_tabu,instancia_mochila)
    

#instancia_mochila = Path('./data/toyProblemInstance_100.csv')
#instancia_mochila = Path('./data/toyProblemInstance_250.csv')
instancia_mochila = Path('./data/toyProblemInstance_500.csv')
#instancia_mochila = Path('data/knapPI_11_500_1000_1.csv')




#qt = QTS(1000, 0.01 * math.pi, 10,2)
#mejor_sol, mejor_it, historial_qts = qt.run(instancia_mochila)
#print(mejor_sol,mejor_it)