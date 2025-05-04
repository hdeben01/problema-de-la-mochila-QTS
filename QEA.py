import numpy as np
import math
from pathlib import Path
import matplotlib as plt
import copy

class QEA:
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
            
            Retorna
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
            self.alpha = matriz[0][0] * self.alpha + matriz[0][1] * self.beta
            self.beta = matriz[1][0] * self.alpha + matriz[1][1] * self.beta

    def crear_matriz_rotacion(self,angulo):
        """Genera una matriz de rotación para operar en un QObjeto con el ángulo dado."""
        return [[math.cos(angulo), -math.sin(angulo)], [math.sin(angulo), math.cos(angulo)]]

    def medir_poblacion(self,poblacion_q):
        """Mide cada qubit de la población de ObjetosCuanticos y devuelve el resultado."""
        return [q.medir() for q in poblacion_q]
    
    def migrar(self,b,B):
        for i,(solucion,valor,peso) in enumerate(B):
            B[i] = b


    def evaluar_solucion(self,poblacion_q, solucion):
        """Evalúa el valor y peso de la solución dada.
        
        Parámetros
        ----------
        poblacion_q : [QObjeto]
            Población de ObjetosCuanticos.
        solucion : [int]
            Solución obtenida de una medición de la población.
            
        Retorna
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
        
        Retorna
        -------
        valor_actual : int
            Valor total de la solución reparada.
        peso_actual : int
            Peso total de la solución reparada.
        """
        while peso_actual > capacidad_max:
            indice = np.random.randint(0, len(solucion)-1)
            if solucion[indice]:
                solucion[indice] = 0
                valor_actual -= poblacion_q[indice].valor
                peso_actual -= poblacion_q[indice].peso
        
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
        
        Retorna
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
        
        Retorna
        -------
        [solucion : [int]]
            Lista de soluciones vecinas.
        """
        vecindario_generado = []
        for i in range(tamano_poblacion):
            vecindario_generado.append([q.medir() for q in poblacion_q[i]])
        return vecindario_generado

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
        
        Retorna
        -------
        [[solucion : [int], valor : int, peso : int]]
            Lista de soluciones reparadas y su evaluación.
        """
        soluciones = []
        index = 0
        for solucion in vecindario:     
            soluciones.append([solucion, *self.evaluar_y_reparar(poblacion_q[index], solucion, capacidad_max)])
            index += 1
        return soluciones

    def actualizar_estado(self, poblacion_q,tamano_poblacion,angulo, vecindario, b):
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
        iteraciones_tabu : int
            Número de iteraciones que un ítem debe permanecer en la lista tabú.
        solucion_actual : [int]
            Solución actual de una medición de la población.
        b : [int]
            Solución para comparar con la actual.
        """

        for poblacion in range(tamano_poblacion):
            diferencia = b[1] - vecindario[poblacion][1]
            for i, q in enumerate(poblacion_q[poblacion]):
                theta = 0
                #implementación de la lookup table del QEA
                if diferencia > 0:
                    if vecindario[poblacion][0][i] == 0 and b[0][i]:
                        theta = angulo
                    elif vecindario[poblacion][0][i] == 1 and b[0][i] == 0:
                        theta = -angulo
                q.actualizar(self.crear_matriz_rotacion(theta))
            
    def guardar_soluciones(self,vecindario,B,k,tamano_poblacion):
        combinado = vecindario + B
        combinado_sorted = sorted(combinado,key=lambda x: x[1], reverse=True)
        #comprobamos que el numero de iteraciones al menos 1
        mejores = max(1, int(tamano_poblacion * k / 100))
        return combinado_sorted[:mejores]
        
            


    def algoritmo_evolutivo_cuantico(self,iteraciones, angulo, tamano_poblacion,k,periodo_migracion,archivo):
        """Ejecuta el algoritmo de evolutivo inspirado en la cuántica (QEA).
        
        Parámetros
        ----------
        iteraciones : int
            Número de iteraciones para ejecutar el algoritmo.
        angulo : float
            Ángulo usado para construir la matriz de rotación.
        tamano_poblacion : int
            Tamaño de la población de vecindarios a generar.
        k : int
            El porcentaje de mejores soluciones que vamos a guardar en B(t) de P(t)
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
        b = []
        B = []
        capacidad_max = 0
        num_items = 0
        optimo = 0
        solucion_actual = []
        
        mejor_iter = -1

        #leemos la entrada del problema
        with open(archivo) as f:
            num_items = int(f.readline().split()[1])
            capacidad_max = int(f.readline().split()[1])
            optimo = int(f.readline().split()[1])
            f.readline()
            for linea in f:
                _, valor, peso, _ = list(map(int, linea.split(',')))
                poblacion_q.append(self.QObjeto(valor, peso))
        
        #creamos la poblacion Q(0) con los estados en superposicion de tamanyo tamano_poblacion
        poblacion_q = [copy.deepcopy(poblacion_q) for _ in range(tamano_poblacion)]

        vecindario_poblacion = self.obtener_vecindario(poblacion_q , tamano_poblacion)
        vecindario = self.evaluar_y_reparar_vecindario(poblacion_q , vecindario_poblacion, capacidad_max)
        B = self.guardar_soluciones(vecindario, B, k, tamano_poblacion)
        b = B[0]
        #historial de soluciones para hacer la comparativa entre algoritmos
        historial_soluciones = [b[1]]

        contador_iter = 0

        while contador_iter < iteraciones:
            contador_iter += 1
            vecindario_poblacion = self.obtener_vecindario(poblacion_q , tamano_poblacion)
            vecindario = self.evaluar_y_reparar_vecindario(poblacion_q , vecindario_poblacion, capacidad_max)
            self.actualizar_estado(poblacion_q,tamano_poblacion,angulo,vecindario,b)
            B = self.guardar_soluciones(vecindario, B, k, tamano_poblacion)
            
            #siempre se actualiza, si b era la mejor sol en B(t -1) también lo será en B(t)
            b = B[0]
            historial_soluciones.append(b[1])
            if(contador_iter % periodo_migracion == 0):
                self.migrar(b,B)
            
            

        return b, mejor_iter, historial_soluciones
    

    def __init__(self,iteraciones,theta,tamano_poblacion,k,periodo_migracion):
        self.iteraciones = iteraciones
        self.theta = theta
        self.tamano_poblacion = tamano_poblacion
        self.k = k
        self.periodo_migracion = periodo_migracion

    def run(self,instancia_mochila):
        return self.algoritmo_evolutivo_cuantico(self.iteraciones,self.theta,self.tamano_poblacion,self.k,self.periodo_migracion,instancia_mochila)
    


