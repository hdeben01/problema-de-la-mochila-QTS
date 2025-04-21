import numpy as mp
import math
from pathlib import Path


class QObjeto:
    """ Abstracción de la representación de un objeto de la mochila que tiene los siguientes atributos:
    ------[Atributos]-------
    
    alfa: float
        valor alfa de la representación de un cúbit
    beta: float
        valor beta de la representación de un cúbit
    valor: int
        valor del objeto del problema de la mochila
    peso: int 
        peso del objeto del problema de la mochila """
    def _init_(self, valor,peso):
        self.valor = valor