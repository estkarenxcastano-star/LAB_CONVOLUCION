# LABORATORIO 2
## CONVOLUCIÓN, CORRELACIÓN Y TRANSFORMADA DE FOURIER
### Objetivo
En este laboratorio se pretende comprender y aplicar los conceptos fundamentales del procesamiento de señales, utilizando la convolución para determinar la respuesta de un sistema discreto, la correlación como medida de similitud entre señales y la Transformada de Fourier como herramienta de análisis en el dominio de la frecuencia.
# PARTE A

Se realizo la convolución de las señales x(n) y h(n), tanto a mano como en python, de cada uno de los integrantes del grupo.
<img width="336" height="1280" alt="image" src="https://github.com/user-attachments/assets/0ac85b41-078e-4aca-bfb9-71c84910b7fb" />

## LIBRERIAS
Las librerias que implementamos fueron las siguientes:

+ **Importación de librerias**
```phyton
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy.signal import correlate
from numpy.fft import fft, fftfreq
```
## A mano
+ **Karen**
### CONVOLUCIÓN
<img width="1000" height="1390" alt="image" src="https://github.com/user-attachments/assets/39e440d0-d69c-41cc-934b-e63bd866082a" />

### GRÁFICA
<img width="1000" height="764" alt="image" src="https://github.com/user-attachments/assets/3cc04951-d661-4bc8-b684-101044d458c9" />

+ **Alissia**
### CONVOLUCIÓN 
![img5](https://github.com/user-attachments/assets/f7b66f0b-dff7-43e4-8338-f408b29190e7)

### GRÁFICA
![img11](https://github.com/user-attachments/assets/a5a13979-f34e-4bfe-83e3-b5b7e26d17fc)

+ **Raúl**
### CONVOLUCIÓN
![img15](https://github.com/user-attachments/assets/650571a2-26b8-4c34-9915-b5651f1a5550)

### GRÁFICA
![img19](https://github.com/user-attachments/assets/8371adad-c301-443c-8681-25e25ca1ac3b)

## En Python
+ **Karen**
```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
h_karen = [5,6,0,0,8,7,1]
x_karen = [1,0,3,1,6,4,8,6,8,5]

# Convolución
y_karen = np.convolve(x_karen, h_karen)
n_karen = np.arange(len(y_karen))

# Resultados
print("=== Karen ===")
print("x[n] =", x_karen)
print("h[n] =", h_karen)
print("y[n] =", y_karen.tolist())

# Gráfico
plt.figure(figsize=(10,4))
plt.stem(n_karen, y_karen, basefmt="k-")
plt.title("Convolución y[n] para Karen")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```
### GRÁFICA
<img width="852" height="437" alt="image" src="https://github.com/user-attachments/assets/382f5793-1779-45d7-b3bf-57beafa23c1c" />

+ **Alissia**
```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
h_alissia = [5,6,0,0,7,2,4]
x_alissia = [1,0,0,3,8,6,5,6,3,5]

# Convolución
y_alissia = np.convolve(x_alissia, h_alissia)
n_alissia = np.arange(len(y_alissia))

# Resultados
print("=== Alissia ===")
print("x[n] =", x_alissia)
print("h[n] =", h_alissia)
print("y[n] =", y_alissia.tolist())

# Gráfico
plt.figure(figsize=(10,4))
plt.stem(n_alissia, y_alissia, basefmt="k-")
plt.title("Convolución y[n] para Alissia")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```
### GRÁFICA
<img width="942" height="439" alt="image" src="https://github.com/user-attachments/assets/6e69c88e-c5ee-4079-b55d-b915515170f3" />

+ **Raúl**
```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
h_raul = [5,6,0,0,6,8,7]
x_raul = [1,1,9,3,5,1,9,6,8,5]

# Convolución
y_raul = np.convolve(x_raul, h_raul)
n_raul = np.arange(len(y_raul))

# Resultados
print("=== Raul ===")
print("x[n] =", x_raul)
print("h[n] =", h_raul)
print("y[n] =", y_raul.tolist())

# Gráfico
plt.figure(figsize=(10,4))
plt.stem(n_raul, y_raul, basefmt="k-")
plt.title("Convolución y[n] para Raul")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```
### GRÁFICA
<img width="902" height="435" alt="image" src="https://github.com/user-attachments/assets/5ba1b078-41b9-4f00-b45c-14ff6aaf6f55" />

Se realizo el cálculo de la convolución de dos señales discretas, donde la función que se utilizó para esta tarea fue `np.convolve` de la librería **Numpy**. Esta función recibe como entrada la señal x(n) y la respuesta al impulso h(n) y devuelve como salida y(n), que corresponde a una convolución líneal.

La convolución es importante porque describe como una señal se modifica cuando pasa por un sistema, esta operación se utiliza para aplicar filtros, mejorar o atenuar ciertas frecuencias, eliminar ruido, o extraer información relevante de una señal.

# PARTE B

Se realizo la correlación cruzada de dos señales dadas y se determino su importancia en el procesamiento digital de señales.
<img width="363" height="1280" alt="image" src="https://github.com/user-attachments/assets/bb076aa0-c52d-4c2b-83f3-44843574d6e1" />

- $x_1[nT_s] = \cos(2\pi 100nT_s), \quad 0 \leq n < 9$
- $x_2[nT_s] = \sin(2\pi 100nT_s), \quad 0 \leq n < 9$

  








