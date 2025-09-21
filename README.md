# LABORATORIO 2
## CONVOLUCIÓN, CORRELACIÓN Y TRANSFORMADA DE FOURIER
### Objetivo
En este laboratorio se pretende comprender y aplicar los conceptos fundamentales del procesamiento de señales, utilizando la convolución para determinar la respuesta de un sistema discreto, la correlación como medida de similitud entre señales y la Transformada de Fourier como herramienta de análisis en el dominio de la frecuencia.
# PARTE A
Se realizo la convolucioón de las señales x(n) y h(n), tanto a mano como en python, de cada uno de los integrantes del grupo.
## A mano
### Karen
CONVOLUCIÓN
<img width="1000" height="1390" alt="image" src="https://github.com/user-attachments/assets/39e440d0-d69c-41cc-934b-e63bd866082a" />
GRÁFICA
<img width="1000" height="764" alt="image" src="https://github.com/user-attachments/assets/3cc04951-d661-4bc8-b684-101044d458c9" />
### Alissia
### Raúl
## En Python
### + Karen

```python
import numpy as np
import matplotlib.pyplot as plt

# Señales
x = np.array([1, 0, 3, 1, 6, 4, 8, 6, 8, 5])
h = np.array([5, 6, 0, 0, 8, 7, 1])

# Índices
x_n = np.arange(len(x))
h_n = np.arange(len(h))

# Convolución
y = np.convolve(x, h)
y_n = np.arange(len(y))

# Imprimir resultados
print("y[n] =", y)

# Gráficas con colores pedidos
plt.stem(y_n, y, linefmt="skyblue", markerfmt="o", basefmt="k-", label="y[n]")
plt.stem(x_n, x, linefmt="pink", markerfmt="o", basefmt="k-", label="x[n]")
plt.stem(h_n, h, linefmt="lightgreen", markerfmt="o", basefmt="k-", label="h[n]")

plt.xlabel("n")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()
```






