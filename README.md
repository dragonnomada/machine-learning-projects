# Proyectos de Machine Learning

Alan Badillo Salas dragonnomada123@gmail.com

## Introducción

Los siguientes proyectos describen problemas relacionados a categorías listadas.

* Uso de _numpy_ y _pandas_
* Visualización de datos y generación de reportes
* Regresión de datos y ajuste de modelos
* Clasificación de datos por árboles de decisión
* Clasificación de datos por redes neuronales

## Proyecto A - Segmentación, Conteo y Visualización de Datos

### Introducción al proyecto - Método de Bayes

El teorema de Bayes nos permite calcular la probabilidad de que ocurra un evento A dado que ya ocurrió un evento B. Esto es útil cuándo sabemos qué probabilidad (por conteo simple) hay de que ocurra el evento A, el evento B y la probabilidad que ocurra el evento B dado que ya ocurrió el evento A.

En el siguiente ejemplo se muestra una tabla de probabilidades y su descripción

Probabilidad del Evento | Valor | Descripción
--- | --- | --- 
P(A) | 0,4 | Probabilidad de elaborar el producto A
P(B) | 0,3 | Probabilidad de elaborar el producto B
P(C) | 0,3 | Probabilidad de elaborar el producto C
P(D/A) | 0,02 | Probabilidad de que el producto A salga defectuoso
P(D/B) | 0,03 | Probabilidad de que el producto B salga defectuoso
P(D/C) | 0,05 | Probabilidad de que el producto C salga defectuoso
P(D) | 0,032 | Probabilidad de que un producto salga defectuoso. P(A) * P(D/A) + P(B) * P(D/B) + P(C) * P(D/C)

La regla de Bayes dice lo siguiente.

`P(X/Y) = P(Y / X) * P(X) / P(Y)`

Entonces podemos calcular la probabilida de que un producto sea A dado que salió defectuoso mediante.

`P(A/D) = P(D/A) * P(A) / P(D)`

Con esto podemos determinar la probabilida de eventos condicionados, Información que no teniamos a priori.

### Descripción del proyecto

El siguiente proyecto consiste en segmentar, contar y calcular las probabilidades que a una persona le guste el fútbol según su género y su rango de edad.

> Segmentación y Conteo

1. Construye un dataset de 100 valores con las siguientes columnas: `Nombre`, `Edad`, `Género`, `Le gusta el Fútbol`, `Le gusta el Teatro`.
2. Cuenta cuántas personas son de cada género y calcula su probabilidad, por ejemplo, `H` - Es hombre, `P(H)` - Probabilidad de ser hombre, `M` - Es Mujer, `P(M)` - Probabilidad de ser mujer, etc. Pista. Segmenta los datos y filtra todos los del género, luego cuéntalos.
3. Cuenta a cuántos hombres les gusta el fútbol, a cuántas mujeres les gusta el fútbol y así para cada género.
4. Calcula la probabilidad por cada género de que le guste el fútbol, por ejemplo, `P(F/H)` - Probabilidad de que le gusta el futbol dado que es hombre (divide el total de hombres que les gusta el futbol entre el número de hombres), así para cada género.
5. Calcula la probabilidad de que a una persona le guste el fútbol como `P(F) = P(H) * P(F/A) + P(M) * P(F/M) + ...`
6. Calcula por cada género la probabilidad de que la persona sea del género, dado que le gusta el fútbol, ejemplo, `P(F/H) = P(H/F) * P(H) / P(F)`.

> Estructuración de Datos

1. Ahora segmenta los datos por rangos de edades: `R1: De 0 a 12 años`, `R2: De 13 a 18 años`, `R3: de 19 a 25 años`, `R4: de 26 a 35`, `R5: de 35 a 50 años`, `R6: de 51 años en adelante`.
2. Repite el estudio por rangos de edades y anota las probabilidades de que les guste el fútbol por cada género, ejemplo: `R1: P(F/H) P(F/M) ...`, `R2: P(F/H) P(F/M) ...`.
3. Arma una tabla con las siguientes columnas: `Rango de Edad`, `Proba. Gusta Fútbol - Hombre`, `Proba. Gusta Fútbol - Mujer`, `...`
4. Incluye el rango de edad `Todos` en la tabla.
5. Exporta la tabla a un archivo `CSV`.

> Visualización de Datos

1. Por cada género, haz una gráfica de `barras` con el eje `x` como el rango de edad y el eje `y` como la probabilidad que le guste el fútbol.
2. Crea una gráfica de `caja` para comparar cómo se distribuyen las probabilidades que a una persona le guste el fútbol por género. https://www.python-graph-gallery.com/boxplot/
3. Por cada rango de edad haz una gráfica que de `dona` con el eje `x` como el género y el eje `y` como la probabilidad que le guste el fútbol. https://www.python-graph-gallery.com/donut-plot/

