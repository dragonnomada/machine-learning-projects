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
5. Calcula la probabilidad de que a una persona le guste el fútbol como `P(F) = P(H) * P(F/H) + P(M) * P(F/M) + ...`. **Nota**: Pueden hacer el conteo de a cuántas personas les gusta el fútbol respecto al total.
6. Calcula por cada género la probabilidad de que la persona sea del género, dado que le gusta el fútbol, ejemplo, `P(H/F) = P(F/H) * P(H) / P(F)`.

> Estructuración de Datos

1. Ahora segmenta los datos por rangos de edades: `R1: De 0 a 12 años`, `R2: De 13 a 18 años`, `R3: de 19 a 25 años`, `R4: de 26 a 35`, `R5: de 36 a 50 años`, `R6: de 51 años en adelante`.
2. Repite el estudio por rangos de edades y anota las probabilidades de que les guste el fútbol por cada género, ejemplo: `R1: P(F/H) P(F/M) ...`, `R2: P(F/H) P(F/M) ...`.
3. Arma una tabla con las siguientes columnas: `Rango de Edad`, `Proba. Gusta Fútbol - Hombre`, `Proba. Gusta Fútbol - Mujer`, `...`
4. Incluye el rango de edad `Todos` en la tabla.
5. Exporta la tabla a un archivo `CSV`.

> Visualización de Datos

1. Por cada género, haz una gráfica de `barras` con el eje `x` como el rango de edad y el eje `y` como la probabilidad que le guste el fútbol.
2. Crea una gráfica de `caja` para comparar cómo se distribuyen las probabilidades que a una persona le guste el fútbol por género. https://www.python-graph-gallery.com/boxplot/
3. Por cada rango de edad haz una gráfica que de `dona` con el eje `x` como el género y el eje `y` como la probabilidad que le guste el fútbol. https://www.python-graph-gallery.com/donut-plot/

## Proyecto B - Clasificadores

### Introducción al proyecto - Movimientos del caballo en una partida de Ajedrez

En el ajedrez, la pieza del caballo se puede mover en L desde su posición, por ejemplo, a una izquierda y dos arriba (LUU), dos izquierdas y una arriba (LLU), dos izquierdas y una abajo (LLD), una izquierda y dos abajo (LDD), una derecha y dos abajo (RDD), dos derechas y una abajo (RRD), dos derechas y una arriba (RRU) o una derecha y dos arriba (RUU). Estos posibles 8 movimientos permiten cambiar de posición al caballo. Si la posición a la que se pretende mover el caballo tiene una pieza de color opuesto que no sea el rey de color opuesto, entonces la captura, si hay una pieza del mismo color, entonces no se puede mover ahí, si está el rey de color opuesto entonces da *jaque*, si la casilla está vacía se mueve a ella y si la casilla no está en el tablero, no puede moverse ahí.

Podemos decir que la percepción del caballo consta de 8 posiciones (las casillas a las que se pretende mover), estas casillas pueden contener `x` que significa que la casilla no existe en el tablero, `-` que la casilla está libre, `p+` un peón color opuesto, `p*` un peón del mismo color, `c+` y `c*` caballos de diferente y mismo color, `a+` y `a*` alfiles, `t+` y `t*` torres, `d+` y `d*` damas y finalmente `r+` y `r*` reyes. Entonces, si escribimos 8 símbolos dispuestos, significaría la percepción del caballo en cada posible movimiento. Podemos además etiquetar la mejor jugada que debería tomar el caballo, por ejemplo, la etiqueta LLU le diría que esa es la mejor casilla a jugar.

Hemos generado un archivo de texto que contiene 10 mil percepciones del caballo y su mejor jugada etiquetada, el archivo contiene en cada linea un significado para cada casilla a la que puede jugar, ordenadas bajo `LUU`, `LLU`, `LLD`, `LDD`, `RDD`, `RRD`, `RRU` y `RUU`. Es decir, los primeros 8 símbolos separados por coma son los valores de las casillas respectivas. El noveno símbolo es la etiqueta de la mejor casilla que debería jugar el caballo.

> Percepción del caballo y mejor casilla codificadas

`x p+ p* - - c+ - d* RRD`

En el ejemplo de arriba podemos ver que la casilla `LUU` no está en el tablero, las casillas `LDD`, `RDD` y `RRU` están vacías, hay un peón enemigo en `LLD` y un peón aliado en `LLD`, finalmente hay un caballo enemigo en `RRD` y una dama aliada en `RRU`. La mejor opción del caballo es atacar al caballo enemigo y capturarlo, es decir, moverse a la casilla `RRD`.


### Descripción del proyecto

> Adquisición, limpieza y estructuración de las muestras

1. Abre el archivo `knight-moves.txt`. Pista `f = open("knight-moves.txt")`
2. Recorre cada linea. Pista `for line in f`
3. Separa la línea por espacios. Pista `parts = line.split(" ")`
4. Obtén los primero 8 símbolos y la etiqueta. Pista `cells = parts[:8]` y `label = parts[8]`
5. Convierte los primeros 8 símbolos en un vector que codifique los símbolos de la siguiente manera `x: -1`, `-: 0`, `p+: 1`, `p*: -1`, `c+: 2`, `c*: -1`, `a+: 3`, `a*: -1`, `t+: 4`, `t*: -1`, `d+: 5`, `d*: -1`, `r+: 6` y `r*: -1`.
6. Para clasificador por Árboles de Decisión y Soporte Vectorial. Convierte la etiqueta en un valor codificada de la siguiente manera: `LUU: 0`, `LLU: 1`, `LLD: 2`, `LDD: 3`, `RDD: 4`, `RRD: 5`, `RRU: 6` y `RUU: 7`
6*. Para clasificador por Redes Neuronales. Convierte la etiqueta en un valor codificada de la siguiente manera: `LUU: [1, 0, 0, 0, 0, 0, 0, 0]`, `LLU: [0, 1, 0, 0, 0, 0, 0, 0]`, `LLD: [0, 0, 1, 0, 0, 0, 0, 0]`, `LDD: [0, 0, 0, 1, 0, 0, 0, 0]`, `RDD: [0, 0, 0, 0, 1, 0, 0, 0]`, `RRD: [0, 0, 0, 0, 0, 1, 0, 0]`, `RRU: [0, 0, 0, 0, 0, 0, 1, 0]` y `RUU: [0, 0, 0, 0, 0, 0, 0, 1]`.
7. Guarda cada muestra `x` en `X` y cada etiqueta `y` en `Y`.

> Preparación de las muestras

1. Separa las muestras en `X_train`, `X_test`, `Y_train`, `Y_test` con `sklearn.model_selection.train_test_split(X, Y)`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

> Clasificación de las muestras

1. Crea un clasificador por Árbol de Decisión con `sklearn.tree.DecisionTreeClassifier`. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontree#sklearn.tree.DecisionTreeClassifier
2. Muestra el `score` para `X_test`, `Y_test`.
3. Pinta el árbol generado usando `sklearn.tree.plot_tree`.

> Validación

1. Recorre cada muestra `X_test`, `Y_test`.
2. Para cada muestra `x`, `y` predice la etiqueta `yp` según el clasificador.
3. Codifica el vector `x` de forma inversa. `-1: *`, `0: -`, `1: p+`, `2: c+`, `3: a+`, `4: t+`, `5: d+`, `6: r+`.
4. Codifica la etiqueta `y` y `yp` de forma inversa: `[1, 0, 0, 0, 0, 0, 0, 0]: LUU`, `...`
5. Imprime una línea con la codificación de `x`, `y` y `yp` separados por un espacio.
6. Escribe un archivo llamado `knight-moves-predict.txt` con cada línea.
7. Escribe al final de la línea: `Score: {score}`.
8. Cuenta los `corrects` y los `fails`.
9. Escribe después del `score` una línea con `Corrects: {corrects}, Fails: {fails}, Total: {total}`.
10. Calcula el porcentaje de `corrects` y el porcentaje de `fails`.
11. Escribe en una línea al final `{pct_corrects} / {pct_fails}`

Opcional: Repite el mismo estudio usando un clasificador por red neuronal de perceptrón multicapa y/o un clasificador por soporte vectorial.

MLPClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlp#sklearn.neural_network.MLPClassifier

SVC - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC