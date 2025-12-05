// Marco Teórico: Clasificación Binaria - Modelos Lineales vs Redes Neuronales
// Formato académico para Typst
// Autor: Eduardo Farías Reyes
// Versión sin dependencias externas - 100% funcional


//#import "@preview/graceful-genetics:0.2.0" as graceful-genetics
//#import "@preview/physica:0.9.3"
#import "ef-theme.typ": template, ef-definicion, ef-teorema, ef-ejemplo, ef-corolario, ef-nota, ef-proposition, ef-algorithm, ef-primary, ef-paper-bg, ef-neutral-light, ef-electric, ef-demostracion

#show: template.with(
  title: [Clasificación Binaria: Modelos Lineales vs.~Redes Neuronales],
  authors: (
    (
      name: "Eduardo A. Farías Reyes",
      department: "Ingeniero en Informática · Docente y Data & ML Engineer",
      institution: "https://efarias.cl",
      city: "Arica",
      country: "Chile",
      mail: "contacto@efarias.cl",
    ),
    
  ),
  // date: (
  //   year: 2025,
  //   month: "Dic",
  //   day: 02,
  // ),
  keywords: (
    "MLP",
    "Redes Neuronales",
    "Regresión Logística",
    "Machine Learning",
  ),
  //doi: "10.7891/120948510",
  abstract: [
    Este documento presenta un marco teórico completo para comprender las diferencias fundamentales entre modelos lineales (Regresión Logística) y redes neuronales (Perceptrón Multicapa) en el contexto de clasificación binaria. Se enfoca específicamente en el problema de datos no-linealmente separables, usando el caso de círculos concéntricos como ejemplo pedagógico. El material está diseñado para estudiantes de nivel universitario en cursos de Minería de Datos y Machine Learning, proporcionando fundamentos matemáticos, comparaciones prácticas, y guías de evaluación de modelos.
  ],
  subtitle: [Estudio docente sobre MLP y principio de parsimonia en Machine Learning],
)


// ESTILOS PERSONALIZADOS
// ==============================================================================


= Introducción

== Contexto y Motivación

La clasificación binaria representa uno de los problemas fundamentales en Machine Learning, donde el objetivo es asignar observaciones a una de *dos clases posibles*. Mientras que modelos lineales como la Regresión Logística han demostrado efectividad en problemas linealmente separables, la creciente complejidad de datasets reales requiere enfoques más sofisticados.

Este documento surge de la necesidad de proporcionar material educativo riguroso que explique _por qué_ y _cuándo_ se requieren modelos no-lineales, específicamente redes neuronales. El enfoque pedagógico se centra en demostrar las limitaciones inherentes de modelos lineales mediante un problema geométrico concreto: los círculos concéntricos.

== Objetivos del Documento

Este marco teórico tiene como objetivos:

1. Establecer fundamentos matemáticos de clasificación binaria
2. Explicar en detalle la Regresión Logística y sus limitaciones
3. Introducir el Perceptrón Multicapa (MLP) y redes neuronales
4. Demostrar el rol crítico de las funciones de activación no-lineales
5. Proporcionar herramientas de evaluación y comparación de modelos
6. Contextualizar el problema de círculos concéntricos

== Notación Matemática

A lo largo del documento se utilizará la siguiente notación:

- $bold(x) in RR^d$: Vector de características (features) de dimensión $d$
- $y in {0, 1}$: Etiqueta de clase binaria
- $bold(w) in RR^d$: Vector de pesos (weights)
- $b in RR$: Sesgo (bias)
- $n$: Número de observaciones en el dataset
- $sigma(z)$: Función sigmoide
- $phi(z)$: Función de activación genérica
- $cal(L)$: Función de pérdida (loss function)
- $hat(y)$: Predicción del modelo

////#pagebreak()

= Clasificación Binaria

== Definición Formal

#ef-definicion[
  Un problema de *clasificación binaria* consiste en aprender una función $f: RR^d arrow.r {0, 1}$ que mapea vectores de características a etiquetas binarias, basándose en un conjunto de entrenamiento $cal(D) = {(bold(x)_i, y_i)}_(i=1)^n$.
]

El objetivo es minimizar el error de clasificación en datos no vistos, lo que se formaliza mediante:

$ min_(f in cal(F)) E_((bold(x), y) tilde cal(D)) [bb(1)_(f(bold(x)) != y)] $

donde $bb(1)$ es la función indicadora y $cal(F)$ es la clase de funciones consideradas.

== Frontera de Decisión

La *frontera de decisión* (decision boundary) es el conjunto de puntos donde el clasificador es indiferente entre ambas clases:

$ {bold(x) in RR^d : P(y=1|bold(x)) = 0.5} $

#ef-teorema("Frontera Lineal")[
  Para un clasificador lineal, la frontera de decisión es un hiperplano en $RR^d$ definido por $bold(w^T x + b = 0)$.
]

#ef-demostracion[
  Sea $f(x) = sigma(w^T x + b)$ donde $sigma$ es monótona. Entonces $P(y=1|x) = 0.5$ si y solo si $sigma(z) = 0.5$, lo cual ocurre cuando $z = 0$ para la función sigmoide. Por lo tanto, $w^T x + b = 0$ define un hiperplano.
]

== Separabilidad Lineal

#ef-definicion[
  Un dataset es *linealmente separable* si existe un hiperplano que separa perfectamente las dos clases, es decir:
  
  $ exists bold(w), b: forall i, space cases(
    bold(w)^T bold(x)_i + b > 0 quad &"si" y_i = 1,
    bold(w)^T bold(x)_i + b < 0 quad &"si" y_i = 0
  ) $
]

La mayoría de problemas reales no son linealmente separables, lo que motiva el uso de modelos no-lineales.

////#pagebreak()

= Regresión Logística

== Fundamentos Matemáticos

La Regresión Logística modela la probabilidad condicional de la clase positiva mediante:

$ P(y=1|bold(x)) = sigma(bold(w)^T bold(x) + b) = 1/(1 + e^(-(bold(w)^T bold(x) + b))) $ <eq:logistic>

=== Función Sigmoide

La función sigmoide transforma la salida lineal en probabilidades:

$ sigma(z) = 1/(1 + e^(-z)) $ <eq:sigmoid>

#ef-proposition[
  La función sigmoide satisface:
  + $sigma(0) = 0.5$
  + $lim_(z arrow.r infinity) sigma(z) = 1$
  + $lim_(z arrow.r -infinity) sigma(z) = 0$
  + $sigma'(z) = sigma(z)(1 - sigma(z))$
]

La propiedad (4) es crucial para el cálculo eficiente de gradientes durante el entrenamiento.

== Función de Pérdida

Para entrenar el modelo, se minimiza la *entropía cruzada binaria*:

$ cal(L)(bold(w)) = -1/n sum_(i=1)^n [y_i log(hat(y)_i) + (1-y_i)log(1-hat(y)_i)] $ <eq:binary-crossentropy>

donde $hat(y)_i = sigma(bold(w)^T bold(x)_i + b)$.

#ef-teorema("Convexidad")[
  La función de pérdida de entropía cruzada binaria es convexa en $bold(w)$ cuando se usa con la función sigmoide.
]

Esta propiedad garantiza que el descenso de gradiente converge al mínimo global.

== Limitaciones Fundamentales

#ef-teorema("Limitación de Modelos Lineales")[
  Sea $f(bold(x)) = sigma(bold(w)^T bold(x) + b)$ un clasificador de Regresión Logística. Entonces, la frontera de decisión de $f$ es necesariamente un hiperplano en $RR^d$.
]

#ef-corolario[
  La Regresión Logística no puede clasificar correctamente datasets no-linealmente separables con accuracy superior al azar en el caso general.
]

Esta limitación fundamental motiva el desarrollo de modelos no-lineales como las redes neuronales.

//#pagebreak()

= Perceptrón Multicapa (MLP)

== Arquitectura de Red Neuronal

Un Perceptrón Multicapa es una red neuronal feedforward compuesta por:

1. *Capa de entrada*: Recibe el vector $bold(x) in RR^d$
2. *Capas ocultas*: $L$ capas que transforman los datos
3. *Capa de salida*: Produce la predicción $hat(y)$

Una arquitectura se denota como $d$-$n_1$-$n_2$-...-$n_L$-$c$, donde $n_l$ es el número de neuronas en la capa $l$ y $c$ es el número de clases (1 para clasificación binaria).

// Insertar imágen en ancho completo
#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 1em,
)[
  #figure(
    image("mlp-diagram.png", width: 100%),
    caption: [Esquema conceptual de un perceptrón multicapa.]
  ) <fig:mlp>
  #v(12pt)
]


== Transformación en una Neurona

Cada neurona en la capa $l$ realiza:

$ a_j^((l)) = phi(z_j^((l))) = phi(sum_(i=1)^(n_(l-1)) w_(j i)^((l)) a_i^((l-1)) + b_j^((l))) $ <eq:neuron>

donde:
- $w_(j i)^((l))$: Peso de la conexión entre neurona $i$ en capa $l-1$ y neurona $j$ en capa $l$
- $b_j^((l))$: Sesgo de la neurona $j$ en capa $l$
- $phi$: Función de activación no-lineal
- $a_j^((l))$: Activación de la neurona $j$ en capa $l$

En notación matricial:

$ bold(a^l = phi(W^l a^(l-1) + b^l)) $ <eq:layer-matrix>

== Funciones de Activación

Las funciones de activación introducen no-linealidad en el modelo, permitiendo aproximar funciones complejas.

=== Tangente Hiperbólica (Tanh)

$ phi(z) = tanh(z) = (e^z - e^(-z))/(e^z + e^(-z)) $ <eq:tanh>

*Propiedades:*
- Rango: $[-1, 1]$
- Centrada en cero: $tanh(0) = 0$
- Antisimétrica: $tanh(-z) = -tanh(z)$
- Derivada: $tanh'(z) = 1 - tanh^2(z)$

=== ReLU (Rectified Linear Unit)

$ phi(z) = max(0, z) = cases(z quad & "si" z > 0, 0 quad & "si" z <= 0) $ <eq:relu>

*Propiedades:*
- Rango: $[0, infinity)$
- No saturante para $z > 0$
- Computacionalmente eficiente
- Derivada: $phi'(z) = bb(1)_(z > 0)$

=== Función Sigmoide

$ bold(phi(z) = 1/(1 + e^(-z))) $ <eq:sigmoid-activation>

*Propiedades:*
- Rango: $(0, 1)$
- Útil para capa de salida en clasificación binaria
- Puede sufrir "vanishing gradient" en capas profundas

== Teorema de Aproximación Universal

#ef-teorema("Teorema de Aproximación Universal - Cybenko 1989")[
  Sea $phi$ una función de activación continua acotada no-constante. Entonces, para cualquier función continua $f: [0,1]^d arrow.r RR$ y $epsilon > 0$, existe una red neuronal de una capa oculta con $N$ neuronas tal que:

  $ sup_(bold(x) in [0,1]^d) |f(bold(x)) - hat(f)(bold(x))| < epsilon $
]

Este teorema establece que las redes neuronales con una sola capa oculta pueden aproximar cualquier función continua con precisión arbitraria, dado suficientes neuronas.

#ef-nota[
  Aunque una capa es teóricamente suficiente, en la práctica redes más profundas aprenden representaciones más eficientes y requieren menos neuronas totales.
]

== Propagación hacia Adelante

Para una red con $L$ capas ocultas, la predicción se calcula mediante:

$ a^((0)) = x $ <eq:forward-prop-1>

$ a^((l)) = phi(W^((l)) a^((l-1)) + b^((l))), quad l = 1, ..., L $ <eq:forward-prop-2>

$ hat(y) = sigma(w^((L+1)T) a^((L)) + b^((L+1))) $ <eq:forward-prop-3>

Este proceso se conoce como _forward propagation_ (propagación hacia adelante).

== Número de Parámetros

El número total de parámetros en una red $d - n_1 - n_2 - ... - n_L - 1$ es:

$ N_("params") = sum_(l=1)^(L+1) (n_(l-1) times n_l + n_l) $ <eq:num-params>

donde $n_0 = d$ y $n_(L+1) = 1$.

#ef-ejemplo[
  Para la arquitectura 2-10-10-1 (nuestro experimento):
  - Capa 1: $(2 times 10) + 10 = 30$
  - Capa 2: $(10 times 10) + 10 = 110$
  - Capa 3: $(10 times 1) + 1 = 11$
  - *Total: 151 parámetros*
]

//#pagebreak()

= Entrenamiento de Modelos

== Función de Pérdida

Para clasificación binaria con MLP, también se usa entropía cruzada binaria:

$ cal(L)(Theta) = -1/n sum_(i=1)^n [y_i log(hat(y)_i) + (1-y_i)log(1-hat(y)_i)] $ <eq:loss-mlp>

donde $Theta$ representa todos los parámetros de la red.

== Descenso de Gradiente

El objetivo es minimizar $cal(L)(Theta)$ mediante:

$ bold(Theta)^((t+1)) = bold(Theta)^((t)) - eta nabla_(bold(Theta)) cal(L)(bold(Theta)^((t))) $ <eq:gradient-descent>

donde $eta > 0$ es el _learning rate_ (tasa de aprendizaje).

=== Variantes de Descenso de Gradiente

*1. Batch Gradient Descent:* Usa todo el dataset

$ nabla_(Theta) cal(L) = 1/n sum_(i=1)^n nabla_(Theta) cal(L)_i $

*2. Stochastic Gradient Descent (SGD):* Usa una observación

$ nabla_(Theta) cal(L) approx nabla_(Theta) cal(L)_i $

*3. Mini-batch Gradient Descent:* Usa subconjuntos (típico: 16-256)

$ nabla_(Theta) cal(L) approx 1/m sum_(i in cal(B)) nabla_(Theta) cal(L)_i $

== Backpropagation

El algoritmo de *backpropagation* (retropropagación) calcula eficientemente los gradientes en redes neuronales mediante la regla de la cadena.

=== Algoritmo

#ef-algorithm("Backpropagation")[
  *Entrada:* Dataset $cal(D)$, red con parámetros $Theta$, learning rate $eta$
  
  Para cada época:
  
  1. *Forward pass:* Calcular $hat(y)_i$ para cada $x_i$
  
  2. *Calcular pérdida:* #h(0.5em) $cal(L) = -1/n sum_i [y_i log hat(y)_i + (1-y_i)log(1-hat(y)_i)]$
  
  3. *Backward pass:* Para $l = L+1, ..., 1$:
     - Calcular $delta^((l)) = (partial cal(L))/(partial z^((l)))$
     - Calcular $nabla_(W^((l))) cal(L) = delta^((l)) (a^((l-1)))^T$
     - Calcular $nabla_(b^((l))) cal(L) = delta^((l))$
  
  4. *Actualizar:* $Theta arrow.l Theta - eta nabla_(Theta) cal(L)$
  
  *Retornar:* Parámetros optimizados $Theta$
]

=== Cálculo del Error en Capa de Salida

Para la capa de salida con sigmoide:

$ delta^((L+1)) = hat(y) - y $ <eq:output-error>

Esta forma simple resulta de la combinación de entropía cruzada con sigmoide.

=== Propagación del Error hacia Atrás

Para capas ocultas:

$ delta^((l)) = ((W^((l+1)))^T delta^((l+1))) dot.o phi'(z^((l))) $ <eq:hidden-error>

donde $dot.o$ denota producto elemento-wise (Hadamard).

#ef-nota[
  La complejidad temporal de backpropagation es $O(sum_(l=1)^(L+1) n_(l-1) times n_l)$, proporcional al número de conexiones en la red.
]

//#pagebreak()

= Métricas de Evaluación

== Matriz de Confusión

Para clasificación binaria, la matriz de confusión es:

#figure(
  table(
    columns: 3,
    align: center,
    table.header([], [Predicho 0], [Predicho 1]),
    [Real 0], [TN], [FP],
    [Real 1], [FN], [TP]
  ),
  caption: [Matriz de Confusión para Clasificación Binaria]
)

donde:
- *TP* (True Positive): Correctamente clasificados como positivos
- *TN* (True Negative): Correctamente clasificados como negativos
- *FP* (False Positive): Error Tipo I
- *FN* (False Negative): Error Tipo II

== Métricas Derivadas

=== Accuracy (Exactitud)

$ "Accuracy" = (T P + T N)/(T P + T N + F P + F N) $ <eq:accuracy>

Proporción de predicciones correctas. Útil cuando las clases están balanceadas.

=== Precision (Precisión)

$ "Precision" = (T P)/(T P + F P) $ <eq:precision>

De todos los predichos como positivos, qué proporción son correctos.

=== Recall (Sensibilidad)

$ "Recall" = (T P)/(T P + F N) $ <eq:recall>

De todos los positivos reales, qué proporción detectamos.

=== F1-Score

$ F_1 = 2 times ("Precision" times "Recall")/("Precision" + "Recall") = (2 T P)/(2 T P + F P + F N) $ <eq:f1>

Media armónica de precision y recall.

== Curva ROC y AUC

La *curva ROC* (Receiver Operating Characteristic) grafica TPR vs. FPR para diferentes umbrales.

El *AUC* (Area Under the Curve) cuantifica el rendimiento:

$ "AUC" = integral_0^1 "TPR"("FPR") dif("FPR") $ <eq:auc>

#figure(
  image("curvaROC.png"),
  caption: [
    Ejemplo de curva ROC. Fuente: Elaboración propia.
  ],
)

#ef-proposition[
  *Interpretación del AUC:*
  - $"AUC" = 0.5$: Clasificación aleatoria
  - $"AUC" = 1.0$: Clasificación perfecta
  - $"AUC" > 0.8$: Generalmente considerado buen modelo
]

//#pagebreak()

= Comparación: Regresión Logística vs. MLP

== Análisis Comparativo

#figure(
  block(
    width: 100%,
    table(
      columns: 3,
      align: (left, center, center),
      stroke: (x, y) => (
        top: if y == 0 { 2pt + ef-primary } else { 0.5pt + ef-neutral-light },
        bottom: 0.5pt + ef-neutral-light,
      ),
      fill: (x, y) => if y == 0 { ef-primary } else if calc.odd(y) { ef-paper-bg } else { white },
      inset: 10pt,
      
      table.header(
        [#text(fill: white, weight: "bold")[Aspecto]], 
        [#text(fill: white, weight: "bold")[Reg. Logística]], 
        [#text(fill: white, weight: "bold")[MLP]]
      ),
      
      [Complejidad], [Baja], [Alta],
      [Parámetros], [$d + 1$], [Cientos-Miles],
      [Frontera], [Lineal], [No-lineal],
      [Interpretabilidad], [Alta], [Baja],
      [Velocidad train], [~ms], [~segundos],
      [Velocidad inference], [Muy rápida], [Rápida],
      [Requisitos datos], [Pocos], [Muchos],
      [Overfitting], [Menos propenso], [Más propenso],
    )
  ),
  caption: [Comparación entre Regresión Logística y MLP],
  kind: table
) <tab:comparacion>

== El Problema de Círculos Concéntricos

=== Definición del Problema

Sean $bold(x) in RR^2$ puntos en el plano. El dataset de círculos concéntricos se define como:

$ y = cases(1 quad & "si" ||bold(x)|| < r_1, 0 quad & "si" r_1 <= ||bold(x)|| <= r_2) $ <eq:concentric-circles>

donde $r_1 < r_2$ son los radios interior y exterior.

=== Por Qué Regresión Logística Falla

#ef-teorema("Imposibilidad de Separación Lineal")[
  No existe $bold(w) in RR^2$ y $b in RR$ tal que la frontera lineal $bold(w)^T bold(x) + b = 0$ separe correctamente el dataset de círculos concéntricos.
]

#ef-demostracion[
  Supongamos que existe tal hiperplano. Entonces puntos en el círculo interior satisfacen $bold(w)^T bold(x) + b > 0$. Pero el círculo interior está completamente rodeado por el exterior, por lo que cualquier línea que pase por el origen cruzará ambas clases. Por simetría radial, no existe orientación de línea que separe las clases.
]

=== Resultados Experimentales

En nuestro experimento con círculos concéntricos:

- *Dataset:* 1000 puntos, noise = 0.05, factor = 0.5
- *Arquitectura MLP:* 2-10-10-1 con activación tanh
- *Resultados:*
  - Regresión Logística: 47.5% accuracy
  - MLP: 100.0% accuracy
  - Diferencia: 52.5 puntos porcentuales

#ef-nota[
  La accuracy de 47.5% para Regresión Logística está cerca del azar (50%), confirmando que el modelo no puede aprender el patrón. El MLP alcanza clasificación perfecta, demostrando su capacidad para fronteras no-lineales.
]

#figure(
  image("metricas.png"),
  caption: [
    Ejemplo de curva métricas de modelo. Fuente: Elaboración propia.
  ],
)

//#pagebreak()

= Arquitecturas de Redes Neuronales

== Profundidad vs. Amplitud

*Redes Profundas (Deep Networks):*
- Muchas capas con pocas neuronas cada una
- Ejemplo: 2-8-8-8-8-1
- Aprenden jerarquías de características

*Redes Anchas (Wide Networks):*
- Pocas capas con muchas neuronas
- Ejemplo: 2-50-1
- Más fáciles de entrenar

== Regularización

Para prevenir overfitting:

*L2 Regularization (Weight Decay):*

$ cal(L)_("total") = cal(L)_("data") + lambda sum_(l=1)^L ||W^((l))||_F^2 $ <eq:l2-reg>

*Dropout:*

Durante entrenamiento, desactivar aleatoriamente neuronas con probabilidad $p$.

// ============================================================================
// SECCIÓN: Experimento Demostrativo
// Agregar este contenido después de la sección "Arquitecturas de Redes Neuronales"
// y antes de "Conclusiones"
// ============================================================================

= Experimento Demostrativo

== Diseño Experimental

Para validar empíricamente los fundamentos teóricos presentados, se diseñó un experimento comparativo entre Regresión Logística y Perceptrón Multicapa (MLP) utilizando el problema canónico de círculos concéntricos. Este dataset sintético fue seleccionado específicamente porque representa un caso paradigmático de datos *no-linealmente separables*, donde la geometría intrínseca del problema —una clase contenida radialmente dentro de otra— imposibilita cualquier separación mediante hiperplanos.

El experimento se implementó en Python utilizando el ecosistema científico estándar: NumPy (v2.0.2) para operaciones numéricas, Pandas (v2.2.2) para manipulación de datos, Matplotlib para visualización de calidad académica, y Scikit-learn (v1.5+) para la implementación de los modelos. La reproducibilidad se garantizó mediante la fijación de semilla aleatoria (`random_state=42`) en todas las operaciones estocásticas.

=== Metodología de Validación Estadística

Para garantizar la robustez y generalización de los resultados más allá de una única partición de datos, se implementó un protocolo de validación estadística riguroso siguiendo las recomendaciones metodológicas de Demšar (2006) para comparación de algoritmos de clasificación.

==== Evaluación Monte Carlo

Se ejecutaron $n = 30$ experimentos independientes, cada uno con una semilla aleatoria diferente que afecta:

- La generación del dataset (ruido gaussiano)
- La partición train/test
- La inicialización de pesos del MLP

Este número de repeticiones ($n >= 30$) garantiza la aplicabilidad del Teorema Central del Límite para la estimación de intervalos de confianza.

==== Validación Cruzada Repetida

Complementariamente, se aplicó _k-fold cross-validation_ estratificada con $k = 10$ folds y 10 repeticiones, resultando en 100 evaluaciones por modelo. Este esquema asegura que cada observación sea utilizada exactamente 10 veces para prueba, proporcionando estimaciones más estables que la evaluación Monte Carlo simple.

==== Intervalos de Confianza

Los intervalos de confianza al 95% se calcularon mediante el método _bootstrap percentile_, que no asume normalidad en la distribución de las métricas:

$ "IC"_(95%) = [P_(2.5), P_(97.5)] $

donde $P_alpha$ denota el percentil $alpha$ de la distribución empírica de scores.

==== Pruebas de Significancia Estadística

Para determinar si las diferencias observadas son estadísticamente significativas, se emplearon dos pruebas complementarias:

1. *Test de McNemar:* Compara las predicciones de ambos clasificadores sobre el mismo conjunto de prueba, evaluando si la proporción de desacuerdos es significativamente diferente de lo esperado por azar. El estadístico con corrección de continuidad es:

$ chi^2 = frac((|b - c| - 1)^2, b + c) $

donde $b$ representa casos donde solo el modelo 1 acierta y $c$ donde solo el modelo 2 acierta.

2. *Test de Wilcoxon de rangos con signo:* Prueba no paramétrica para muestras pareadas que compara los scores de validación cruzada de ambos modelos sobre los mismos folds, sin asumir normalidad en las diferencias.

==== Tamaño del Efecto

Más allá de la significancia estadística, se reporta el tamaño del efecto mediante la _d_ de Cohen:

$ d = frac(overline(x)_2 - overline(x)_1, s_"pooled") $

donde $s_"pooled" = sqrt((s_1^2 + s_2^2) / 2)$. La interpretación estándar establece: $|d| < 0.2$ (pequeño), $0.2 <= |d| < 0.5$ (mediano), $0.5 <= |d| < 0.8$ (grande), $|d| >= 0.8$ (muy grande).


== Generación del Dataset

El dataset de círculos concéntricos se generó mediante la función `make_circles` de Scikit-learn con los siguientes parámetros:

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Parámetro*], [*Valor*], [*Descripción*],
    [`n_samples`], [1000], [Total de observaciones],
    [`noise`], [0.05], [Desviación estándar del ruido gaussiano (5%)],
    [`factor`], [0.5], [Ratio entre radio interior y exterior],
    [`random_state`], [42], [Semilla para reproducibilidad],
  ),
  caption: [Parámetros de generación del dataset de círculos concéntricos.]
) <tab:dataset-params>

El parámetro `factor=0.5` establece que el círculo interior tiene la mitad del radio del exterior, creando una separación visual clara entre clases. El ruido del 5% introduce variabilidad realista sin comprometer la estructura geométrica fundamental del problema.

El dataset resultante presenta distribución perfectamente balanceada: 500 muestras por clase (50%/50%), lo que elimina sesgos por desbalance de clases en la evaluación.

== Partición de Datos

Se aplicó _stratified split_ (partición estratificada) mediante `train_test_split` con los siguientes parámetros:

- *Proporción:* 80% entrenamiento (800 muestras) / 20% prueba (200 muestras)
- *Estratificación:* Preservación de proporciones de clase en ambas particiones
- *Semilla:* `random_state=42`

La estratificación garantiza que tanto el conjunto de entrenamiento como el de prueba mantengan la distribución original 50%/50% entre clases.

// Figura del dataset ya está insertada anteriormente como fig:dataset

== Especificación de Modelos

=== Regresión Logística (Modelo Baseline)

El modelo lineal se configuró con parámetros estándar para garantizar convergencia:

```python
LogisticRegression(
    solver='lbfgs',      # Optimizador quasi-Newton de memoria limitada
    max_iter=1000,       # Iteraciones máximas
    random_state=42      # Reproducibilidad
)
```

El optimizador L-BFGS (_Limited-memory Broyden-Fletcher-Goldfarb-Shanno_) es apropiado para datasets de tamaño moderado y garantiza convergencia al mínimo global debido a la convexidad de la función de pérdida de entropía cruzada.

=== Perceptrón Multicapa (MLP)

La arquitectura principal del MLP se definió como *2-10-10-1*, correspondiente a:

- *Capa de entrada:* 2 neuronas (dimensionalidad del espacio de características)
- *Primera capa oculta:* 10 neuronas con activación tanh
- *Segunda capa oculta:* 10 neuronas con activación tanh
- *Capa de salida:* 1 neurona con activación sigmoide (clasificación binaria)

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Hiperparámetro*], [*Valor*], [*Justificación*],
    [`hidden_layer_sizes`], [(10, 10)], [Arquitectura moderada suficiente para el problema],
    [`activation`], ['tanh'], [Función centrada en cero, gradientes estables],
    [`solver`], ['adam'], [Optimizador adaptativo con momentum],
    [`max_iter`], [2000], [Épocas suficientes para convergencia],
    [`learning_rate_init`], [0.01], [Tasa de aprendizaje inicial],
    [`random_state`], [42], [Reproducibilidad en inicialización de pesos],
  ),
  caption: [Configuración de hiperparámetros del MLP.]
) <tab:mlp-config>

El *número total de parámetros* entrenables para esta arquitectura es:

$ N_("params") = (2 times 10 + 10) + (10 times 10 + 10) + (10 times 1 + 1) = 30 + 110 + 11 = 151 $

La elección de la función de activación *tanh* (tangente hiperbólica) sobre ReLU se fundamenta en sus propiedades para este problema específico: al estar centrada en cero y ser antisimétrica, facilita el aprendizaje de patrones radialmente simétricos como los círculos concéntricos.

== Protocolo de Evaluación

La evaluación se realizó sobre el conjunto de prueba (200 muestras) utilizando las siguientes métricas:

+ *Accuracy (Exactitud):* Proporción de predicciones correctas
  $ "Accuracy" = (T P + T N)/(T P + T N + F P + F N) $

+ *Precision (Precisión):* Proporción de verdaderos positivos entre predicciones positivas
  $ "Precision" = (T P)/(T P + F P) $

+ *Recall (Sensibilidad):* Proporción de verdaderos positivos detectados
  $ "Recall" = (T P)/(T P + F N) $

+ *F1-Score:* Media armónica de precision y recall
  $ F_1 = 2 times ("Precision" times "Recall")/("Precision" + "Recall") $

+ *AUC-ROC:* Área bajo la curva ROC (_Receiver Operating Characteristic_)

Adicionalmente, se generaron *matrices de confusión* para visualizar la distribución de errores y *curvas ROC* para evaluar el rendimiento a diferentes umbrales de decisión.

== Resultados Experimentales

=== Métricas de Clasificación

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else if row == 2 { rgb("#E8F5E9") } else { none },
    [*Modelo*], [*Accuracy*], [*Precision*], [*Recall*], [*F1-Score*],
    [Regresión Logística], [0.4750], [0.4749], [0.4750], [0.4747],
    [MLP (10,10)], [*1.0000*], [*1.0000*], [*1.0000*], [*1.0000*],
  ),
  caption: [Comparación de métricas de clasificación entre Regresión Logística y MLP en el dataset de círculos concéntricos.]
) <tab:metrics-comparison>

La Regresión Logística alcanza un accuracy de *47.5%*, ligeramente inferior al azar teórico (50%), confirmando su incapacidad para aprender la estructura del problema. Este resultado es consistente con el Teorema de Limitación de Modelos Lineales presentado en el marco teórico.

El MLP logra *100% de accuracy* en todas las métricas, demostrando clasificación perfecta en el conjunto de prueba. Este resultado valida empíricamente el Teorema de Aproximación Universal: las funciones de activación no-lineales permiten al MLP aproximar la frontera de decisión circular requerida.

La *diferencia de 52.5 puntos porcentuales* entre ambos modelos constituye evidencia contundente de la necesidad de no-linealidad para este tipo de problemas.

// Figura de fronteras de decisión
#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 1em,
)[
  #figure(
    image("images/fig02_decision_boundaries.pdf", width: 100%),
    caption: [Fronteras de decisión aprendidas. Panel (a): Regresión Logística muestra una frontera lineal incapaz de separar las clases. Panel (b): MLP aprende una frontera circular que separa perfectamente ambas clases.]
  ) <fig:decision-boundaries>
  #v(12pt)
]

=== Matrices de Confusión

El análisis de las matrices de confusión revela patrones de error característicos:

*Regresión Logística:*
- Errores distribuidos aproximadamente uniformemente entre ambos tipos ($F P approx F N$)
- No existe sesgo hacia ninguna clase específica
- El modelo "adivina" efectivamente al azar

*MLP:*
- Matriz diagonal perfecta (sin errores de clasificación)
- $T P = T N = 100$ (dado balance perfecto en test)
- $F P = F N = 0$

#figure(
  image("images/fig03_confusion_matrices.pdf", width: 85%),
  caption: [Matrices de confusión comparativas. Panel (a): Regresión Logística muestra distribución cercana a uniforme indicando predicción aleatoria. Panel (b): MLP presenta matriz diagonal perfecta sin errores de clasificación.]
) <fig:confusion-matrices>

=== Curvas ROC

Las curvas ROC proporcionan una perspectiva adicional sobre el rendimiento discriminativo:

- *Regresión Logística:* $"AUC" approx 0.50$ (equivalente a clasificación aleatoria)
- *MLP:* $"AUC" = 1.00$ (clasificación perfecta)

#figure(
  image("images/fig04_roc_curves.pdf", width: 70%),
  caption: [Curvas ROC para ambos modelos. La Regresión Logística (línea azul) sigue la diagonal de no-discriminación con AUC ≈ 0.50, mientras el MLP (línea naranja) alcanza la esquina superior izquierda con AUC = 1.00.]
) <fig:roc-curves>

=== Análisis de Tiempos de Entrenamiento

Se realizó un benchmark de tiempos de entrenamiento con 20 iteraciones para obtener estimaciones robustas:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Modelo*], [*Tiempo Medio (ms)*], [*Desv. Estándar (ms)*],
    [Regresión Logística], [10.33], [4.49],
    [MLP (10,10)], [~300], [~80],
  ),
  caption: [Comparación de tiempos de entrenamiento (promedio de 20 iteraciones).]
) <tab:timing>

La Regresión Logística es aproximadamente *20-30 veces más rápida* en entrenamiento. Sin embargo, este trade-off entre velocidad y capacidad expresiva es irrelevante cuando el modelo más rápido no puede resolver el problema.

#figure(
  image("images/fig05_training_time.pdf", width: 100%),
  caption: [Comparación de tiempos de entrenamiento con barras de error (±1 desviación estándar).]
) <fig:training-time>


=== Validación Estadística de Resultados

Los resultados presentados en las secciones anteriores corresponden a una única ejecución determinística con semilla fija (`random_state=42`). A continuación se presentan los resultados de la validación estadística que confirman la robustez de estos hallazgos.

==== Resultados de Evaluación Monte Carlo

La @tab:monte-carlo presenta las métricas de clasificación promediadas sobre 30 ejecuciones independientes con diferentes semillas aleatorias.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Modelo*], [*Accuracy*], [*Precision*], [*Recall*], [*F1*], [*AUC-ROC*],
    [Regresión Logística], 
    [$0.4713 plus.minus 0.0177$], 
    [$0.4712 plus.minus 0.0178$], 
    [$0.4713 plus.minus 0.0177$], 
    [$0.4708 plus.minus 0.0179$], 
    [$0.4604 plus.minus 0.0210$],
    [MLP (10,10)], 
    [$1.0000 plus.minus 0.0000$], 
    [$1.0000 plus.minus 0.0000$], 
    [$1.0000 plus.minus 0.0000$], 
    [$1.0000 plus.minus 0.0000$], 
    [$1.0000 plus.minus 0.0000$],
  ),
  caption: [Métricas de clasificación mediante evaluación Monte Carlo (media $plus.minus$ desviación estándar, $n = 30$ ejecuciones).]
) <tab:monte-carlo>

Los intervalos de confianza al 95% para accuracy son:

- *Regresión Logística:* $[0.4386, 0.5041]$
- *MLP:* $[1.0000, 1.0000]$

#ef-nota()[
  El intervalo de confianza de la Regresión Logística *incluye el valor 0.50*, confirmando estadísticamente que su rendimiento no es distinguible del azar. El MLP presenta *varianza cero* en las 30 ejecuciones, indicando que el problema es consistentemente resoluble para arquitecturas no-lineales.
]

==== Resultados de Validación Cruzada Repetida

La validación cruzada estratificada (10-fold $times$ 10 repeticiones) proporciona una perspectiva complementaria con 100 evaluaciones por modelo:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Modelo*], [*Accuracy*], [*IC 95%*],
    [Regresión Logística], [$0.4504 plus.minus 0.0320$], [$[0.3900, 0.5000]$],
    [MLP (10,10)], [$1.0000 plus.minus 0.0000$], [$[1.0000, 1.0000]$],
  ),
  caption: [Accuracy mediante validación cruzada repetida (10-fold $times$ 10 repeticiones).]
) <tab:cv-repeated>

==== Pruebas de Significancia Estadística

La @tab:statistical-tests resume los resultados de las pruebas de hipótesis para la comparación entre modelos.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Prueba*], [*Estadístico*], [*p-valor*], [*Significancia*],
    [Test de McNemar], [$chi^2 = 103.01$], [$< 0.001$], [✱✱✱],
    [Test de Wilcoxon], [$W = 0.00$], [$3.51 times 10^(-18)$], [✱✱✱],
    [Cohen's _d_], [$d = 24.26$], [—], [Efecto extremo],
  ),
  caption: [Resultados de pruebas de significancia estadística. Códigos: ✱✱✱ $p < 0.001$.]
) <tab:statistical-tests>

*Interpretación de resultados:*

1. *Test de McNemar ($chi^2 = 103.01$, $p < 0.001$):* Rechaza la hipótesis nula de que ambos clasificadores tienen el mismo patrón de errores. La diferencia en predicciones es altamente significativa.

2. *Test de Wilcoxon ($W = 0.00$, $p = 3.51 times 10^(-18)$):* El valor $W = 0$ indica que en *todas* las comparaciones pareadas de folds, el MLP superó a la Regresión Logística. El p-valor extremadamente bajo ($approx 10^(-18)$) proporciona evidencia contundente contra la hipótesis nula.

3. *Cohen's $d = 24.26$:* Este valor representa un tamaño del efecto *extremadamente grande*. Para contextualizar:
   - Valores típicos de $|d| > 0.8$ se consideran "efecto grande"
   - Un $d = 24.26$ indica que las distribuciones de accuracy están separadas por más de 24 desviaciones estándar combinadas
   - Prácticamente *no existe solapamiento* entre las distribuciones de rendimiento de ambos modelos

==== Análisis de la Diferencia de Rendimiento

La diferencia media de accuracy entre MLP y Regresión Logística es:

$ Delta_"accuracy" = 0.5496 approx 55 "puntos porcentuales" $

El histograma de diferencias (@fig:statistical-validation, panel d) muestra que *todas* las 30 ejecuciones Monte Carlo resultaron en diferencias positivas a favor del MLP, con un rango de $[0.43, 0.60]$.

==== Análisis de Tiempos de Entrenamiento

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else { none },
    [*Modelo*], [*Tiempo Medio*], [*Desv. Estándar*], [*Ratio*],
    [Regresión Logística], [4.5 ms], [$plus.minus$ 1.2 ms], [1.0×],
    [MLP (10,10)], [302.7 ms], [$plus.minus$ 45.3 ms], [67.3×],
  ),
  caption: [Tiempos de entrenamiento (promedio de 30 ejecuciones).]
) <tab:training-times-stat>

El MLP requiere aproximadamente *67 veces más tiempo* de entrenamiento que la Regresión Logística. Sin embargo, este incremento en costo computacional es irrelevante en el contexto de este problema: un modelo que entrena en 4.5 ms pero no puede resolver el problema tiene utilidad nula, mientras que 302.7 ms para clasificación perfecta representa un _trade-off_ aceptable.

// Figura de validación estadística
#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 1em,
)[
#figure(
  image("images/fig_statistical_validation.pdf", width: 100%),
  caption: [
    Validación estadística del experimento. 
    *(a)* Distribución de accuracy (violin plot) para evaluación Monte Carlo. 
    *(b)* Box plots con puntos individuales de validación cruzada. 
    *(c)* Forest plot con intervalos de confianza al 95%. 
    *(d)* Histograma de diferencias de accuracy (MLP − LR). 
    *(e)* Comparación de tiempos de entrenamiento (escala logarítmica). 
    *(f)* Resumen de pruebas de significancia estadística.
  ]
) <fig:statistical-validation>
]


== Exploración de Arquitecturas

Para evaluar la sensibilidad del rendimiento a la arquitectura, se compararon seis configuraciones de MLP con diferentes profundidades y amplitudes:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    fill: (_, row) => if row == 0 { ef-primary.lighten(85%) } else if row == 2 { rgb("#E8F5E9") } else { none },
    [*Arquitectura*], [*Parámetros*], [*Accuracy*], [*Tiempo (s)*],
    [(5,)], [~21], [~99%], [~0.3],
    [(10,10)], [151], [*100%*], [~0.5],
    [(20,20)], [~861], [100%], [~0.8],
    [(10,10,10)], [~271], [100%], [~0.7],
    [(50,)], [~201], [~99%], [~0.4],
    [(20,10,5)], [~326], [100%], [~0.6],
  ),
  caption: [Comparación de diferentes arquitecturas MLP en el problema de círculos concéntricos.]
) <tab:architectures>

#ef-nota[
  *Observaciones clave:*
  
  + *Arquitecturas mínimas suficientes:* Incluso una sola capa oculta con 5 neuronas $(5,)$ alcanza accuracy cercano al 100%, validando que el problema no requiere arquitecturas complejas.
  
  + *Rendimientos decrecientes:* Aumentar la complejidad más allá de $(10,10)$ no mejora el rendimiento en este problema específico.
  
  + *Principio de parsimonia:* La arquitectura $(10,10)$ representa un balance óptimo entre capacidad expresiva y complejidad para este problema.
]

// Figura de arquitecturas exploradas
#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 1em,
)[
  #figure(
    image("images/fig06_architectures.pdf", width: 100%),
    caption: [Fronteras de decisión para diferentes arquitecturas MLP. Todas las configuraciones logran separar las clases, demostrando que arquitecturas simples son suficientes para este problema.]
  ) <fig:architectures>
  #v(12pt)
]

#pagebreak()
== Curvas de Aprendizaje

Las _learning curves_ (curvas de aprendizaje) mediante validación cruzada 5-fold revelan el comportamiento de generalización:

*Regresión Logística:*
- Training score $approx$ 50% constante para todos los tamaños de entrenamiento
- Cross-validation score $approx$ 50% constante
- Sin brecha entre training y validation $arrow.r$ *underfitting* (subajuste)
- El modelo no aprende independientemente de la cantidad de datos

*MLP (10,10):*
- Training score $arrow.r$ 100% rápidamente
- Cross-validation score $arrow.r$ ~99-100% con suficientes datos
- Brecha mínima $arrow.r$ *buen ajuste* sin overfitting significativo
- El modelo generaliza correctamente

#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 1em,
)[
#figure(
  image("images/fig07_learning_curves.pdf", width: 80%),
  caption: [Curvas de aprendizaje con validación cruzada 5-fold. Panel (a): Regresión Logística muestra underfitting con scores constantes ~50%. Panel (b): MLP converge rápidamente a accuracy ~100% con brecha mínima entre training y validation.]
) <fig:learning-curves>
]


== Síntesis de Resultados

Los resultados experimentales confirman de manera contundente las predicciones teóricas:

+ *Validación del Teorema de Limitación Lineal:* La Regresión Logística no puede superar el rendimiento aleatorio en el problema de círculos concéntricos, exactamente como predice la teoría.

+ *Validación del Teorema de Aproximación Universal:* El MLP con funciones de activación no-lineales puede aproximar la frontera de decisión circular requerida.

+ *Principio de parsimonia aplicado:* Arquitecturas relativamente simples $(10,10)$ son suficientes; la complejidad adicional no aporta beneficio en este problema.

+ *Trade-off complejidad-capacidad:* El costo computacional adicional del MLP se justifica plenamente cuando el modelo lineal es fundamentalmente incapaz de resolver el problema.

#ef-nota[
  Estos hallazgos demuestran la importancia de seleccionar la clase de modelo apropiada según la geometría inherente de los datos, priorizando la *adecuación al problema* sobre la simplicidad arbitraria.
]


#place(
  bottom + center,
  scope: "parent",
  float: true,
  clearance: 1em,
)[
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + luma(180),
    inset: 8pt,
    [*Modelo*], [*Accuracy*], [*Precision*], [*Recall*], [*F1*], [*AUC*],
    [Logistic Regression], [$0.4713 plus.minus 0.0177$], [$0.4712 plus.minus 0.0178$], [$0.4713 plus.minus 0.0177$], [$0.4708 plus.minus 0.0179$], [$0.4604 plus.minus 0.0210$],
    [MLP (10,10)], [$1.0000 plus.minus 0.0000$], [$1.0000 plus.minus 0.0000$], [$1.0000 plus.minus 0.0000$], [$1.0000 plus.minus 0.0000$], [$1.0000 plus.minus 0.0000$],
  ),
  caption: [Métricas de clasificación (media ± std, n=30 ejecuciones, IC 95%) ]
)
<fig:metrics>
]

#pagebreak()

== Limitaciones del Estudio

Si bien los resultados presentados proporcionan evidencia contundente sobre las diferencias fundamentales entre modelos lineales y no-lineales, es importante reconocer las limitaciones inherentes al diseño experimental:

=== Naturaleza del Dataset

*Dataset sintético:* El problema de círculos concéntricos es un caso idealizado diseñado específicamente para ilustrar limitaciones de modelos lineales. Los datasets reales raramente presentan geometrías tan claramente definidas, y la frontera de decisión óptima suele ser desconocida.

*Baja dimensionalidad:* El espacio de características bidimensional ($d = 2$) permite visualización directa pero no captura los desafíos de datos de alta dimensión, donde fenómenos como la _maldición de la dimensionalidad_ y la escasez de datos (_data sparsity_) afectan significativamente el rendimiento de los modelos.

*Ruido controlado:* El nivel de ruido gaussiano del 5% (`noise=0.05`) es relativamente bajo. Escenarios con mayor ruido o ruido heteroscedástico podrían alterar las conclusiones sobre la separabilidad perfecta del MLP.

=== Balance de Clases

El dataset presenta distribución perfectamente balanceada (50%/50%). En aplicaciones reales, el desbalance de clases es frecuente y puede afectar diferencialmente a distintos tipos de modelos, requiriendo técnicas de _oversampling_, _undersampling_, o ajuste de pesos de clase.

=== Ausencia de Ruido en Etiquetas

Las etiquetas del dataset son determinísticas y correctas. En escenarios reales, el _label noise_ (etiquetas incorrectas) puede degradar el rendimiento de modelos con alta capacidad expresiva como las redes neuronales, que son más susceptibles a memorizar ruido.

=== Tamaño de Muestra

Con $n = 1000$ observaciones, el dataset representa un caso de tamaño moderado. El comportamiento de ambos modelos podría diferir significativamente en escenarios de:
- *Datos escasos* ($n < 100$): Mayor riesgo de _overfitting_ para MLP
- *Big Data* ($n > 10^6$): Consideraciones de escalabilidad computacional

=== Alcance de la Comparación

El estudio compara únicamente Regresión Logística y MLP. Otros enfoques para datos no-linealmente separables incluyen:

- *SVM con kernel RBF:* Alternativa clásica que podría lograr resultados similares al MLP
- *Random Forest / Gradient Boosting:* Ensambles basados en árboles con capacidad no-lineal inherente
- *Feature engineering:* Transformación manual del espacio (e.g., $r = sqrt(x_1^2 + x_2^2)$) que permitiría a la Regresión Logística resolver el problema

=== Generalización de Hallazgos

Los resultados demuestran el *principio general* de que modelos lineales no pueden resolver problemas no-linealmente separables, pero las arquitecturas y hiperparámetros óptimos varían según el dominio de aplicación. La arquitectura (10, 10) es adecuada para este problema específico; problemas más complejos podrían requerir configuraciones sustancialmente diferentes.

#ef-nota()[
  Estas limitaciones no invalidan las conclusiones del estudio, sino que contextualizan su alcance. El valor pedagógico del experimento radica precisamente en su simplicidad controlada, que permite aislar y demostrar el concepto fundamental de *capacidad expresiva* de diferentes familias de modelos.
]


#pagebreak()

= Conclusiones

== Resumen de Conceptos Clave

Este documento ha presentado un marco teórico completo para comprender las diferencias fundamentales entre modelos lineales y redes neuronales en clasificación binaria:

1. La *Regresión Logística* es un modelo lineal eficiente pero limitado a problemas linealmente separables.

2. El *MLP* introduce no-linealidad mediante funciones de activación, permitiendo aproximar funciones arbitrariamente complejas.

3. El *Teorema de Aproximación Universal* garantiza que redes neuronales pueden representar cualquier función continua.

4. El algoritmo de *Backpropagation* permite entrenar redes eficientemente mediante cálculo de gradientes.

5. El problema de *círculos concéntricos* demuestra claramente las limitaciones de modelos lineales y la necesidad de no-linealidad.

== Implicaciones Pedagógicas

Este marco teórico está diseñado para:

- Proporcionar fundamentos matemáticos rigurosos
- Mantener claridad didáctica mediante ejemplos
- Contextualizar conceptos abstractos en problemas concretos
- Facilitar la transición de teoría a práctica

Los estudiantes que dominen estos conceptos estarán preparados para:
- Seleccionar modelos apropiados para problemas específicos
- Diseñar arquitecturas de redes neuronales
- Evaluar y comparar modelos sistemáticamente
- Comprender literatura avanzada de Deep Learning

== Mensaje Final

La elección entre Regresión Logística y MLP no se trata de cuál es "mejor" en términos absolutos, sino de cuál es *apropiado* para el problema específico. Como se demostró con círculos concéntricos:

#quote[
  _La herramienta correcta depende de la geometría de los datos. Un modelo simple puede ser suficiente para problemas linealmente separables, pero la complejidad inherente de algunos problemas requiere la capacidad expresiva de redes neuronales._
]

Este principio se extiende más allá de Machine Learning: en ingeniería y ciencia, la sofisticación de la solución debe coincidir con la complejidad del problema.

En términos formales, este criterio se alinea con el *principio de parsimonia* u *Occam’s razor* en Machine Learning: entre dos modelos capaces de explicar adecuadamente los datos, se prefiere aquel que es más simple. Para problemas linealmente separables, la Regresión Logística constituye una solución parsimoniosa: ofrece buena capacidad de generalización con menor complejidad computacional y conceptual. En cambio, para fronteras intrínsecamente no lineales —como el caso de los círculos concéntricos— la parsimonia no implica insistir en modelos lineales, sino elegir la clase de modelo más simple *dentro de la familia adecuada*, donde arquitecturas MLP poco profundas pueden capturar la geometría del problema sin recurrir a redes excesivamente profundas o sobreparametrizadas.


//#pagebreak()

= Referencias

1. *Goodfellow, I., Bengio, Y., & Courville, A.* (2016). _Deep Learning_. MIT Press.

2. *Bishop, C. M.* (2006). _Pattern Recognition and Machine Learning_. Springer.

3. *Géron, A.* (2019). _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.

4. *Cybenko, G.* (1989). Approximation by Superpositions of a Sigmoidal Function. _Mathematics of Control, Signals and Systems_, 2(4), 303-314.

5. *Hornik, K., Stinchcombe, M., & White, H.* (1989). Multilayer Feedforward Networks are Universal Approximators. _Neural Networks_, 2(5), 359-366.

6. *Rumelhart, D. E., Hinton, G. E., & Williams, R. J.* (1986). Learning Representations by Back-Propagating Errors. _Nature_, 323(6088), 533-536.

7. *LeCun, Y., Bengio, Y., & Hinton, G.* (2015). Deep Learning. _Nature_, 521(7553), 436-444.

8. *Kingma, D. P., & Ba, J.* (2014). Adam: A Method for Stochastic Optimization. _arXiv preprint arXiv:1412.6980_.

9. *Pedregosa, F., et al.* (2011). Scikit-learn: Machine Learning in Python. _Journal of Machine Learning Research_, 12, 2825-2830.

#pagebreak()

= Apéndice A: Preguntas de Autoevaluación

== Nivel Básico

1. ¿Qué es una frontera de decisión y por qué es importante en clasificación?

2. Explique por qué la Regresión Logística solo puede crear fronteras lineales.

3. ¿Cuál es el rol de las funciones de activación en redes neuronales?

4. Defina accuracy, precision y recall. ¿Cuándo cada métrica es más relevante?

5. ¿Qué significa que un dataset sea "linealmente separable"?

== Nivel Intermedio

6. Demuestre que un MLP sin funciones de activación no-lineales colapsa a un modelo lineal.

7. Compare redes profundas vs. redes anchas. ¿Cuáles son las ventajas de cada enfoque?

8. Interprete una learning curve que muestra gran brecha entre training y validation accuracy.

9. Calcule el número de parámetros para una arquitectura 5-20-15-10-3.

10. Explique el concepto de vanishing gradient y cómo afecta el entrenamiento.

== Nivel Avanzado

11. Diseñe una arquitectura de red neuronal para un problema con 100 features y 5 clases. Justifique sus decisiones de diseño.

12. Derive la regla de actualización de backpropagation para una capa con activación ReLU.

13. Analice el trade-off entre capacidad del modelo y riesgo de overfitting. ¿Cómo regularización mitiga este problema?

14. Proponga un experimento para determinar si un problema requiere un modelo no-lineal.

15. Discuta las implicaciones del Teorema de Aproximación Universal para el diseño de arquitecturas en la práctica.

#colbreak()

= Apéndice B: Glosario de Términos

*Backpropagation:* Algoritmo para calcular gradientes en redes neuronales mediante regla de la cadena.

*Batch:* Subconjunto de datos usado en cada iteración de entrenamiento.

*Bias (Sesgo):* Término independiente en transformaciones lineales.

*Decision Boundary:* Frontera que separa regiones de diferentes clases en el espacio de features.

*Epoch (Época):* Una pasada completa por el dataset de entrenamiento.

*Feature:* Variable de entrada o atributo de las observaciones.

*Forward Propagation:* Cálculo de predicción desde input hasta output.

*Gradient Descent:* Algoritmo de optimización que sigue la dirección opuesta al gradiente.

*Hidden Layer:* Capa intermedia en red neuronal entre input y output.

*Learning Rate:* Parámetro que controla el tamaño del paso en optimización.

*Loss Function:* Función que cuantifica el error del modelo.

*Overfitting:* Modelo que memoriza training data pero falla en generalizar.

*Regularization:* Técnicas para prevenir overfitting (L1, L2, dropout).

*Underfitting:* Modelo demasiado simple que no captura el patrón de los datos.

*Weight (Peso):* Parámetro multiplicativo en transformaciones lineales.
