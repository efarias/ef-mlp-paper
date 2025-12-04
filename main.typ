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

// OPCIÓN 1: Usando place() - MÁS CONTROL
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

//#pagebreak()

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
