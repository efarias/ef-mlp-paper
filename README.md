# Clasificación Binaria: Modelos Lineales vs. Redes Neuronales

Este repositorio contiene el código fuente en **Typst** para el paper "Clasificación Binaria: Modelos Lineales vs. Redes Neuronales".

## Descripción

Este documento presenta un marco teórico completo que compara modelos lineales (específicamente Regresión Logística) con redes neuronales (Perceptrón Multicapa o MLP) en el contexto de clasificación binaria.

El trabajo se centra en:
-   Fundamentos matemáticos de la clasificación binaria.
-   Limitaciones de los modelos lineales (ej. problema de círculos concéntricos).
-   Arquitectura y funcionamiento del Perceptrón Multicapa (MLP).
-   Teorema de Aproximación Universal.
-   Algoritmo de Backpropagation.
-   Métricas de evaluación (Matriz de confusión, ROC, AUC, etc.).

## Estructura del Repositorio

-   `main.typ`: Archivo principal del documento.
-   `ef-theme.typ`: Definición del tema y estilos personalizados.
-   `impl.typ`: Implementaciones auxiliares.
-   `referencias.bib`: Archivo de bibliografía.
-   `*.png`: Imágenes y diagramas utilizados en el documento.

## Requisitos

Para compilar este documento, necesitas tener instalado [Typst](https://typst.app/).

## Compilación

Para generar el PDF, ejecuta el siguiente comando en tu terminal:

```bash
typst compile main.typ
```

## Autor

**Eduardo A. Farías Reyes**
-   Ingeniero en Informática
-   Docente y Data & ML Engineer
-   [efarias.cl](https://efarias.cl)
