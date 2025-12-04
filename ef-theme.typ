// ==========================================================
// EF THEME — Versión Mejorada v1.3
// Diseño editorial profesional con identidad EF
// Mejoras de espaciado, tipografía y contraste
// ==========================================================

// -----------------------------
// 1. PALETA EF (Mejorada con mejor contraste)
// -----------------------------
#let ef-primary  = rgb("#0E254E")   // Azul profundo
#let ef-electric = rgb("#3BA4F2")   // Azul eléctrico
#let ef-accent   = rgb("#7DDCF6")   // Cian suave
#let ef-accent2  = rgb("#5FB3E6")   // Azul soft

// Fondos con mejor contraste
#let ef-bg1      = rgb("#E0EAF5")   // Fondo definiciones (más saturado)
#let ef-bg2      = rgb("#D5E4F3")   // Fondo teoremas (más saturado)
#let ef-bg3      = rgb("#C5DEFF")   // Fondo ejemplos (más saturado)
#let ef-paper-bg = rgb("#FAFBFD")   // Blanco azulado
#let ef-abstract-bg = rgb("#F5F9FC") // Fondo abstract

#let ef-neutral-dark   = rgb("#2A2E33")
#let ef-neutral-medium = rgb("#6A6F77")
#let ef-neutral-light  = rgb("#D5D8DD")

// -----------------------------
// 2. CONTADORES PARA NUMERACIÓN
// -----------------------------
#let def-counter = counter("definicion")
#let thm-counter = counter("teorema")
#let ej-counter = counter("ejemplo")

// -----------------------------
// 3. CUADROS EF MEJORADOS
// -----------------------------

// Definición (con numeración opcional)
#let ef-definicion(numbered: false, body) = {
  if numbered {
    def-counter.step()
  }
  v(10pt)
  block(
    fill: ef-bg1,
    stroke: (paint: ef-electric, thickness: 1pt),
    inset: 14pt,
    radius: 8pt,
    width: 100%,
    breakable: false,
  )[
    #text(font: "Montserrat", weight: "bold", fill: ef-primary)[
      #box(baseline: 20%, image("icons/book.svg", width: 12pt)) Definición#if numbered [ #context def-counter.display()]: 
    ] 
    #text(font: "Libertinus Serif")[#body]
  ]
  v(10pt)
}

// Algoritmo
#let ef-algorithm(title, body) = block(
  fill: ef-accent,
  stroke: (paint: ef-electric, thickness: 1pt),
  inset: 10pt,
  radius: 6pt,
  width: 100%,
)[
  #text(font: "Montserrat", weight: "bold", fill: ef-primary)[
    #box(baseline: 20%, image("icons/notebook.svg", width: 12pt))Algoritmo: #title]
  #v(0.5em)
  #body
]

// Proposición: mismo estilo, acento en azul eléctrico
#let ef-proposition(body) = block(
  fill: ef-accent,
  stroke: (paint: ef-electric, thickness: 0.8pt),
  inset: 10pt,
  radius: 6pt,
  width: 100%,
)[
  #text(weight: "bold", fill: ef-primary)[Proposición: ] #body
]

// Teorema (con numeración opcional)
#let ef-teorema(title, numbered: false, body) = {
  if numbered {
    thm-counter.step()
  }
  v(10pt)
  block(
    fill: ef-bg2,
    stroke: (paint: ef-electric, thickness: 1.4pt),
    inset: 14pt,
    radius: 8pt,
    width: 100%,
    breakable: false,
  )[
    #text(font: "Montserrat", weight: "bold", fill: ef-primary)[
    #box(baseline: 20%, image("icons/delta.svg", width: 12pt)) Teorema#if numbered [ #context thm-counter.display()] (#title): 
  ] 
    #text(font: "Libertinus Serif")[#body]
  ]
  v(10pt)
}

// Corolario
#let ef-corolario(body) = {
  v(10pt)
  block(
    fill: ef-bg2,
    stroke: (paint: ef-accent, thickness: 1pt),
    inset: 14pt,
    radius: 8pt,
    width: 100%,
    breakable: false,
  )[
    #text(font: "Montserrat", weight: "bold", fill: ef-primary)[
  #box(baseline: 20%, image("icons/pin.svg", width: 12pt)) Corolario: ] 
    #text(font: "Libertinus Serif")[#body]
  ]
  v(10pt)
}

// Nota informativa
#let ef-nota(body) = {
  v(10pt)
  block(
    fill: ef-paper-bg,
    stroke: (paint: ef-accent, thickness: 1pt),
    inset: 12pt,
    radius: 8pt,
    width: 100%,
  )[
    #text(font: "Montserrat", weight: "bold", fill: ef-primary)[#box(baseline: 20%, image("icons/info.svg", width: 12pt)) Nota: ] 
    #text(font: "Libertinus Serif")[#body]
  ]
  v(10pt)
}

// Demostración
#let ef-demostracion(body) = block(
  inset: (left: 10pt),
)[
  #text(font: "Montserrat", weight: "bold", fill: ef-primary)[#box(baseline: 20%, image("icons/proof.svg", width: 12pt)) Demostración: ] 
    #text(font: "Libertinus Serif")[#body]
]

// Ejemplo (con numeración opcional)
#let ef-ejemplo(numbered: false, body) = {
  if numbered {
    ej-counter.step()
  }
  v(10pt)
  block(
    fill: ef-bg3,
    stroke: (paint: ef-electric, thickness: 0.8pt),
    inset: 12pt,
    radius: 8pt,
    width: 100%,
    breakable: false,
  )[
    #text(font: "Montserrat", weight: "bold", fill: ef-primary)[
    #box(baseline: 20%, image("icons/idea.svg", width: 12pt)) Ejemplo#if numbered [ #context ej-counter.display()]: 
  ] 
    #text(font: "Libertinus Serif")[#body]
  ]
  v(10pt)
}

// -----------------------------
// 4. ENCABEZADO (VENUE) Y TÍTULO
// -----------------------------

#let make-venue = move(dy: -1.9cm, {
  box(rect(
    fill: ef-primary,
    inset: 10pt,
    height: 2.5cm,
    width: 100%,
  )[
    #grid(
      columns: (auto, 1fr),
      gutter: 15pt,
      align: (horizon, horizon),
      // Logo
      image("logo-efarias_transp.png", width: 1.5cm),
      // Texto
      align(left + horizon)[
        #set text(font: "Montserrat", fill: white, weight: 700, size: 18pt)
        Machine Learning & Data Science
      ]
    )
  ])
})

// #let make-venue-with-subtitle(subtitle) = move(dy: -1.9cm, {
//   box(rect(
//     fill: ef-primary,
//     inset: 10pt,
//     height: 2.2cm,
//   )[
//     #set text(font: "Montserrat", fill: white, weight: 700, size: 18pt)
//     #align(bottom)[Machine Learning & Data Science]
//   ])
//   set text(font: "Montserrat", fill: ef-accent2, size: 12pt)
//   box(pad(left: 10pt, bottom: 10pt, subtitle))
// })

// Bloque de título mejorado
#let make-title(
  title,
  authors,
  abstract,
  keywords,
  subtitle: none,
) = {
  //set par(spacing: 6pt)
  v(-30pt)  // ← AGREGAR ESTA LÍNEA (ajusta el valor según necesites)
  set par(spacing: 1em)
  set text(font: "Montserrat")
  
  // Título
  par(
    justify: false,
    text(24pt, fill: ef-primary, weight: "bold", title)
  )

  // Subtítulo debajo del título
  if subtitle != none {
    v(6pt)
    par(
      justify: false,
      text(14pt, fill: ef-accent2, style: "italic", subtitle)
    )
  }
  
  v(6pt)
  // FOTO + AUTOR
  grid(
    columns: (auto, 1fr),
    //gutter: 0pt,
    align: (top, left),
    // Foto
    image("foto-ef.png", width: 3cm),
    // Información del autor
    block[
      // Nombre
      #text(12pt, font: "Montserrat", weight: "bold", fill: ef-primary)[
        \ Eduardo A. Farías Reyes
      ]
      #v(1pt)      
      // Afiliación
      #set text(9pt, font: "Montserrat", fill: ef-neutral-dark)
      Ingeniero en Informática — IP Santo Tomás \
      IA, Machine Learning & Ciencia de Datos \
      #link("mailto:contacto@efarias.cl") · https://efarias.cl
    ]
  )

  v(12pt)

  // // Autores
  // text(12pt,
  //   authors.enumerate()
  //   .map(((i, author)) => box[#author.name #super[#(i + 1)]])
  //   .join(", ")
  // )
  // parbreak()

  // // Afiliaciones
  // for (i, author) in authors.enumerate() [
  //   #set text(8pt, fill: ef-neutral-dark)
  //   #super[#(i + 1)]
  //   #author.institution
  //   — #link("mailto:" + author.mail) \
  // ]

  v(0pt)
  set text(10pt)
  set par(justify: true)

  // Abstract mejorado con fondo sutil
  block(
    fill: ef-abstract-bg,
    inset: 6pt,
    radius: 6pt,
    width: 100%,
  )[
    #heading(outlined: false, bookmarked: false, numbering: none)[Abstract]
    #text(font: "Libertinus Serif", fill: ef-neutral-dark, abstract)
    #v(6pt)
    #text(font: "Montserrat")[*Keywords:* #keywords.join("; ")]
  ]
  v(6pt)
}

// -----------------------------
// 5. TEMPLATE PRINCIPAL MEJORADO
// -----------------------------

#let template(
    title: [],
    authors: (),
    keywords: (),
    abstract: [],
    subtitle: none,
    make-venue: make-venue,
    make-title: make-title,
    body,
) = {
    // Configuración de página con mejores márgenes
    set page(
      paper: "a4",
      margin: (top: 2.5cm, bottom: 1.5cm, x: 1.8cm),
      columns: 2,
      fill: ef-paper-bg,
    )
    
    // Estilos base - Serif para cuerpo, Sans para UI
    set par(justify: true, spacing: 0.7em, leading: 0.65em)
    set text(10pt, font: "Libertinus Serif", fill: ef-neutral-dark)
    set list(indent: 10pt)
    
    // Enlaces en azul eléctrico
    show link: set text(fill: ef-electric, baseline: 0pt)
    
    // Headings con Montserrat
    show heading: set text(font: "Montserrat", fill: ef-primary)
    
    // Heading nivel 1 con línea decorativa
    show heading.where(level: 1): it => {
      set text(size: 13pt, weight: "bold")
      v(14pt)
      block(below: 12pt)[
        #it.body
        #v(4pt)
        #line(length: 100%, stroke: (paint: ef-electric, thickness: 2pt))
      ]
    }
    
    // Heading nivel 2
    show heading.where(level: 2): set text(size: 11pt, weight: "bold")
    show heading: set block(below: 8pt, above: 12pt)
    
    // FIGURAS (imágenes)
    show figure.where(kind: image): it => {
      block(below: 14pt, above: 14pt)[
        // Contenido de la imagen
        #it.body
        
        // Espacio entre imagen y caption
        #v(8pt)
        
        // Caption personalizado
        #block(inset: (x: 8%))[
          #set text(font: "Montserrat", size: 9pt, fill: ef-neutral-dark)
          #text(weight: "bold")[Figura #it.counter.display():] #it.caption.body
        ]
      ]
    }

    // TABLAS
    show figure.where(kind: table): it => {
      block(below: 14pt, above: 14pt)[
        // Contenido de la tabla
        #it.body
        
        // Espacio entre tabla y caption
        #v(8pt)
        
        // Caption personalizado
        #block(inset: (x: 8%))[
          #set text(font: "Montserrat", size: 9pt, fill: ef-neutral-dark)
          #text(weight: "bold")[Tabla #it.counter.display():] #it.caption.body
        ]
      ]
    }
    
    // Strong en Montserrat para destacar
    //show strong: set text(font: "Montserrat", weight: 350)
    
    // Footer mejorado con línea separadora
    set page(footer: context[
      #line(length: 100%, stroke: (paint: ef-neutral-light, thickness: 0.5pt))
      #v(6pt)
      #grid(
        columns: (1fr, auto, 1fr),
        align: (left, center, right),
        [#text(8pt, style: "italic", fill: ef-neutral-medium, font: "Montserrat")[Eduardo A. Farías Reyes]],
        [#text(8pt, fill: ef-neutral-medium, font: "Montserrat")[pág. #counter(page).display()]],
        [#text(8pt, style: "italic", fill: ef-neutral-medium, font: "Montserrat")[https://efarias.cl]],
      )
    ])

    // Venue (con o sin subtítulo)
    place(make-venue, top, scope: "parent", float: true)
    
    // Título
    place(
      make-title(title, authors, abstract, keywords, subtitle: subtitle), 
      top, 
      scope: "parent",
      float: true
    )

    body
}
