# Slide Deck — Vista previa de la presentación

Este documento contiene el **contenido slide-por-slide** para una
presentación de vista previa (~15-20 min) de los avances de la tesis.
Cada slide tiene:

- **Título** (lo que va arriba en la slide).
- **Bullets** (texto principal — copiar y pegar tal cual).
- **Figura** (qué imagen incluir, ruta relativa al repo).
- **Notas para el orador** (lo que dices al hablar pero NO va en la slide).

Las figuras están en `data/derived/figures/preview/p01_*.png`
(generadas hoy con `scripts/_plots_preview.py` — re-ejecutables en
cualquier momento).

Lenguaje: español neutro, accesible a ingenieros sin formación en
finanzas o control. Sin pruebas formales, pero sí mostrando claramente
qué hace cada componente y qué medimos.

---

## SLIDE 1 — Portada

### Título
**Ejecución adaptativa en mercados financieros con filtro de Kalman y algoritmo Hedge**
*Una arquitectura de control para BTC/USDT*

### Bullets
- Damián Muñoz Díaz
- Vista previa de tesis — mayo 2026
- Repositorio: `adversity-is-controllable`

### Notas
"Voy a presentar una vista previa de mi tesis. Es un sistema de control
adaptativo que decide cómo ejecutar órdenes de compra en el mercado de
criptomonedas. Hoy quiero mostrarles tres cosas: la arquitectura, los
resultados que tenemos, y la metodología estadística que usamos."

---

## SLIDE 2 — Agenda

### Título
**Lo que veremos hoy**

### Bullets
1. El problema — comprar 1 BTC cada segundo
2. Arquitectura — Kalman + Hedge
3. El filtro de Kalman — cómo funciona, qué estima
4. El algoritmo Hedge — cómo aprende a decidir
5. Metodología — A/B pareado, 4 sesiones, 10 semillas
6. Resultados — el resultado positivo y el resultado negativo
7. La barrera de toxicidad — el hallazgo central
8. Próximos pasos

### Notas
"Quince minutos, ocho temas. La idea es que terminen entendiendo
qué hace cada pieza del sistema, qué probamos, y qué encontramos."

---

## SLIDE 3 — El problema

### Título
**El problema: ¿cómo comprar 1 BTC cada segundo?**

### Bullets
- Cada segundo, un agente está obligado a comprar 1 BTC en Binance.
- Tres opciones disponibles cada segundo:
  - 🟢 **AGRESIVA** — orden a mercado, fill garantizado, pago el spread.
  - 🔴 **PASIVA** — orden límite al best bid, sólo se ejecuta si el precio cae.
  - 🟡 **ESPERAR** — no hago nada, intento de nuevo en el siguiente tick.
- Cada decisión tiene un costo:
  - **Slippage** = lo que pago por encima del precio justo (mid).
  - **Adverse move** = cuánto se mueve el precio en mi contra después de comprar.
- Pregunta de tesis: **¿cómo decidir qué hacer cada segundo?**

### Notas
"Imaginen que su jefe les dice 'tienes que comprar 1 BTC cada segundo
durante 21 horas'. No pueden decir que no. Lo único que pueden elegir
es CÓMO comprar. La pregunta de la tesis es: ¿qué política toma esa
decisión y minimiza el costo? Eso es un problema de ejecución óptima,
no de trading. No estamos prediciendo precios, estamos optimizando
la ejecución de una compra ya decidida."

---

## SLIDE 4 — Arquitectura

### Título
**Arquitectura del sistema: Kalman + Hedge**

### Bullets
- **Kalman** estima dos números ocultos del mercado a partir de 3 features observables:
  - **Presión** — ¿están comprando o vendiendo más fuerte?
  - **Régimen** — ¿qué tan turbulento está el mercado?
- **Hedge** decide la acción usando esos dos números como contexto.
- Separación limpia: percepción (Kalman) → decisión (Hedge).

### Figura sugerida
Diagrama de bloques. Si no tienen uno gráfico, pueden hacer una caja
horizontal con cinco bloques:

```
[ Binance WS ] → [ Order book ] → [ Features ] → [ Kalman ] → [ Hedge ] → [ Acción ]
                                                                 ↑               │
                                                                 └── Loss ←──────┘
```

### Notas
"La arquitectura es deliberadamente clásica. Kalman es una herramienta
de control de los años 60, viene del programa Apolo. Hedge es de
aprendizaje en línea de los 90, viene de teoría de juegos. Ninguno de
los dos lo inventamos nosotros. Lo que es propio de la tesis es la
forma en la que los acoplamos sobre datos reales de criptomercado."

---

## SLIDE 5 — El filtro de Kalman: ¿qué hace?

### Título
**Filtro de Kalman: estimar lo invisible a partir de lo visible**

### Bullets
- Imaginen que están manejando con neblina:
  - Sus sentidos son ruidosos (la velocidad indica ±5 km/h, el GPS marca posiciones imprecisas).
  - Pero saben que el coche obedece leyes físicas (la posición no salta de golpe, la velocidad cambia gradualmente).
  - El filtro combina **lo que mide** y **lo que sabe** para estimar dónde está realmente.
- Aquí es lo mismo:
  - **Lo que medimos**: depth_imbalance, ofi_l1, vol_30s (3 features ruidosas del libro de órdenes).
  - **Lo que estimamos**: presión y régimen (2 estados ocultos).
- Es óptimo bajo supuestos lineales-gaussianos. Forma cerrada, sin entrenamiento.

### Notas
"El filtro de Kalman responde una pregunta: si tengo varias mediciones
ruidosas y un modelo de cómo cambia el sistema, ¿cuál es la mejor
estimación del estado real? La respuesta es matemática y elegante.
Lo importante es que NO predice el futuro: estima el presente."

---

## SLIDE 6 — Filtro de Kalman: las matrices

### Título
**Las cuatro matrices del filtro de Kalman**

### Bullets
- **F (transición)** — cómo cambia el estado oculto de un tick al siguiente.
  - F = diag(0.90, 0.95) → la presión decae 10% por tick, el régimen 5% (más persistente).
- **H (observación)** — cómo se relacionan los features con los estados ocultos.
  - depth_imbalance y ofi_l1 → presión.
  - vol_30s → régimen.
- **Q (ruido del proceso)** — qué tanto puede cambiar el estado por aleatoriedad.
- **R (ruido de la observación)** — qué tan ruidosas son las mediciones.

### Figura sugerida
Pueden mostrar las matrices directamente:

```
F = [0.90  0  ]    H = [1  0]    Q = [0.010  0    ]    R = [0.4486  0       0  ]
    [0     0.95]        [1  0]        [0      0.005]        [0       2.2878  0  ]
                        [0  1]                              [0       0       1.0]
```

### Notas
"No quiero que se pierdan en los números. Lo importante es que **cada
celda tiene un significado físico**. Por ejemplo, el 0.95 en F[1,1]
significa 'creo que la turbulencia del mercado cambia despacio,
mantiene 95% de su valor entre un tick y el siguiente'. El 0.4486 en
R[0,0] es la varianza empírica del depth_imbalance medida sobre 39
mil ticks de datos limpios. No son números mágicos, son medidas."

---

## SLIDE 7 — Filtro de Kalman: los dos pasos

### Título
**Los dos pasos del filtro de Kalman: predecir + corregir**

### Bullets
- **Paso 1: PREDECIR.**
  - Tomamos el estado anterior, le aplicamos F, y obtenemos una predicción de cómo se ve el estado ahora.
  - Pero la predicción tiene incertidumbre — sumamos Q.
- **Paso 2: CORREGIR.**
  - Llega una nueva observación z (depth_imb, ofi_l1, vol_30s).
  - Comparamos: ¿qué tanto se equivocó la predicción?
  - Combinamos predicción + observación, **pesado por la incertidumbre de cada una**.
  - Si la observación es muy ruidosa → confiamos más en la predicción.
  - Si la predicción es muy incierta → confiamos más en la observación.
- Ese balance se llama **ganancia de Kalman** y se calcula automáticamente.

### Notas
"El filtro nunca confía 100% en una sola fuente. Siempre está balanceando
'lo que pensaba' contra 'lo que veo'. La ganancia de Kalman es ese
balance, y es óptima en el sentido de mínimo error cuadrático medio."

---

## SLIDE 8 — Filtro de Kalman en acción

### Título
**El filtro de Kalman en operación**

### Figura
**`p04_kalman_trace.png`** — tres paneles apilados:
- arriba: precio mid de BTCUSDT durante ~33 minutos
- en medio: presión estimada por Kalman (verde = compradora, rojo = vendedora)
- abajo: régimen estimado por Kalman

### Bullets
- Cada punto del gráfico viene de combinar 3 features ruidosas en 2 números limpios.
- Cuando el precio sube, la presión se vuelve verde (compradores empujan).
- Cuando el precio cae o lateraliza, la presión se vuelve roja o cero.
- El régimen es más estable — refleja la "turbulencia" promedio.

### Notas
"Acá pueden ver al filtro funcionando. Arriba el precio sube y baja en
saltos pequeños. En medio, la presión estimada por el filtro:
positiva = compradores empujan, negativa = vendedores empujan. Y abajo
el régimen, que se mueve más despacio. Esto es lo que va a usar el
algoritmo Hedge para decidir."

---

## SLIDE 9 — Filtro de Kalman: features observables

### Título
**Las features observables (lo que el filtro recibe como entrada)**

### Figura
**`p05_features_trace.png`** — tres paneles apilados con depth_imbalance, ofi_l1, vol_30s.

### Bullets
- **depth_imbalance**: ¿hay más volumen del lado de los bids o de los asks?
  - +1 = todo bid, −1 = todo ask, 0 = balanceado.
- **ofi_l1**: cambio neto en la profundidad del top of book entre dos ticks.
  - + = los compradores están entrando, − = los vendedores.
- **vol_30s**: desviación estándar de los retornos del mid en los últimos 30 segundos.
  - Mide turbulencia de corto plazo.

### Notas
"Estas son las tres mediciones crudas que entran al filtro. Son
features estándar de microestructura. Lo que hace el Kalman es
fusionar las tres en dos números que tienen un significado más
abstracto: dirección y turbulencia."

---

## SLIDE 10 — El algoritmo Hedge: la idea

### Título
**El algoritmo Hedge: aprender a decidir sin saber qué pasará**

### Bullets
- En cada tick tengo 3 acciones disponibles: WAIT, PASSIVE, AGGRESSIVE.
- No sé cuál es la mejor. Quiero aprender por experiencia.
- **Idea**: mantengo una probabilidad para cada acción, parto de uniforme (1/3, 1/3, 1/3).
- Cada vez que tomo una acción y observo su pérdida:
  - Si la pérdida fue alta → bajo la probabilidad de esa acción.
  - Si la pérdida fue baja o negativa → subo su probabilidad.
- Después de muchos ticks, las acciones que pierden poco dominan la distribución.

### Notas
"Hedge es como una bolsa de inversión: las acciones ganadoras crecen,
las perdedoras se encogen. Lo elegante es que la regla matemática es
una sola línea, y tiene garantías formales sobre qué tan cerca está
del óptimo."

---

## SLIDE 11 — El algoritmo Hedge: la regla

### Título
**La regla de actualización de Hedge**

### Bullets
- **Antes** del tick: tengo pesos w(WAIT), w(PASSIVE), w(AGGRESSIVE) que suman 1.
- **Muestreo** una acción proporcional a esos pesos.
- **Observo** la pérdida L de la acción que tomé (slippage + λ·adverse).
- **Actualizo** sólo el peso de esa acción:

```
w_nuevo(acción) = w_antiguo(acción) × exp(−η × L)
```

- **Renormalizo** para que sigan sumando 1.
- η es la tasa de aprendizaje (la usamos = 0.10).

### Notas
"Una sola línea. Si L es grande (mala acción), exp(−η·L) es chico, el
peso se encoge. Si L es pequeña o negativa (buena acción), exp(−η·L)
es ≥ 1, el peso crece o se mantiene. Y porque es multiplicativo, las
buenas decisiones compuestan exponencialmente. Así es como aprende."

---

## SLIDE 12 — El algoritmo Hedge: ¿por qué exp?

### Título
**¿Por qué exponencial y no lineal?**

### Bullets
- Si usáramos resta lineal: `w(acción) = w(acción) − η·L`.
  - Problema 1: el peso podría volverse negativo. Una probabilidad negativa no tiene sentido.
  - Problema 2: la velocidad de actualización es la misma para una acción que pierde mucho y una que pierde poco.
- Con `exp(−η·L)`:
  - El peso siempre es positivo.
  - Una acción que pierde 2× tiene su peso dividido por una constante mayor (escala multiplicativa).
  - Es la solución óptima para minimizar el "regret" — un teorema clásico de Freund & Schapire (1997).

### Notas
"La forma exponencial no es arbitraria. Es la que aparece naturalmente
cuando uno deriva el algoritmo óptimo bajo la métrica de regret
acumulado. No vamos a ver la derivación, pero quédense con esto: hay
una garantía matemática de que Hedge nunca se aleja demasiado del
mejor escenario posible."

---

## SLIDE 13 — Hedge condicionado por estado (1D bucketed)

### Título
**Hedge condicionado: una distribución por bucket de presión**

### Bullets
- Hedge clásico: **una sola** distribución de probabilidades para todos los ticks.
- Hedge condicionado: **una distribución por estado del mercado**.
- Discretizamos la presión en 6 buckets:

| Bucket | Rango de presión | Significado |
|---|---|---|
| 0 | ≤ −0.5 | Vendedores fuertes |
| 1 | (−0.5, −0.2] | Vendedores moderados |
| 2 | (−0.2, 0] | Casi balanceado, lado vendedor |
| 3 | (0, +0.2] | Casi balanceado, lado comprador |
| 4 | (+0.2, +0.5] | Compradores moderados |
| 5 | > +0.5 | Compradores fuertes |

- Cada bucket corre su propio Hedge — aprende independientemente qué hacer en su contexto.

### Notas
"En lugar de una sola política, tengo seis. La política para
'compradores fuertes' aprende muy distinto a la política para
'vendedores fuertes'. La idea es que la mejor acción depende del
estado del mercado, así que tiene sentido tener una política
distinta por estado."

---

## SLIDE 14 — La política aprendida

### Título
**La política aprendida — qué hace en cada estado**

### Figura
**`p07_per_bucket_final_weights.png`** — barras apiladas por bucket.

### Bullets
- Cada barra suma 1 = una distribución de probabilidades.
- 🟡 amarillo = WAIT, 🔴 rojo = PASSIVE, 🟢 verde = AGGRESSIVE.
- **Bucket 1 (vendedores moderados)**: 88% WAIT — "si están vendiendo, no compres ahora, espera".
- **Bucket 4 (compradores moderados)**: 86% AGGRESSIVE — "si están comprando, cruza el spread antes de que el precio se mueva".
- **Buckets 2 y 3 (cerca del cero)**: distribuciones mixtas — el filtro no tiene una señal clara.
- **Buckets 0 y 5 (extremos)**: muy pocas visitas, pesos casi uniformes — el algoritmo nunca tuvo suficientes datos para aprender.

### Notas
"Esta es la política que el sistema aprendió **por sí solo**. Nadie
le dijo 'cuando hay vendedores, espera'. El algoritmo lo descubrió
porque cada vez que cruzó el spread con vendedores empujando,
perdió plata. Después de miles de ejemplos, la probabilidad de
WAIT en ese bucket subió a 0.88. Eso es aprendizaje en línea funcionando."

---

## SLIDE 15 — Cómo aprende — convergencia visual

### Título
**Cómo aprende — convergencia de los pesos a lo largo de la sesión**

### Figura
**`p06_hedge_weights_evolution.png`** — cuatro paneles, uno por bucket interno.

### Bullets
- Cada panel muestra cómo evolucionan los pesos a lo largo de **una sesión completa** (21 horas, 76k ticks).
- Al inicio: distribución uniforme (1/3, 1/3, 1/3).
- A medida que avanzan los ticks: las áreas de colores se separan.
- **Bucket 1**: el área amarilla (WAIT) crece y domina.
- **Bucket 4**: el área verde (AGGRESSIVE) crece y domina.
- **Buckets 2 y 3**: distribuciones mixtas, más oscilación.
- A los pocos miles de ticks ya está convergido, pero sigue ajustándose.

### Notas
"La curva de aprendizaje es visualmente clara. Empiezan todas mezcladas
y se separan. La velocidad de separación depende de qué tan claro es
el ganador — bucket 1 y 4 son obvios y se separan rápido. Bucket 2 y 3
son más ambiguos, las distribuciones quedan mezcladas. El algoritmo
NO fuerza una respuesta donde no la hay; refleja la estructura de los
datos."

---

## SLIDE 16 — Metodología estadística

### Título
**Metodología estadística — A/B pareado**

### Bullets
- Para cada par (modo, sesión, semilla):
  - Inicio de cero el filtro de Kalman y la política.
  - Replay tick por tick toda la sesión.
  - Acumulo: pérdida total, slippage, adverse, mezcla de acciones.
- **A/B pareado por semilla**: comparo `marginal[seed=k]` vs `1D[seed=k]` con la **misma semilla**.
  - Esto cancela el ruido del muestreo aleatorio.
- 10 semillas × 4 sesiones × 2 modos = 80 corridas.
- Estadístico: **t pareado** sobre 10 diferencias por sesión.

### Notas
"El A/B pareado es lo mismo que en pruebas clínicas. Comparo dos
condiciones sobre los mismos sujetos. La 'semilla' acá es el generador
de números aleatorios que decide qué acción muestrea en cada tick.
Si fijo la semilla y cambio sólo la política, cualquier diferencia
viene de la política, no de la suerte. Y como tengo 10 semillas
independientes, puedo calcular un t pareado clásico."

---

## SLIDE 17 — La regla de las dos sesiones

### Título
**La regla de las dos sesiones — corrección por comparaciones múltiples**

### Bullets
- Con 16 comparaciones simultáneas (4 sesiones × 4 modos), por puro azar esperaríamos ~0.8 falsos positivos a |t| > 2.
- **Regla**: para promover una variante, debe vencer al baseline en **al menos 2 sesiones independientes** con |t| ≥ 2.0 y mismo signo.
- Equivale aproximadamente a una corrección de Bonferroni, pero usando réplicas en lugar de bajar el umbral.
- Esta regla nació después de ver `ofi_window` ganar UNA vez (|t|=2.37) y NO replicar (|t|=0.23). Una hipótesis se descartó así.

### Notas
"Esto es importante: si yo corro 16 experimentos al 5%, voy a tener
en promedio 0.8 victorias por azar. Eso me podía haber engañado. La
regla de las dos sesiones es mi forma de mitigar ese riesgo: requiero
que el resultado se replique antes de confiar en él. Es más conservadora
que Bonferroni y mejor adaptada al contexto."

---

## SLIDE 18 — Resultado central — primera mitad

### Título
**Resultado #1: condicionar sobre presión vence a no condicionar**

### Figura
**`p01_marginal_vs_1d_bars.png`** — barras de pérdida total por sesión, marginal vs 1D.

### Bullets
- Las 4 sesiones, las 10 semillas: **1D bucketed pierde menos en TODAS**.

| Sesión | marginal | 1D bucketed | Mejora |
|---|---|---|---|
| S1 (39k ticks) | 1985 | 1415 | **−28.7%** |
| S2 (39k ticks) | 2131 | 1562 | **−26.7%** |
| S3 (64k ticks) | 3817 | 2500 | **−34.5%** |
| S7 (76k ticks) | 4155 | 2869 | **−31.0%** |

- Conclusión: **el filtro de Kalman gana su lugar.** Saber el estado del mercado mejora la decisión consistentemente entre 27 y 35 %.

### Notas
"Este es el resultado positivo central. Cuatro sesiones independientes,
diez semillas en cada una, cuarenta comparaciones pareadas en total.
En las cuarenta, 1D vence a marginal. La mejora media es ~30%. Eso
quiere decir que el filtro de Kalman está aportando información útil:
no es decorativo."

---

## SLIDE 19 — Resultado central — significancia estadística

### Título
**Significancia estadística: t pareado por sesión**

### Figura
**`p02_marginal_vs_1d_paired.png`** — barras con etiquetas de |t|.

### Bullets
- Diferencia pareada (1D − marginal) — siempre negativa (1D gana).
- Estadísticos t por sesión:
  - S1: |t| = 64.2
  - S2: |t| = 36.8
  - S3: |t| = 95.1
  - S7: |t| = 105.3
- Para n=10, |t|=2.3 es ya el umbral del 5%. Estamos uno o dos órdenes de magnitud por encima.
- **Resultado robusto, no es un artefacto del azar.**

### Notas
"En estadística, |t|=2 es el umbral mínimo de significancia, |t|=3 ya
se considera convincente, |t|=10 es 'sin duda alguna'. Acá tenemos
|t| de 36 a 105. La probabilidad de obtener estos valores por azar
es astronómicamente pequeña."

---

## SLIDE 20 — Resultado central — visualización por semilla

### Título
**Por semilla: 40 / 40 victorias para 1D**

### Figura
**`p03_marginal_vs_1d_scatter.png`** — 4 paneles, cada uno con 10 puntos vs línea y=x.

### Bullets
- Cada punto = una semilla. Eje x = pérdida con marginal. Eje y = pérdida con 1D.
- Línea diagonal = empate.
- **Cada punto está debajo de la línea**: 1D ganó esa semilla.
- 4 sesiones × 10 semillas = 40 comparaciones, 40 victorias para 1D.

### Notas
"Esta es la prueba visual final del resultado central. Si una sola
semilla, en una sola sesión, hubiera estado por encima de la línea,
sería interesante. Pero todas están debajo. La política condicionada
por presión no es 'mejor en promedio' — es mejor SIEMPRE."

---

## SLIDE 21 — Cómo se logra la mejora

### Título
**¿De dónde viene la mejora? La distribución de acciones cambia**

### Figura
**`p08_action_mix_marginal_vs_1d.png`** — barras apiladas por modo, una columna por sesión.

### Bullets
- En cada sesión, comparamos la mezcla promedio de acciones.
- 1D toma:
  - Más WAIT (+3-5 pp).
  - Menos PASSIVE (−5-7 pp).
  - Casi igual AGGRESSIVE.
- En palabras simples: 1D **aprende a no postear órdenes pasivas en momentos donde son tóxicas**.
  - Marginal posa pasivas en cualquier momento porque no tiene contexto.
  - 1D posa pasivas sólo cuando la presión sugiere que está OK.

### Notas
"La mejora no viene de descubrir un truco mágico. Viene de SABER CUÁNDO
no actuar. La política marginal pone órdenes pasivas en cualquier
contexto, incluso cuando los vendedores están empujando — entonces se
queda atrapada con compras tóxicas. La política 1D aprende que en
contextos vendedores no conviene poner pasivas, y así se ahorra el
costo. Es así de simple."

---

## SLIDE 22 — La línea de ahorro

### Título
**El ahorro acumulado vs un baseline trivial**

### Figura
**`p09_savings_vs_aggr.png`** — dos paneles: pérdida acumulada y línea de ahorro.

### Bullets
- Comparamos contra el baseline más simple posible: **siempre cruzar el spread (always-AGGRESSIVE)**.
- En la sesión S7 (21 horas, 76k ticks):
  - Baseline acumula 3,728 USDT de pérdida.
  - Política 1D acumula 2,891 USDT.
  - **Ahorro: $838.64 sobre 76k decisiones, ~$0.011 por decisión.**
- La línea verde abajo crece monotónicamente — el sistema gana terreno todo el tiempo.

### Notas
"Esta es una visualización que les puede gustar a no-técnicos. Arriba,
dos curvas de pérdida acumulada — la baseline trivial y la nuestra,
sobre 21 horas. Abajo, la diferencia: el dinero que se ahorró el
sistema. La línea es siempre verde, siempre creciente. Eso es la
política funcionando."

---

## SLIDE 23 — Resultado #2 — el negativo

### Título
**Resultado #2: agregar más estado NO ayuda**

### Bullets
- Probamos cuatro variantes "más ricas" sobre el mismo 1D:
  - 2D regimen (presión × régimen del Kalman)
  - vol_delta (presión × volatilidad rolling 60s)
  - ofi_window (presión × OFI rolling 60s)
  - spread_delta (presión × cambio de spread 60s)
- 4 sesiones × 10 semillas × 4 variantes = **160 corridas**.
- **Resultados:**
  - 2D regimen: pierde 20/20 vs 1D
  - vol_delta: pierde 40/40 vs 1D
  - spread_delta: pierde 40/40 vs 1D
  - ofi_window: gana 1 sesión (|t|=2.37), pero NO replica en otra sesión (|t|=0.23) — descartada por la regla de las dos sesiones.
- Conclusión: **agregar dimensiones de estado al condicionamiento no es gratis.** Aplicado mecánicamente, perjudica.

### Notas
"Este resultado es muy interesante por ser inesperado. La intuición de
ingeniería es 'más información = mejor decisión'. Acá pasa lo
contrario. Probamos cuatro segundos ejes, todos fallan. Y fallan por
el mismo mecanismo, lo cual es la pista."

---

## SLIDE 24 — La barrera de toxicidad

### Título
**¿Por qué fallan? La barrera de toxicidad**

### Figura
**`p10_loss_decomposition.png`** — Pareto de slippage vs adverse, 4 modos × 4 sesiones.

### Bullets
- En cada sesión, los 4 modos forman una **línea ascendente** en el plano (slippage, adverse).
- Cualquier modo que ahorra slippage **paga proporcionalmente más adverse**.
- **Mecanismo común**: las cuatro señales son detectores de "fill fácil".
  - Cuando la señal se activa, la política aprende a postear pasivas.
  - Pero las pasivas que se ejecutan en esos momentos son **flujo tóxico**: el precio se mueve en contra justo después.
- 1D pressure-only es la política con MÁS slippage — y la que MENOS adverse paga. Es la frontera más eficiente que encontramos.

### Notas
"Esta es la figura central de la tesis. Cada línea conecta los cuatro
modos sobre la MISMA sesión. Todas suben de izquierda a derecha. Esto
significa que NO hay manera de ahorrar slippage SIN incrementar el
adverse. El espacio de políticas bucketed-Hedge sobre este problema
está en una frontera Pareto plana. Nadie la rompe. La hipótesis que
generó esta tesis era 'más estado = mejor política' y lo que la
evidencia muestra es que eso es FALSO en este régimen, por una razón
mecanística específica."

---

## SLIDE 25 — Lo que aprendimos

### Título
**Lo que aprendimos — síntesis**

### Bullets
- **Resultado positivo:** condicionar Hedge sobre la presión Kalman vence a Hedge marginal en 4/4 sesiones, ~30% de mejora, |t| > 36.
- **Resultado negativo:** agregar un segundo eje de estado (régimen, vol, OFI rolling, spread rolling) no mejora — y mecánicamente empeora.
- **Mecanismo identificado:** la barrera de toxicidad. Toda señal del segundo eje detecta "fill fácil" → la política toma más pasivas → más toxicidad → mayor adverse.
- **Implicación:** el espacio de políticas bucketed-Hedge tiene un techo en este régimen. Para romperlo hay que cambiar **el espacio de acciones**, no el espacio de estado.

### Notas
"Si me piden resumir la tesis en 4 puntos, son estos. El positivo es
sólido. El negativo es mecanístico, lo cual es más interesante que
'no funcionó'. Y la conclusión apunta a la dirección del próximo
experimento: cambiar el set de acciones."

---

## SLIDE 26 — Próximos pasos

### Título
**Próximos pasos**

### Bullets
- **Cambiar el espacio de acciones:** probar Hedge con tamaño variable de orden.
  - Acción set viejo: {WAIT, PASSIVE_1, AGGRESSIVE_1}.
  - Acción set nuevo: {WAIT, PASSIVE_{0.5,1,2}, AGGRESSIVE_{0.5,1,2}}.
- Hipótesis: con tamaño variable, la política puede modular su exposición a toxicidad por fill (pasivas pequeñas en buckets neutros, agresivas grandes en buckets direccionales).
- Esto crea **una nueva frontera Pareto**, no un nuevo punto sobre la vieja.
- Implementación: extender el enum de Action y escalar slippage/adverse proporcional al tamaño. ~medio día de código.
- A/B vs 1D fixed-size sobre las 4 sesiones, regla de dos sesiones.

### Notas
"El próximo capítulo de la tesis es testear esta hipótesis. Si funciona,
es el cierre positivo de la tesis. Si no funciona, el resultado
negativo central — la barrera de toxicidad — sigue siendo publicable
por sí mismo, porque identifica un techo estructural en una familia
entera de políticas."

---

## SLIDE 27 — Cierre

### Título
**Cierre**

### Bullets
- ✅ Sistema funciona end-to-end sobre datos reales de Binance: ingesta, libro, features, Kalman, Hedge, simulador.
- ✅ Resultado positivo robusto: condicionar sobre presión vence a no condicionar (4 sesiones, |t| ≥ 36).
- ✅ Resultado negativo mecanístico: la barrera de toxicidad limita las políticas bucketed-Hedge.
- 🔄 En curso: nueva variante con tamaño de orden variable.
- 🌐 GUI tipo Wireshark para visualización en vivo y replay.
- 📁 Todo reproducible: 240 corridas guardadas en JSON, plots regenerables.

### Notas
"En resumen: el sistema funciona, tenemos resultados sólidos, sabemos
exactamente cuál es la próxima pregunta. La tesis está sólida y el
trabajo que falta para terminarla es acotado y bien definido. Gracias."

---

## Anexo — listado de figuras (rutas exactas)

Para copiar/pegar al insertar imágenes en PowerPoint o Keynote:

```
data/derived/figures/preview/p01_marginal_vs_1d_bars.png
data/derived/figures/preview/p02_marginal_vs_1d_paired.png
data/derived/figures/preview/p03_marginal_vs_1d_scatter.png
data/derived/figures/preview/p04_kalman_trace.png
data/derived/figures/preview/p05_features_trace.png
data/derived/figures/preview/p06_hedge_weights_evolution.png
data/derived/figures/preview/p07_per_bucket_final_weights.png
data/derived/figures/preview/p08_action_mix_marginal_vs_1d.png
data/derived/figures/preview/p09_savings_vs_aggr.png
data/derived/figures/preview/p10_loss_decomposition.png
```

Para regenerar todas las figuras desde cero (después de re-correr la harness):

```bash
PYTHONPATH=. .venv/bin/python scripts/_ab_1d_vs_marginal_4sessions.py
PYTHONPATH=. .venv/bin/python scripts/_plots_preview.py
```

---

## Notas finales para el orador

- **Ritmo sugerido**: ~30-45 segundos por slide → 15-20 min total para las 27 slides.
- **Si te falta tiempo**: puedes saltar slides 7, 9, 12 (son detalles técnicos del Kalman/Hedge — el público engineer puede no necesitarlos).
- **Si te sobra tiempo**: agrega una demo en vivo del GUI Wireshark replay para una o dos sesiones (slide 26 anticipa esto).
- **Lo más importante a transmitir**:
  1. El sistema **funciona** (resultado positivo, 4 sesiones, |t|>36).
  2. El sistema tiene un **techo identificado** (barrera de toxicidad).
  3. El **próximo experimento** está bien definido y es factible.

Si los miembros del jurado son ingenieros sin formación financiera, las
slides 3-4 (problema y arquitectura) y 18-22 (resultados) son las más
importantes. Las slides 5-9 (Kalman) y 10-15 (Hedge) son las que
demuestran tu mastery del sistema interno; tenerlas pero estar listo
para saltarlas si el público está perdido.
