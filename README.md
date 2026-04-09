# DistortionScore

Herramienta para detectar y cuantificar defectos en estructuras atómicas cristalinas mediante una puntuación de distorsión robusta por átomo (`d_RB`). Combina descriptores SOAP, reducción de dimensionalidad PCA y estadística robusta (MCD) para identificar átomos cuyo entorno local difiere significativamente del bulto de referencia.

---

## Tabla de contenidos

1. [Fundamentos físicos](#fundamentos-físicos)
   - [Entorno local atómico](#entorno-local-atómico)
   - [Descriptor SOAP](#descriptor-soap)
   - [Reducción de dimensionalidad: PCA](#reducción-de-dimensionalidad-pca)
   - [Determinante de Covarianza Mínima (MCD)](#determinante-de-covarianza-mínima-mcd)
   - [Distancia de Mahalanobis y score d\_RB](#distancia-de-mahalanobis-y-score-d_rb)
   - [Umbral de detección](#umbral-de-detección)
2. [Flujo del algoritmo](#flujo-del-algoritmo)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Parámetros](#parámetros)
6. [Archivos de salida](#archivos-de-salida)
7. [Interpretación física del score](#interpretación-física-del-score)
8. [Referencias](#referencias)

---

## Fundamentos físicos

### Entorno local atómico

En un cristal perfecto, cada átomo ocupa una posición de equilibrio dentro de una red de Bravais y sus vecinos forman un entorno geométricamente regular. Los [defectos cristalinos](https://es.wikipedia.org/wiki/Defecto_cristalino) (vacantes, intersticiales, dislocaciones, límites de grano) distorsionan ese entorno de manera característica.

La clave del método es **representar el entorno local de cada átomo como un vector numérico** y luego medir cuánto se desvía ese vector respecto a la distribución del bulto de referencia.

---

### Descriptor SOAP

El descriptor [SOAP](https://en.wikipedia.org/wiki/Smooth_overlap_of_atomic_positions) (*Smooth Overlap of Atomic Positions*) convierte el entorno local de un átomo en un vector de números reales que es **invariante bajo rotaciones y permutaciones** de los átomos vecinos.

#### Densidad atómica local

La densidad de átomos vecinos alrededor del átomo central `i` se expresa como una suma de Gaussianas:

```
ρᵢ(r) = Σⱼ exp(−|r − rⱼ|² / 2σ²)
```

donde `rⱼ` son las posiciones de los vecinos dentro del radio de corte `r_cut` y `σ` es el ancho de la Gaussiana ([función gaussiana](https://es.wikipedia.org/wiki/Funci%C3%B3n_gaussiana)).

#### Expansión en base ortonormal

Esa densidad se expande en una base de funciones radiales `g_n(r)` y [armónicos esféricos](https://es.wikipedia.org/wiki/Arm%C3%B3nicos_esf%C3%A9ricos) `Y_lm(θ,φ)`:

```
ρᵢ(r) = Σ_{n,l,m} c_{nlm} · g_n(r) · Y_lm(θ,φ)
```

#### Vector SOAP (espectro de potencia)

Los coeficientes de la expansión se combinan para obtener un descriptor invariante rotacional, el **espectro de potencia**:

```
p_{n₁n₂l} = π √(8/(2l+1)) · Σ_m (c_{n₁lm})* · c_{n₂lm}
```

Este vector `p` tiene dimensión proporcional a `n_max² · (l_max + 1)` y captura toda la información sobre la geometría local hasta el orden angular `l_max`.

**Parámetros clave:**

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `r_cut`   | Radio de corte (Å) — define el tamaño del entorno considerado | 5.5 |
| `n_max`   | Orden de expansión radial — más alto = más detalle radial | 8 |
| `l_max`   | Orden angular — más alto = más detalle angular | 6 |
| `sigma`   | Ancho de las Gaussianas (Å) — controla la suavidad | 0.4 |

---

### Reducción de dimensionalidad: PCA

El vector SOAP tiene típicamente 100–200 componentes. Antes de ajustar el modelo estadístico se aplica [Análisis de Componentes Principales (PCA)](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales) para:

1. Eliminar componentes redundantes (correlaciones lineales).
2. Evitar matrices de covarianza singulares.
3. Reducir el coste computacional.

PCA encuentra la transformación ortogonal `W` tal que:

```
X_red = X · W
```

donde `W` contiene los autovectores de la [matriz de covarianza](https://es.wikipedia.org/wiki/Matriz_de_covarianza) ordenados por autovalor descendente. Se retienen las componentes que explican el **99% de la varianza total**.

---

### Determinante de Covarianza Mínima (MCD)

El estimador clásico de la media `μ` y la covarianza `Σ` de una muestra multivariante es sensible a valores atípicos. Para una estructura de entrenamiento que pueda contener una pequeña fracción de defectos se emplea el estimador robusto [MCD (Minimum Covariance Determinant)](https://en.wikipedia.org/wiki/Minimum_covariance_determinant).

#### Idea central

Dado un conjunto de `n` descriptores, MCD busca el subconjunto `h` (con `h ≈ 0.95 · n`) cuya [matriz de covarianza](https://es.wikipedia.org/wiki/Matriz_de_covarianza) tiene **determinante mínimo**:

```
(μ̂_MCD, Σ̂_MCD) = arg min_{|H|=h} det( Cov({xᵢ : i ∈ H}) )
```

El subconjunto óptimo `H` corresponde a los átomos cuyo entorno es más homogéneo entre sí, excluyendo automáticamente los outliers. El estimador resultante es mucho más resistente a la contaminación que la covarianza muestral ordinaria.

#### Punto de ruptura

El parámetro `support_fraction = h/n = 0.95` determina la fracción del conjunto de entrenamiento que se considera "limpia". Un valor de 0.95 tolera hasta un 5% de contaminación (defectos en el bulto de referencia).

El algoritmo de búsqueda utilizado es [C-step (Concentration Step)](https://en.wikipedia.org/wiki/Minimum_covariance_determinant#C-step_algorithm), que converge iterativamente al mínimo local.

---

### Distancia de Mahalanobis y score d\_RB

Una vez ajustado el modelo `(μ̂_MCD, Σ̂_MCD)`, la **puntuación de distorsión robusta** `d_RB` de un átomo `i` es su [distancia de Mahalanobis](https://es.wikipedia.org/wiki/Distancia_de_Mahalanobis) al centro de la distribución de referencia:

```
d_RB(i) = sqrt( (xᵢ − μ̂)ᵀ · Σ̂⁻¹ · (xᵢ − μ̂) )
```

donde:

- `xᵢ` — descriptor SOAP del átomo `i` (proyectado por PCA)
- `μ̂` — media robusta MCD del conjunto de entrenamiento
- `Σ̂⁻¹` — inversa de la covarianza robusta MCD

A diferencia de la [distancia euclidiana](https://es.wikipedia.org/wiki/Distancia_euclidiana), la distancia de Mahalanobis tiene en cuenta las correlaciones entre componentes del descriptor y la escala de cada una, lo que la hace invariante a transformaciones lineales.

> **Nota de implementación:** `sklearn.covariance.MinCovDet.mahalanobis()` devuelve `d²`; el código extrae la raíz cuadrada para obtener `d_RB`.

---

### Umbral de detección

Un átomo se clasifica como **defecto/outlier** si `d_RB > umbral`. El umbral se puede calcular de dos maneras:

#### Modo `chi2` (estadístico)

Bajo la hipótesis de que los descriptores siguen una [distribución normal multivariante](https://es.wikipedia.org/wiki/Distribuci%C3%B3n_normal_multivariante), `d_RB²` sigue asintóticamente una [distribución chi-cuadrado](https://es.wikipedia.org/wiki/Distribuci%C3%B3n_chi_cuadrado) con `p` grados de libertad (donde `p` es la dimensión tras PCA):

```
d_RB² ~ χ²(p)
```

El umbral al nivel de confianza `α` es:

```
umbral = sqrt( χ²_{α,p} )
```

Por ejemplo, con `α = 0.975` se marcan como outliers los átomos cuyo `d_RB²` supera el percentil 97.5% de la distribución chi-cuadrado.

#### Modo `empirical` (cuantil de entrenamiento)

El umbral se fija directamente como el cuantil `α` de los scores `d_RB` calculados sobre el propio conjunto de entrenamiento:

```
umbral = quantile(d_RB_train, α)
```

Este modo no asume normalidad y es más apropiado cuando la distribución real de los descriptores se desvía significativamente de la gaussiana.

---

## Flujo del algoritmo

```
Estructura de entrenamiento (bulto perfecto)
            │
            ▼
   [Leer con ASE / parser LAMMPS]
            │
            ▼
   [SOAP] → matriz X_train [N_train × F]
            │
            ▼
   [PCA (99% varianza)] → X_pca [N_train × p]
            │
            ▼
   [MinCovDet] → μ̂, Σ̂  (estimadores robustos)
            │
            ▼
   [Calcular umbral] (chi2 o empírico)
            │
            ▼
       MODELO ENTRENADO
            │
Estructura de prueba (con defectos)
            │
            ▼
   [SOAP] → X_test [N_test × F]
            │
            ▼
   [PCA.transform] → X_test_pca [N_test × p]
            │
            ▼
   [d_RB = sqrt(Mahalanobis²)] → score por átomo
            │
            ▼
   [d_RB > umbral → outlier]
            │
            ▼
   [Exportar CSV, XYZ extendido, JSON, figuras]
```

---

## Instalación

```bash
pip install numpy pandas scipy scikit-learn ase dscribe matplotlib
```

Requiere Python ≥ 3.8.

---

## Uso

```bash
python distortion_score_mcd.py \
    --train  bulk_perfect.dump \
    --test   dump_defectuoso.dump \
    --species Ni \
    --rcut 5.5 \
    --nmax 8 \
    --lmax 6 \
    --sigma 0.4 \
    --contamination 0.07 \
    --threshold-mode chi2 \
    --alpha 0.975 \
    --out-prefix ni_test
```

Para sistemas multicomponente separar las especies con coma:

```bash
--species Fe,Ni
```

---

## Parámetros

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `--train` | archivo | Estructura de referencia (bulto perfecto), formato LAMMPS dump o cualquier formato ASE |
| `--test` | archivo | Estructura a analizar |
| `--species` | string | Símbolo(s) de elementos, separados por coma (ej. `Ni` o `Fe,Ni`) |
| `--format` | string | Formato de archivo ASE (opcional; se detecta automáticamente si se omite) |
| `--rcut` | float | Radio de corte SOAP en Å (defecto: 5.5) |
| `--nmax` | int | Orden de expansión radial SOAP (defecto: 8) |
| `--lmax` | int | Orden de expansión angular SOAP (defecto: 6) |
| `--sigma` | float | Ancho de Gaussiana SOAP en Å (defecto: 0.4) |
| `--contamination` | float | Fracción esperada de outliers en el entrenamiento (defecto: 0.07) |
| `--threshold-mode` | `chi2`/`empirical` | Método para calcular el umbral (defecto: `chi2`) |
| `--alpha` | float | Nivel de confianza para el umbral (defecto: 0.975) |
| `--out-prefix` | string | Prefijo para todos los archivos de salida (defecto: `output`) |
| `--strata` | string | Valores de corte para estratificación, separados por coma (opcional) |

---

## Archivos de salida

| Archivo | Descripción |
|---------|-------------|
| `{prefix}_scores.csv` | Score `d_RB` y flag `is_outlier` por átomo, con posición y símbolo |
| `{prefix}_scored.extxyz` | Estructura en formato Extended XYZ con propiedades `dRB` e `is_outlier` |
| `{prefix}_summary.json` | Metadatos completos: parámetros, estadísticas, umbral, dimensión PCA |
| `{prefix}_hist_dRB.png` | Histograma superpuesto de train vs test con línea de umbral |
| `{prefix}_spatial_*.png` | Scatter 3D de los átomos coloreados por `d_RB` |
| `{prefix}_slices_*.png` | Proyecciones XY a distintas alturas Z |
| `{prefix}_stratification.png` | Gráfico de barras con conteo de átomos por rango de `d_RB` |
| `{prefix}_dashboard.png` | Panel resumen multipanel con métricas y parámetros |

---

## Interpretación física del score

| Rango de `d_RB` | Interpretación física |
|-----------------|----------------------|
| `d_RB < umbral` | Entorno local consistente con el bulto de referencia → átomo en posición regular |
| `d_RB` ligeramente mayor que el umbral | Distorsión leve — posible átomo en la frontera de un defecto extendido |
| `d_RB` muy superior al umbral | Distorsión severa — vacante, intersticial, dislocación, límite de grano, superficie |

El score es **sensible al entorno**, no a la posición absoluta. Dos átomos en posiciones alejadas pueden tener el mismo `d_RB` si sus vecinos están igualmente distribuidos. Esto permite detectar defectos en cualquier orientación y sin conocer a priori su geometría.

---

## Referencias

- **SOAP descriptor:** Bartók, A. P. et al. *On representing chemical environments.* Physical Review B, 87, 184115 (2013). — [Wikipedia: Smooth overlap of atomic positions](https://en.wikipedia.org/wiki/Smooth_overlap_of_atomic_positions)
- **MCD (Minimum Covariance Determinant):** Rousseeuw, P. J. & Van Driessen, K. *A fast algorithm for the minimum covariance determinant estimator.* Technometrics 41, 212–223 (1999). — [Wikipedia: Minimum covariance determinant](https://en.wikipedia.org/wiki/Minimum_covariance_determinant)
- **Distancia de Mahalanobis:** Mahalanobis, P. C. *On the generalised distance in statistics.* Proceedings of the National Institute of Sciences of India, 2, 49–55 (1936). — [Wikipedia: Distancia de Mahalanobis](https://es.wikipedia.org/wiki/Distancia_de_Mahalanobis)
- **Distribución chi-cuadrado:** [Wikipedia: Distribución chi cuadrado](https://es.wikipedia.org/wiki/Distribuci%C3%B3n_chi_cuadrado)
- **PCA:** [Wikipedia: Análisis de componentes principales](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales)
- **Defectos cristalinos:** [Wikipedia: Defecto cristalino](https://es.wikipedia.org/wiki/Defecto_cristalino)
- **Armónicos esféricos:** [Wikipedia: Armónicos esféricos](https://es.wikipedia.org/wiki/Arm%C3%B3nicos_esf%C3%A9ricos)
- **DScribe (biblioteca SOAP):** Himanen, L. et al. *DScribe: Library of descriptors for machine learning in materials science.* Computer Physics Communications, 247, 106949 (2020).
