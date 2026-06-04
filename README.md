# Mini Turnitin v2 — Sistema Híbrido de Detección de Similitud Textual

Sistema optimizado estilo Turnitin con arquitectura híbrida: **chunking + FAISS + TF-IDF + MinHash + embeddings semánticos + detección de IA**. Interfaz web con Streamlit, tres modos de análisis, persistencia de resultados con session state. Diseñado para CPU, en inglés, con énfasis en velocidad y bajo consumo de RAM.

---

## Pipeline

```
PDF / DOCX / TXT
  ↓ PyMuPDF / python-docx
Texto plano
  ↓ spaCy (en_core_web_sm)
Preprocesamiento + Sentence Splitting
  ↓
Chunking (~100 words/chunk, overlap=20)
  ↓
sentence-transformers / ONNX (all-MiniLM-L6-v2, 384d)
  ↓
Embeddings → Cache (diskcache)
  ↓
FAISS Index (IndexFlatIP) + Cross-Encoder reranker
  ↓
Top-k similar chunks
  ↓
TF-IDF (ngram 1-3) + MinHash (k=128) reranking
  ↓
Score ponderado final + Detección IA (GPT-2 perplexity + burstiness)
  ↓
Resultados persistentes en st.session_state
```

---

## Arquitectura del proyecto

```
mini_turnitin/
├── app.py                     # Interfaz Streamlit (3 modos, session state)
├── app/
│   ├── config.py              # Configuración centralizada (dataclass)
│   ├── services/
│   │   ├── analyzer.py        # Orquestador (analyze + analyze_corpus)
│   │   ├── semantic.py        # all-MiniLM-L6-v2 + ONNX + batch + cache
│   │   ├── lexical.py         # TF-IDF + MinHash
│   │   ├── reranker.py        # Cross-Encoder (cross-encoder/stsb-MiniLM-L6-v2)
│   │   └── detector.py        # Detección IA (GPT-2 perplexity + burstiness)
│   ├── models/
│   │   ├── cache.py           # Cache de embeddings (diskcache)
│   │   └── index.py           # FAISS index (IndexFlatIP / IndexHNSWFlat)
│   └── utils/
│       ├── preprocess.py      # Limpieza, lematización, sentence splitting
│       ├── chunking.py        # Chunking por oraciones (~100 words, overlap)
│       └── pdf.py             # Extracción de PDF / DOCX / TXT + load_corpus
├── data/
│   └── corpus/                # 34 PDFs académicos (referencia)
├── cache/                     # Cache persistente de embeddings
├── onnx/                      # Modelo exportado a ONNX (auto-generado)
├── .venv/                     # Entorno virtual (Python 3.12)
├── pyproject.toml
└── README.md
```

---

## Funcionalidades

### 3 modos de análisis

| Modo | Descripción |
|------|-------------|
| **Comparar dos documentos** | Análisis pairwise con detección de fragmentos similares + score híbrido + detección IA |
| **Comparar contra corpus** | Documento vs. 34 PDFs académicos indexados con FAISS |
| **Detectar IA** | Análisis de texto con GPT-2 (perplejidad + burstiness) |

### Score final (modo similitud)

| Componente | Peso | Entrada |
|-----------|------|---------|
| TF-IDF    | 0.25 | Texto preprocesado (lemas), ngramas 1–3 |
| MinHash   | 0.15 | Texto preprocesado (tokens), 128 permutaciones |
| Semántico | 0.60 | Embedding del documento completo (384d) |

```
final = 0.25 · score_tfidf + 0.15 · score_minhash + 0.60 · score_semantico
```

**Niveles de alerta**: `>0.75 Alto` | `0.45–0.75 Medio` | `<0.45 Bajo`

### Detección de IA

Basada en perplejidad (GPT-2) + burstiness (variación de longitud de oraciones):

```
score_ia = (1 - perplexity / 200) · 0.6 + (1 - burstiness) · 0.4
```

**Niveles**: `>0.60 Probable IA` | `≤0.60 Probable humano`

---

## Módulos

### `app/config.py` — Configuración centralizada

Dataclass `Config` con todos los parámetros del sistema:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `model_name` | `all-MiniLM-L6-v2` | Modelo de embeddings |
| `embedding_dim` | 384 | Dimensiones del embedding |
| `cross_encoder_model` | `cross-encoder/stsb-MiniLM-L6-v2` | Reranker |
| `chunk_max_words` | 100 | Tamaño de chunk |
| `chunk_overlap_words` | 20 | Superposición entre chunks |
| `faiss_index_type` | `flat` | Tipo de índice FAISS |
| `tfidf_weight` | 0.25 | Peso TF-IDF |
| `minhash_weight` | 0.15 | Peso MinHash |
| `semantic_weight` | 0.60 | Peso semántico |
| `corpus_dir` | `data/corpus` | Directorio del corpus |

### `app/services/semantic.py` — Embeddings semánticos

Usa `all-MiniLM-L6-v2` (384 dims, solo inglés). Optimizaciones:

- **ONNX Runtime**: exporta el modelo a ONNX en primera ejecución; hasta 2–3× más rápido en CPU
- **Batching**: codifica múltiples chunks por lote (batch_size=64 por defecto)
- **Caching**: guarda embeddings calculados en disco (diskcache) para evitar recomputación
- **Carga lazy**: el modelo se carga en el primer llamado, no al importar

### `app/utils/preprocess.py` — Preprocesamiento NLP

Pipeline de limpieza textual usando spaCy `en_core_web_sm`:

1. **Minúsculas + normalización**: `re.sub(r"[^a-z'\s]", "", text.lower())`
2. **Tokenización** con pipeline de spaCy
3. **Eliminación de stopwords**
4. **Lematización** a forma canónica

```
"Students are learning NLP and data science"
→ ["student", "learn", "nlp", "data", "science"]
```

Además expone `split_sentences(text)` para división en oraciones usando spaCy.

### `app/utils/chunking.py` — Chunking con superposición

Divide documentos en fragmentos de ~100 palabras respetando límites de oraciones, con 20 palabras de superposición entre chunks consecutivos para evitar perder similitudes en los bordes.

```python
chunk_by_sentences(text, max_words=100, overlap_words=20) → list[str]
```

### `app/services/lexical.py` — Análisis léxico

**TF-IDF** con n-gramas (1–3) y similitud coseno. **MinHash** con 128 permutaciones para detección de near-duplicates.

### `app/services/reranker.py` — Cross-Encoder

Usa `cross-encoder/stsb-MiniLM-L6-v2` para re-puntuar pares de fragmentos candidatos con una red Transformer que procesa ambos textos simultáneamente, obteniendo puntuaciones más precisas que la similitud coseno de embeddings independientes.

### `app/models/index.py` — FAISS Index

Wrapper sobre `faiss.IndexFlatIP` (inner product) con normalización L2 de vectores, equivalente a similitud coseno. Soporta también `IndexHNSWFlat` para búsqueda aproximada en corpus grandes.

- `add(embeddings, doc_ids, chunk_texts)`: indexa embeddings de chunks
- `search(query_emb, k=5)`: recupera top-k chunks similares

### `app/models/cache.py` — Embedding Cache

Usa `diskcache` para persistir embeddings en disco. Clave = `sha256(text):model_name`. Reduce tiempo de corpus mode de ~2 min a ~10 s con cache poblado.

### `app/utils/pdf.py` — Extracción de documentos

| Formato | Librería |
|---------|----------|
| PDF | PyMuPDF (`fitz`) |
| DOCX | `python-docx` |
| TXT | Lectura directa UTF-8 |

Incluye `load_corpus(corpus_dir)` que escanea y extrae todos los documentos del directorio, retornando lista con texto, nombre y ruta.

### `app/services/detector.py` — Detección de IA

Usa GPT-2 para calcular perplejidad (loss promedio del modelo al predecir cada token) y burstiness (coeficiente de variación de longitudes de oraciones). Integrado en los modos de comparación y disponible como herramienta independiente. Carga lazy del modelo.

### `app/services/analyzer.py` — Orquestador

#### `analyze(doc1, doc2)`

1. Preprocesa ambos documentos para TF-IDF y MinHash
2. Chunking (~100 words, overlap 20) para ambos
3. Embeddings por chunk con cache
4. FAISS index sobre chunks del doc2
5. Búsqueda de chunks similares del doc1
6. Reranking con Cross-Encoder
7. Score global semántico (embedding del documento completo)
8. Detección de IA sobre doc1
9. Pondera: `final = 0.25·TF-IDF + 0.15·MinHash + 0.60·Semántico`

#### `analyze_corpus(doc, corpus_dir)`

1. Carga + chunking + embeddings del query
2. Indexa todo el corpus en FAISS (ThreadPoolExecutor con 2 workers)
3. Busca cada chunk del query contra el índice
4. Agrega hits semánticos por documento
5. Computa TF-IDF + MinHash por documento
6. Ordena resultados
7. Retorna ranking descendente con detección IA del query

### `app.py` — Interfaz Streamlit

Tres modos en pestañas (radio horizontal). Los resultados se almacenan en `st.session_state` para persistir entre reruns de Streamlit, evitando pérdida de datos al interactuar con widgets. Usa Material Symbols para íconos.

---

## Optimizaciones clave

| Técnica | Problema que resuelve |
|---------|----------------------|
| **Chunking con overlap** | Documentos enteros exceden 512 tokens del Transformer; overlap evita perder similitudes en cortes |
| **FAISS** | Búsqueda O(log n) en lugar de O(n·m) pairwise |
| **ONNX Runtime** | Inferencia 2–3× más rápida en CPU |
| **diskcache** | Evita recalcular embeddings entre ejecuciones |
| **Batching** | Reduce overhead de llamadas múltiples al modelo |
| **Cross-Encoder** | Mayor precisión que coseno en fragmentos candidatos |
| **Carga lazy** | Modelos ML se cargan solo al primer request, no al importar |
| **Session state** | Resultados persisten ante reruns de Streamlit |
| **Config centralizada** | Todos los parámetros en un dataclass |

---

## Cómo ejecutar

```bash
# Activar entorno
source .venv/bin/activate

# Instalar dependencias
uv sync

# Lanzar interfaz
streamlit run app.py
```

---

## Dependencias

| Librería | Propósito |
|----------|-----------|
| spaCy | Pipeline NLP (tokenización, lematización, sentence splitting) |
| scikit-learn | TF-IDF Vectorizer + cosine similarity |
| datasketch | MinHash (Jaccard aproximado) |
| sentence-transformers | Embeddings semánticos (all-MiniLM-L6-v2) |
| **faiss-cpu** | Búsqueda ANN en índices vectoriales |
| **onnxruntime + optimum** | Aceleración de inferencia en CPU |
| **diskcache** | Cache persistente de embeddings |
| **polars** | Procesamiento eficiente de resultados del corpus |
| PyMuPDF | Extracción de texto de PDFs |
| python-docx | Extracción de texto de DOCX |
| streamlit | Interfaz de usuario web |
| tqdm | Barras de progreso |

---

## Limitaciones conocidas

- Modelo semántico limitado a 512 tokens por entrada (mitigado por chunking)
- TF-IDF no captura paráfrasis (peso bajo: 0.25)
- ONNX export requiere ~2 min en primera ejecución
- Corpus mode lento en primera indexación (embeddings se cachean)
- Solo inglés (modelo all-MiniLM-L6-v2 no es multilingüe)
