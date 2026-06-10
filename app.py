import tempfile
from pathlib import Path

import streamlit as st

from app.config import cfg
from app.services.analyzer import analyze, analyze_corpus
from app.services.detector import ai_probability
from app.utils.pdf import extract_text

CORPUS_DIR = cfg.corpus_dir


def extract_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    return extract_text(tmp_path)


def nivel_badge(nivel: str) -> str:
    colors = {"Alto": "red", "Medio": "orange", "Bajo": "green"}
    c = colors.get(nivel, "gray")
    return f":{c}[**{nivel}**]"


def mostrar_resultado(result: dict, titulo: str | None = None):
    if titulo:
        st.subheader(titulo)

    final_pct = result["score_final"] * 100
    st.progress(min(result["score_final"], 1.0))
    col_score, col_nivel = st.columns([1, 1])
    col_score.metric("Score final", f"{final_pct:.1f}%")
    col_nivel.markdown(f"### Nivel de alerta: {nivel_badge(result['nivel'])}")

    st.markdown("#### Componentes del score")
    col1, col2, col3, _ = st.columns([1, 1, 1, 1])
    col1.metric("Léxico (TF-IDF)", f"{result['score_lexico'] * 100:.1f}%",
                help="Similitud basada en n-gramas de texto")
    col2.metric("MinHash", f"{result['score_minhash'] * 100:.1f}%",
                help="Similitud de conjuntos de palabras (near-duplicates)")
    col3.metric("Semántico", f"{result['score_semantico'] * 100:.1f}%",
                help="Similitud por embeddings (all-MiniLM-L6-v2)")

    ai_prob = result.get("ai_probability")
    ai_label = result.get("ai_label")
    if ai_prob is not None:
        st.markdown("---")
        st.markdown(f"#### Detección de IA")
        ai_pct = ai_prob * 100
        if ai_prob > 0.6:
            st.error(f"**{ai_pct:.0f}%** — {ai_label}")
        else:
            st.success(f"**{ai_pct:.0f}%** — {ai_label}")
        st.progress(min(ai_prob, 1.0))
        ai_details = result.get("deteccion_ia", {})
        if ai_details:
            with st.expander("Ver métricas detalladas"):
                cols = st.columns(3)
                cols[0].metric("Perplejidad", ai_details.get("perplexity", "-"))
                cols[1].metric("Burstiness", ai_details.get("burstiness", "-"))
                cols[2].metric("Score IA", f"{ai_details.get('ai_probability', 0) * 100:.1f}%")

    fragmentos = result.get("fragmentos_similares", [])
    if fragmentos:
        st.markdown("---")
        st.markdown(f"#### Fragmentos similares (top {len(fragmentos)})")
        for i, frag in enumerate(fragmentos, 1):
            pct = frag["score"] * 100
            label = f"#{i} — {pct:.1f}% de similitud"
            with st.expander(f"{'🔴' if pct > 75 else '🟡' if pct > 45 else '🟢'} {label}"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Documento A**")
                    st.markdown(f"> {frag['frag_doc1']}")
                with col_b:
                    st.markdown("**Documento B**")
                    st.markdown(f"> {frag['frag_doc2']}")


st.set_page_config(
    page_title="Mini Turnitin",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key in ("result_2doc", "result_corpus", "result_ia", "last_mode"):
    st.session_state.setdefault(key, None)

with st.sidebar:
    st.markdown("### :material/tune: Configuración")
    st.markdown(f"- **Modelo:** `{cfg.model_name}`")
    st.markdown(f"- **Dimensiones:** {cfg.embedding_dim}")
    n_corpus = len(list(CORPUS_DIR.glob("*.pdf"))) if CORPUS_DIR.exists() else 0
    st.markdown(f"- **Corpus:** {n_corpus} documentos")
    st.markdown(f"- **Chunk size:** {cfg.chunk_max_words} palabras")
    st.markdown(f"- **Overlap:** {cfg.chunk_overlap_words} palabras")
    st.markdown(f"- **Índice FAISS:** `{cfg.faiss_index_type}`")
    st.markdown("---")
    st.markdown("**Mini Turnitin v2**")
    st.markdown("Curso PLN · Escuela de Ciencia de Datos")
    st.caption("Arquitectura híbrida: TF-IDF + MinHash + Embeddings + FAISS + Cross-Encoder")

st.title(":material/search_notes: Mini Turnitin — Detección de Similitud Textual")
st.caption("Sistema híbrido con chunking, FAISS, embeddings semánticos y detección de IA")

modo = st.radio(
    "Modo de análisis",
    [":material/description: Comparar dos documentos",
     ":material/library_books: Comparar contra corpus",
     ":material/psychology: Detectar IA"],
    horizontal=True,
)

# ── Modo 1: Comparar dos documentos ──
if ":material/description:" in modo:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Documento A")
        tipo_a = st.radio("Tipo", [":material/edit_note: Texto", ":material/attach_file: Archivo"],
                          key="tipo_a", horizontal=True)
        doc_a = ""
        if ":material/edit_note:" in tipo_a:
            doc_a = st.text_area("Pegar texto", height=250, key="text_a",
                                 placeholder="Pega aquí el contenido del documento A...")
        else:
            f_a = st.file_uploader("Subir archivo (PDF, DOCX, TXT)",
                                   type=["pdf", "docx", "txt"], key="pdf_a")
            if f_a:
                doc_a = extract_from_upload(f_a)
                st.success(f":material/check: Cargado: `{f_a.name}` ({len(doc_a):,} caracteres)")

    with col2:
        st.markdown("##### Documento B")
        tipo_b = st.radio("Tipo", [":material/edit_note: Texto", ":material/attach_file: Archivo"],
                          key="tipo_b", horizontal=True)
        doc_b = ""
        if ":material/edit_note:" in tipo_b:
            doc_b = st.text_area("Pegar texto", height=250, key="text_b",
                                 placeholder="Pega aquí el contenido del documento B...")
        else:
            f_b = st.file_uploader("Subir archivo (PDF, DOCX, TXT)",
                                   type=["pdf", "docx", "txt"], key="pdf_b")
            if f_b:
                doc_b = extract_from_upload(f_b)
                st.success(f":material/check: Cargado: `{f_b.name}` ({len(doc_b):,} caracteres)")

    if st.button(":material/analytics: Analizar similitud", type="primary",
                 use_container_width=True, disabled=not (doc_a and doc_b)):
        with st.spinner("Procesando... (chunking → embeddings → FAISS → reranking)"):
            st.session_state.result_2doc = analyze(doc_a, doc_b)
        st.session_state.last_mode = "2doc"

    if st.session_state.last_mode == "2doc" and st.session_state.result_2doc:
        st.divider()
        mostrar_resultado(st.session_state.result_2doc)

# ── Modo 2: Comparar contra corpus ──
elif ":material/library_books:" in modo:
    st.markdown("##### Documento a analizar")
    col_inp, _ = st.columns([2, 1])
    with col_inp:
        tipo = st.radio("Tipo", [":material/edit_note: Texto", ":material/attach_file: Archivo"],
                        horizontal=True, key="corpus_tipo")
        doc = ""
        if ":material/edit_note:" in tipo:
            doc = st.text_area("Pegar texto", height=200, key="corpus_text",
                               placeholder="Pega aquí el texto a comparar contra el corpus...")
        else:
            f = st.file_uploader("Subir archivo (PDF, DOCX, TXT)",
                                 type=["pdf", "docx", "txt"], key="corpus_pdf")
            if f:
                doc = extract_from_upload(f)
                st.success(f":material/check: Cargado: `{f.name}` ({len(doc):,} caracteres)")

    if st.button(":material/analytics: Analizar contra corpus", type="primary",
                 use_container_width=True, disabled=not doc):
        if not CORPUS_DIR.exists():
            st.error(f"El directorio del corpus no existe: `{CORPUS_DIR}`")
            st.stop()
        with st.spinner(f"Indexando {n_corpus} documentos con FAISS..."):
            st.session_state.result_corpus = analyze_corpus(doc, CORPUS_DIR)
        st.session_state.last_mode = "corpus"

    results = st.session_state.result_corpus
    if st.session_state.last_mode == "corpus" and results:
        st.divider()
        st.markdown(f"#### Resultados del corpus (ordenados por similitud)")
        for r in results[:10]:
            pct = r["score_final"] * 100
            badge = nivel_badge(r["nivel"])
            doc_name = r["documento"]
            ruta = r.get("ruta_archivo", "")
            with st.expander(
                f"{'🔴' if pct > 75 else '🟡' if pct > 45 else '🟢'} "
                f"`{doc_name[:50]}` — {pct:.1f}% {badge}"
            ):
                if ruta and Path(ruta).exists():
                    with open(ruta, "rb") as fh:
                        st.download_button(
                            label=f":material/download: Descargar {Path(ruta).name}",
                            data=fh,
                            file_name=Path(ruta).name,
                            mime="application/pdf" if ruta.endswith(".pdf") else None,
                            use_container_width=True,
                        )
                elif ruta:
                    st.markdown(f":material/folder_open: **Ruta:** `{ruta}`")
                mostrar_resultado(r, titulo=None)

# ── Modo 3: Detectar IA ──
else:
    st.markdown("##### Detectar si un texto fue generado por IA")
    st.caption("Basado en perplejidad (GPT-2) + burstiness de oraciones")
    col_inp, _ = st.columns([2, 1])
    with col_inp:
        tipo = st.radio("Tipo", [":material/edit_note: Texto", ":material/attach_file: Archivo"],
                        horizontal=True, key="ia_tipo")
        doc = ""
        if ":material/edit_note:" in tipo:
            doc = st.text_area("Pegar texto", height=300, key="ia_text",
                               placeholder="Pega aquí el texto a analizar...")
        else:
            f = st.file_uploader("Subir archivo (PDF, DOCX, TXT)",
                                 type=["pdf", "docx", "txt"], key="ia_pdf")
            if f:
                doc = extract_from_upload(f)
                st.success(f":material/check: Cargado: `{f.name}` ({len(doc):,} caracteres)")

    if st.button(":material/psychology: Detectar IA", type="primary",
                 use_container_width=True, disabled=not doc):
        with st.spinner("Analizando con GPT-2..."):
            st.session_state.result_ia = ai_probability(doc)
        st.session_state.last_mode = "ia"

    result = st.session_state.result_ia
    if st.session_state.last_mode == "ia" and result:
        st.divider()
        pct = result["ai_probability"] * 100
        label = result["label"]

        if result["ai_probability"] > 0.6:
            st.error(f"### {pct:.0f}% — {label}")
        else:
            st.success(f"### {pct:.0f}% — {label}")
        st.progress(min(result["ai_probability"], 1.0))

        st.markdown("#### Métricas detalladas")
        cols = st.columns(3)
        cols[0].metric("Perplejidad", result["perplexity"],
                       help="Menor perplejidad → texto más predecible (posible IA)")
        cols[1].metric("Burstiness", result["burstiness"],
                       help="Menor burstiness → poca variación en largo de oraciones (posible IA)")
        cols[2].metric("Score IA", f"{result['ai_probability'] * 100:.1f}%",
                       help="Combinación ponderada de perplejidad y burstiness")
        with st.expander("Interpretación"):
            st.markdown("""
            - **Perplejidad baja** (< 80): el texto es muy predecible, patrón típico de IA
            - **Burstiness baja** (< 0.5): todas las oraciones tienen largo similar, patrón típico de IA
            - **Humano**: suele tener perplejidad más alta y burstiness más variable
            """)
