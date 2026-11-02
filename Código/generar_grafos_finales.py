# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DE GRAFOS DEFINITIVOS Y ASIGNACIONES COMUNITARIAS (SNA)
Subreddits: Anxiety, SuicideWatch, depression
Archivos: RMHD_ultimo.parquet + embeddings.npy
================================================================================
"""

import sys
import time
import os
import gc
import json
import numpy as np
import pandas as pd
import faiss
import igraph as ig
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PARQUET_FILE = "../RMHD_ultimo.parquet" if os.path.exists("../RMHD_ultimo.parquet") else "RMHD_ultimo.parquet"
EMB_FILE = "../embeddings.npy" if os.path.exists("../embeddings.npy") else "embeddings.npy"

FACTOR_COLS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
MONTH_TO_Q = {
    'jan': 'Q1', 'feb': 'Q1', 'mar': 'Q1',
    'apr': 'Q2', 'may': 'Q2', 'jun': 'Q2',
    'jul': 'Q3', 'aug': 'Q3', 'sep': 'Q3',
    'oct': 'Q4', 'nov': 'Q4', 'dec': 'Q4'
}

OPT_CONFIGS = {
    "Anxiety":      {"k": 30, "threshold": 0.95, "resolution": 0.10, "target_k": 10},
    "SuicideWatch": {"k": 30, "threshold": 0.95, "resolution": 0.10, "target_k": 10},
    "depression":   {"k": 30, "threshold": 0.95, "resolution": 0.10, "target_k": 12}
}

CUSTOM_STOPWORDS = [
    "like", "feel", "just", "want", "know", "really", "get",
    "feeling", "felt", "feels", "going", "got", "go", "make",
    "think", "thing", "things", "people", "would", "could", "one",
    "even", "time", "day", "much", "im", "ive", "dont", "cant",
    "someone", "also", "well", "see", "back", "take", "way"
]

def extract_top_words_ctfidf(texts, memberships, top_k=5):
    """Extrae palabras clave representativas por macrocomunidad mediante c-TF-IDF."""
    unique_comms = sorted([c for c in set(memberships) if c != -1])
    comm_docs = {c: [] for c in unique_comms}
    for t, m in zip(texts, memberships):
        if m != -1:
            comm_docs[m].append(str(t or ""))
            
    joined_docs = [" ".join(comm_docs[c]) for c in unique_comms]
    
    vec = CountVectorizer(stop_words=CUSTOM_STOPWORDS, min_df=3, max_df=0.6, max_features=12000)
    try:
        X_counts = vec.fit_transform(joined_docs)
        words = np.array(vec.get_feature_names_out())
        tf = X_counts.toarray()
        doc_lens = tf.sum(axis=1, keepdims=True)
        doc_lens[doc_lens == 0] = 1
        tf_norm = tf / doc_lens
        global_tf = tf.sum(axis=0)
        avg_len = doc_lens.mean()
        idf = np.log(1 + (avg_len / (global_tf + 1e-6)))
        ctfidf = tf_norm * idf
        
        topics = {}
        for i, c in enumerate(unique_comms):
            top_idx = np.argsort(ctfidf[i])[::-1][:top_k]
            # Limpiar caracteres incompatibles
            clean_words = [str(w).encode('ascii', errors='ignore').decode('ascii') for w in words[top_idx]]
            clean_words = [w for w in clean_words if len(w) > 2]
            topics[c] = clean_words if clean_words else [f"topico_{c}"]
        return topics
    except Exception as e:
        print(f"    [!] Error en c-TF-IDF: {e}. Usando palabras por defecto.")
        return {c: [f"Comunidad_{c}"] for c in unique_comms}

def build_graph_from_neighbors(I_top, D_top, n_nodes, k_target, threshold):
    k_eff = min(k_target, I_top.shape[1] - 1)
    I_sub = I_top[:, 1:k_eff+1]
    D_sub = D_top[:, 1:k_eff+1]
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), k_eff)
    cols = I_sub.astype(np.int64).ravel()
    vals = D_sub.ravel()
    mask = (vals >= threshold) & (cols != rows) & (cols >= 0)
    r, c, w = rows[mask], cols[mask], vals[mask]
    if len(r) == 0: return ig.Graph(n=n_nodes)
    lo, hi = np.minimum(r, c), np.maximum(r, c)
    keys = lo * n_nodes + hi
    order = np.argsort(keys)
    keys, w = keys[order], w[order]
    mask_diff = np.empty(len(keys), dtype=bool)
    mask_diff[0] = True
    mask_diff[1:] = (keys[1:] != keys[:-1])
    start_idx = np.flatnonzero(mask_diff)
    uniq_keys = keys[start_idx]
    uniq_w = np.maximum.reduceat(w, start_idx)
    lo_u = (uniq_keys // n_nodes).astype(np.int32)
    hi_u = (uniq_keys % n_nodes).astype(np.int32)
    edges = np.column_stack((lo_u, hi_u))
    return ig.Graph(n=n_nodes, edges=edges, edge_attrs={"weight": uniq_w})

def generar_grafos_completos():
    print("="*80)
    print("CONSTRUCCIÓN Y EXPORTACIÓN DE GRAFOS DEFINITIVOS SIN MST (100% DATOS)")
    print("="*80)
    
    df_full = pd.read_parquet(PARQUET_FILE)
    df_full["period"] = df_full["year"].astype(str) + df_full["month"].map(MONTH_TO_Q)
    emb_full = np.load(EMB_FILE, mmap_mode="r")
    
    for sub, cfg in OPT_CONFIGS.items():
        pkl_file = f"Grafo_{sub}_best.pkl"
        topics_file = f"topics_{sub}_best.json"
        asignaciones_file = f"asignaciones_comunidades_{sub}.parquet"
        
        if os.path.exists(pkl_file) and os.path.exists(topics_file) and os.path.exists(asignaciones_file):
            print(f"\n[+] r/{sub}: Archivos ya generados ('{pkl_file}', '{topics_file}'). Omitiendo recálculo.")
            continue
            
        print(f"\n[*] PROCESANDO r/{sub}...")
        sub_indices = np.flatnonzero(df_full["subreddit"] == sub)
        n_sub = len(sub_indices)
        print(f"    Total de comentarios: {n_sub:,}")
        
        df_sub = df_full.iloc[sub_indices].copy().reset_index(drop=True)
        df_sub["orig_dataset_index"] = sub_indices
        
        X_sub = np.array(emb_full[sub_indices], dtype=np.float32)
        X_sub -= X_sub.mean(axis=0)
        faiss.normalize_L2(X_sub)
        dim = X_sub.shape[1]
        
        print(f"    Indexando con FAISS IndexFlatIP (k={cfg['k']})...")
        t0 = time.time()
        idx_flat = faiss.IndexFlatIP(dim)
        idx_flat.add(X_sub)
        batch_sz = 35000
        D_list, I_list = [], []
        for s in range(0, n_sub, batch_sz):
            e = min(s + batch_sz, n_sub)
            db, ib = idx_flat.search(X_sub[s:e], cfg["k"] + 1)
            D_list.append(db)
            I_list.append(ib)
        del idx_flat; gc.collect()
        D_f = np.vstack(D_list)
        I_f = np.vstack(I_list)
        del D_list, I_list; gc.collect()
        print(f"    Búsqueda FAISS completada en {time.time()-t0:.1f}s")
        
        print("    Construyendo grafo igraph...")
        G_base = build_graph_from_neighbors(I_f, D_f, n_sub, k_target=cfg["k"], threshold=cfg["threshold"])
        del D_f, I_f; gc.collect()
        print(f"    Grafo base: {G_base.vcount():,} nodos, {G_base.ecount():,} aristas.")
        
        # Detección de comunidades base con Leiden
        print(f"    Detectando comunidades con Leiden (res={cfg['resolution']})...")
        cl_leiden = G_base.community_leiden(
            objective_function="modularity",
            weights=G_base.es["weight"],
            resolution=cfg["resolution"],
            n_iterations=2
        )
        base_membership = np.array(cl_leiden.membership)
        sizes_base = cl_leiden.sizes()
        subst_comms = [i for i, s in enumerate(sizes_base) if s >= 500]
        cov_pct = (sum(sizes_base[i] for i in subst_comms) / n_sub) * 100.0
        print(f"    Comunidades base sustanciales: {len(subst_comms)} ({cov_pct:.2f}% de cobertura)")
        
        # Agrupación jerárquica aglomerativa hacia target_k comunidades
        target_k = cfg["target_k"]
        print(f"    Agrupando semánticamente en K={target_k} macrocomunidades...")
        centroids_base = np.array([X_sub[base_membership == c].mean(axis=0) for c in subst_comms])
        centroids_base /= (np.linalg.norm(centroids_base, axis=1, keepdims=True) + 1e-8)
        
        agg = AgglomerativeClustering(n_clusters=target_k, metric="cosine", linkage="average")
        macro_labels = agg.fit_predict(centroids_base)
        c_to_macro = {subst_comms[i]: macro_labels[i] for i in range(len(subst_comms))}
        
        # Asignación final con trazabilidad
        df_sub["orig_leiden_comm"] = base_membership
        df_sub["final_macro_comm"] = [c_to_macro.get(m, -1) for m in base_membership]
        df_sub["is_residual"] = (df_sub["final_macro_comm"] == -1)
        
        # Subgrafo podado sin comunidades residuales para análisis y visualización
        valid_nodes = np.flatnonzero(df_sub["final_macro_comm"] != -1)
        print(f"    Subgrafo filtrado: {len(valid_nodes):,} nodos válidos (residuo excluido: {n_sub - len(valid_nodes):,} comentarios, {(100-cov_pct):.2f}%).")
        
        G = G_base.subgraph(valid_nodes)
        del G_base; gc.collect()
        
        df_valid = df_sub.iloc[valid_nodes].copy().reset_index(drop=True)
        
        # Asignar atributos al grafo
        G.vs["orig_index"] = df_valid["orig_dataset_index"].tolist()
        G.vs["community"] = df_valid["final_macro_comm"].tolist()
        G.vs["orig_leiden"] = df_valid["orig_leiden_comm"].tolist()
        G.vs["period"] = df_valid["period"].tolist()
        G.vs["year"] = df_valid["year"].tolist()
        G.vs["month"] = df_valid["month"].tolist()
        G.vs["factor"] = df_valid["emocion"].tolist()
        G.vs["comentario_completo"] = df_valid["text"].tolist()
        for fcol in FACTOR_COLS:
            G.vs[fcol] = df_valid[fcol].tolist()
            
        # Extraer tópicos c-TF-IDF
        print("    Extrayendo tópicos semánticos con c-TF-IDF...")
        topics_dict = extract_top_words_ctfidf(df_valid["text_clean"].tolist(), df_valid["final_macro_comm"].tolist(), top_k=5)
        for c, words in sorted(topics_dict.items()):
            w_str = ", ".join(words)
            print(f"      Macro C{c} ({sum(df_valid['final_macro_comm'] == c):,} nodos): {w_str}")
            
        G.vs["topic"] = [", ".join(topics_dict.get(c, ["N/A"])) for c in G.vs["community"]]
        
        # Guardar Grafo Pickle y Topics JSON
        G.write_pickle(pkl_file)
        with open(topics_file, "w", encoding="utf-8") as tf:
            json.dump({str(k): v for k, v in topics_dict.items()}, tf, indent=2, ensure_ascii=False)
            
        # Guardar tabla de asignaciones para reproducibilidad
        df_sub[["orig_dataset_index", "orig_leiden_comm", "final_macro_comm", "is_residual", "period", "emocion"]].to_parquet(asignaciones_file, index=False)
        
        print(f"    [OK] Guardados: '{pkl_file}', '{topics_file}' y '{asignaciones_file}'.")
        del G, df_sub, df_valid, X_sub; gc.collect()

    print("\n" + "="*80)
    print("TODOS LOS GRAFOS DEFINITIVOS HAN SIDO GENERADOS Y ALMACENADOS CON ÉXITO")
    print("="*80)

if __name__ == "__main__":
    generar_grafos_completos()
