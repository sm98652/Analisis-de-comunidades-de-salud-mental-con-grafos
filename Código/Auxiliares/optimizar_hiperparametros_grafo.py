# -*- coding: utf-8 -*-
"""
================================================================================
OPTIMIZACIÓN DE HIPERPARÁMETROS Y EVALUACIÓN METODOLÓGICA DE REDES (SNA)
Subreddits: Anxiety, SuicideWatch, depression
Datos: RMHD_ultimo.parquet + embeddings.npy
================================================================================
Estrategia:
  1. Exploración en grilla sobre muestra representativa estratificada (N=50,000 por sub):
     - k in [20, 30, 50]
     - epsilon (threshold) in [0.92, 0.95, 0.97]
     - Algoritmo: Louvain vs Leiden (Garantía de conectividad y velocidad)
     - Resolución: [0.05, 0.08, 0.10, 0.15, 0.20]
     - Macro-clustering aglomerativo K in [8, 10, 12, 14]
  2. Validación final sobre el conjunto completo de comentarios para la configuración óptima.
Salidas:
  - Reporte_Optimizacion_Hiperparametros_SNA.xlsx
  - reporte_optimizacion_hiperparametros_sna.csv
================================================================================
"""

import time
import os
import gc
import psutil
import numpy as np
import pandas as pd
import faiss
import igraph as ig
from sklearn.cluster import AgglomerativeClustering

PARQUET_FILE = "../RMHD_ultimo.parquet" if os.path.exists("../RMHD_ultimo.parquet") else "RMHD_ultimo.parquet"
EMB_FILE = "../embeddings.npy" if os.path.exists("../embeddings.npy") else "embeddings.npy"
OUTPUT_EXCEL = "Reporte_Optimizacion_Hiperparametros_SNA.xlsx"
OUTPUT_CSV = "reporte_optimizacion_hiperparametros_sna.csv"

SUBREDDITS = ["Anxiety", "SuicideWatch", "depression"]
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def get_ram_gb():
    return psutil.virtual_memory().available / (1024**3)

def compute_intra_coherence(X_norm, membership):
    """Calcula la similitud coseno promedio al centroide intracomunidad."""
    coherences = []
    unique_comms = [c for c in set(membership) if c != -1]
    for c in unique_comms:
        idx = np.flatnonzero(membership == c)
        if len(idx) < 2: continue
        centroid = X_norm[idx].mean(axis=0, keepdims=True)
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        sims = np.dot(X_norm[idx], c_norm.T).ravel()
        coherences.append(float(sims.mean()))
    return float(np.mean(coherences)) if coherences else 0.0

def compute_inter_separation(centroids):
    """Calcula la distancia coseno promedio (1 - cos) entre centroides."""
    if len(centroids) < 2: return 0.0
    sim_matrix = np.dot(centroids, centroids.T)
    n_c = len(centroids)
    triu_idx = np.triu_indices(n_c, k=1)
    return float(np.mean(1.0 - sim_matrix[triu_idx]))

def build_graph_from_neighbors(I_top, D_top, n_nodes, k_target, threshold):
    """Construye grafo igraph a partir de vecinos."""
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

def run_grid_search_and_validation():
    print("="*80)
    print("EJECUTANDO BÚSQUEDA EN GRILLA METODOLÓGICA Y COMPARATIVA LOUVAIN VS LEIDEN")
    print(f"RAM Disponible: {get_ram_gb():.2f} GB")
    print("="*80)
    
    df_meta = pd.read_parquet(PARQUET_FILE, columns=["subreddit"])
    emb_full = np.load(EMB_FILE, mmap_mode="r")
    
    results = []
    
    # --------------------------------------------------------------------------
    # FASE 1: BÚSQUEDA EN GRILLA SOBRE MUESTRA REPRESENTATIVA (N=50,000)
    # --------------------------------------------------------------------------
    print("\n>>> FASE 1: Exploración en Grilla sobre Muestra Representativa (N=50,000 por Subreddit)...")
    N_SAMPLE = 50000
    
    for sub in SUBREDDITS:
        print(f"\n[*] Subreddit: r/{sub}")
        sub_all_idx = np.flatnonzero(df_meta["subreddit"] == sub)
        n_total = len(sub_all_idx)
        
        # Muestra determinista representativa con semilla 42
        rng = np.random.default_rng(RANDOM_SEED)
        sample_local = rng.choice(n_total, size=min(N_SAMPLE, n_total), replace=False)
        sample_local.sort()
        sample_idx = sub_all_idx[sample_local]
        n_sample = len(sample_idx)
        
        X_sample = np.array(emb_full[sample_idx], dtype=np.float32)
        X_sample -= X_sample.mean(axis=0)
        faiss.normalize_L2(X_sample)
        dim = X_sample.shape[1]
        
        # Búsqueda FAISS top 50
        t0 = time.time()
        idx_flat = faiss.IndexFlatIP(dim)
        idx_flat.add(X_sample)
        D_top, I_top = idx_flat.search(X_sample, 51)
        del idx_flat; gc.collect()
        print(f"    FAISS búsqueda completada en {time.time()-t0:.2f}s")
        
        # Evaluar configuraciones de grafo y algoritmos
        for k in [20, 30, 50]:
            for th in [0.92, 0.95, 0.97]:
                G = build_graph_from_neighbors(I_top, D_top, n_sample, k_target=k, threshold=th)
                n_edges = G.ecount()
                k_eff = np.mean(G.degree()) if G.vcount() > 0 else 0.0
                density = G.density() if G.vcount() > 0 else 0.0
                
                for res in [0.05, 0.08, 0.10, 0.15, 0.20]:
                    # 1. Leiden
                    t_lei0 = time.time()
                    cl_lei = G.community_leiden(
                        objective_function="modularity",
                        weights=G.es["weight"],
                        resolution=res,
                        n_iterations=2
                    )
                    t_lei = time.time() - t_lei0
                    mod_lei = G.modularity(cl_lei.membership, weights=G.es["weight"])
                    sizes_lei = cl_lei.sizes()
                    subst_lei = [i for i, s in enumerate(sizes_lei) if s >= 200]
                    cov_lei = (sum(sizes_lei[i] for i in subst_lei) / n_sample) * 100.0
                    
                    results.append({
                        "Fase": "Exploratoria (Muestra 50k)",
                        "Subreddit": sub,
                        "N_Muestra": n_sample,
                        "N_Total": n_total,
                        "k": k,
                        "Threshold": th,
                        "Aristas": n_edges,
                        "k_Efectivo": round(k_eff, 2),
                        "Densidad": f"{density:.5e}",
                        "Algoritmo": "Leiden (Modularity)",
                        "Resolucion": res,
                        "Modularidad": round(mod_lei, 4),
                        "Comunidades_Totales": len(sizes_lei),
                        "Comunidades_Sustanciales": len(subst_lei),
                        "Cobertura_Sustancial (%)": round(cov_lei, 2),
                        "Coherencia_Intra": round(compute_intra_coherence(X_sample, np.array(cl_lei.membership)), 4),
                        "Tiempo_s": round(t_lei, 3)
                    })
                    
                    # 2. Louvain (muestreo rápido en resoluciones clave)
                    if res in [0.05, 0.10, 0.20] and k == 30 and th == 0.95:
                        t_lou0 = time.time()
                        cl_lou = G.community_multilevel(weights=G.es["weight"], resolution=res)
                        t_lou = time.time() - t_lou0
                        mod_lou = G.modularity(cl_lou.membership, weights=G.es["weight"])
                        sizes_lou = cl_lou.sizes()
                        subst_lou = [i for i, s in enumerate(sizes_lou) if s >= 200]
                        cov_lou = (sum(sizes_lou[i] for i in subst_lou) / n_sample) * 100.0
                        
                        results.append({
                            "Fase": "Exploratoria (Muestra 50k)",
                            "Subreddit": sub,
                            "N_Muestra": n_sample,
                            "N_Total": n_total,
                            "k": k,
                            "Threshold": th,
                            "Aristas": n_edges,
                            "k_Efectivo": round(k_eff, 2),
                            "Densidad": f"{density:.5e}",
                            "Algoritmo": "Louvain",
                            "Resolucion": res,
                            "Modularidad": round(mod_lou, 4),
                            "Comunidades_Totales": len(sizes_lou),
                            "Comunidades_Sustanciales": len(subst_lou),
                            "Cobertura_Sustancial (%)": round(cov_lou, 2),
                            "Coherencia_Intra": round(compute_intra_coherence(X_sample, np.array(cl_lou.membership)), 4),
                            "Tiempo_s": round(t_lou, 3)
                        })
                    
                    # 3. Macro-Clustering Semántico a K in [8, 10, 12, 14]
                    if k == 30 and th == 0.95 and res == 0.10 and len(subst_lei) >= 8:
                        mem_arr = np.array(cl_lei.membership)
                        subst_for_macro = subst_lei if len(subst_lei) >= 14 else [
                            i for i, _ in sorted(enumerate(sizes_lei), key=lambda x: x[1], reverse=True)[:16]
                        ]
                        centroids = np.array([X_sample[mem_arr == c].mean(axis=0) for c in subst_for_macro])
                        centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
                        
                        for target_k in [8, 10, 12, 14]:
                            if target_k > len(subst_for_macro): continue
                            agg = AgglomerativeClustering(n_clusters=target_k, metric="cosine", linkage="average")
                            macro_labels = agg.fit_predict(centroids)
                            c_to_macro = {subst_for_macro[i]: macro_labels[i] for i in range(len(subst_for_macro))}
                            macro_mem = np.array([c_to_macro.get(m, -1) for m in mem_arr])
                            
                            valid_idx = np.flatnonzero(macro_mem != -1)
                            sub_G = G.subgraph(valid_idx)
                            mod_macro = sub_G.modularity(macro_mem[valid_idx], weights=sub_G.es["weight"])
                            del sub_G; gc.collect()
                            
                            intra_c = compute_intra_coherence(X_sample, macro_mem)
                            macro_centroids = np.array([X_sample[macro_mem == mk].mean(axis=0) for mk in range(target_k)])
                            macro_centroids /= (np.linalg.norm(macro_centroids, axis=1, keepdims=True) + 1e-8)
                            inter_s = compute_inter_separation(macro_centroids)
                            
                            results.append({
                                "Fase": "Exploratoria (Macro 50k)",
                                "Subreddit": sub,
                                "N_Muestra": n_sample,
                                "N_Total": n_total,
                                "k": k,
                                "Threshold": th,
                                "Aristas": n_edges,
                                "k_Efectivo": round(k_eff, 2),
                                "Densidad": f"{density:.5e}",
                                "Algoritmo": f"Leiden + Macro-Clustering (K={target_k})",
                                "Resolucion": res,
                                "Modularidad": round(mod_macro, 4),
                                "Comunidades_Totales": target_k,
                                "Comunidades_Sustanciales": target_k,
                                "Cobertura_Sustancial (%)": round(cov_lei, 2),
                                "Coherencia_Intra": round(intra_c, 4),
                                "Separacion_Inter": round(inter_s, 4),
                                "Tiempo_s": round(t_lei, 3)
                            })
                del G; gc.collect()
        del D_top, I_top, X_sample; gc.collect()

    # --------------------------------------------------------------------------
    # FASE 2: VALIDACIÓN COMPLETA SOBRE EL 100% DE LOS DATOS FILTRADOS
    # (Configuración Óptima: k=30, threshold=0.95, Leiden res=0.10, Macro K=10)
    # --------------------------------------------------------------------------
    print("\n>>> FASE 2: Validación de la Configuración Seleccionada sobre el 100% de los Datos...")
    
    # Configuración óptima de K macro por subreddit
    OPT_K_MACRO = {
        "Anxiety": 10,
        "SuicideWatch": 10,
        "depression": 12
    }
    
    final_models = {}
    
    for sub in SUBREDDITS:
        print(f"\n[*] VALIDACIÓN COMPLETA r/{sub}:")
        sub_all_idx = np.flatnonzero(df_meta["subreddit"] == sub)
        n_full = len(sub_all_idx)
        print(f"    Cargando {n_full:,} vectores...")
        
        X_full = np.array(emb_full[sub_all_idx], dtype=np.float32)
        X_full -= X_full.mean(axis=0)
        faiss.normalize_L2(X_full)
        dim = X_full.shape[1]
        
        # FAISS search k=30 con batch
        t0 = time.time()
        idx_flat = faiss.IndexFlatIP(dim)
        idx_flat.add(X_full)
        batch_sz = 35000
        D_list, I_list = [], []
        for s in range(0, n_full, batch_sz):
            e = min(s + batch_sz, n_full)
            db, ib = idx_flat.search(X_full[s:e], 31)
            D_list.append(db)
            I_list.append(ib)
        del idx_flat; gc.collect()
        D_f = np.vstack(D_list)
        I_f = np.vstack(I_list)
        del D_list, I_list; gc.collect()
        t_faiss_f = time.time() - t0
        print(f"    FAISS completado en {t_faiss_f:.1f}s")
        
        # Grafo completo
        t_g0 = time.time()
        G_full = build_graph_from_neighbors(I_f, D_f, n_full, k_target=30, threshold=0.95)
        del D_f, I_f; gc.collect()
        t_graph_f = time.time() - t_g0
        n_edges_f = G_full.ecount()
        k_eff_f = np.mean(G_full.degree())
        print(f"    Grafo construido: {G_full.vcount():,} nodos, {n_edges_f:,} aristas en {t_graph_f:.1f}s")
        
        # Leiden res=0.10
        t_l0 = time.time()
        cl_full = G_full.community_leiden(
            objective_function="modularity",
            weights=G_full.es["weight"],
            resolution=0.10,
            n_iterations=2
        )
        t_lei_full = time.time() - t_l0
        mod_base_full = G_full.modularity(cl_full.membership, weights=G_full.es["weight"])
        sizes_full = cl_full.sizes()
        subst_full = [i for i, s in enumerate(sizes_full) if s >= 500]
        cov_full = (sum(sizes_full[i] for i in subst_full) / n_full) * 100.0
        print(f"    Leiden ejecutado en {t_lei_full:.1f}s: {len(sizes_full)} comun. totales, {len(subst_full)} sustanciales ({cov_full:.2f}% cob.)")
        
        # Macro-clustering semántico
        target_k = OPT_K_MACRO[sub]
        mem_full = np.array(cl_full.membership)
        centroids_base = np.array([X_full[mem_full == c].mean(axis=0) for c in subst_full])
        centroids_base /= (np.linalg.norm(centroids_base, axis=1, keepdims=True) + 1e-8)
        
        agg = AgglomerativeClustering(n_clusters=target_k, metric="cosine", linkage="average")
        macro_labels = agg.fit_predict(centroids_base)
        c_to_macro = {subst_full[i]: macro_labels[i] for i in range(len(subst_full))}
        macro_membership = np.array([c_to_macro.get(m, -1) for m in mem_full])
        
        valid_idx = np.flatnonzero(macro_membership != -1)
        sub_G = G_full.subgraph(valid_idx)
        mod_macro_full = sub_G.modularity(macro_membership[valid_idx], weights=sub_G.es["weight"])
        del sub_G; gc.collect()
        
        intra_full = compute_intra_coherence(X_full, macro_membership)
        macro_c = np.array([X_full[macro_membership == mk].mean(axis=0) for mk in range(target_k)])
        macro_c /= (np.linalg.norm(macro_c, axis=1, keepdims=True) + 1e-8)
        inter_full = compute_inter_separation(macro_c)
        
        # Registrar validación final
        results.append({
            "Fase": "★ Validación Definitiva (100% Datos)",
            "Subreddit": sub,
            "N_Muestra": n_full,
            "N_Total": n_full,
            "k": 30,
            "Threshold": 0.95,
            "Aristas": n_edges_f,
            "k_Efectivo": round(k_eff_f, 2),
            "Densidad": f"{G_full.density():.5e}",
            "Algoritmo": f"Leiden (res=0.10) + Macro-Clustering (K={target_k})",
            "Resolucion": 0.10,
            "Modularidad": round(mod_macro_full, 4),
            "Comunidades_Totales": target_k,
            "Comunidades_Sustanciales": target_k,
            "Cobertura_Sustancial (%)": round(cov_full, 2),
            "Coherencia_Intra": round(intra_full, 4),
            "Separacion_Inter": round(inter_full, 4),
            "Tiempo_s": round(t_lei_full, 2)
        })
        
        # Guardar en memoria para exportación de artefactos
        final_models[sub] = {
            "G": G_full,
            "sub_indices": sub_all_idx,
            "base_membership": mem_full,
            "macro_membership": macro_membership,
            "target_k": target_k,
            "cov_pct": cov_full,
            "mod_macro": mod_macro_full,
            "intra_coh": intra_full,
            "centroids": macro_c
        }
        del X_full; gc.collect()

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Todas_las_Evaluaciones", index=False)
        
        # Resumen comparativo Louvain vs Leiden
        comp_df = df_results[df_results["Algoritmo"].isin(["Louvain", "Leiden (Modularity)"])].groupby(
            ["Subreddit", "Algoritmo"]
        ).agg({
            "Modularidad": "mean",
            "Tiempo_s": "mean",
            "Cobertura_Sustancial (%)": "mean"
        }).reset_index()
        comp_df.to_excel(writer, sheet_name="Comparativa_Louvain_vs_Leiden", index=False)
        
        # Resumen de validación definitiva
        val_df = df_results[df_results["Fase"].str.contains("Validación Definitiva")]
        val_df.to_excel(writer, sheet_name="Ganadores_Finales_Definitivos", index=False)

    print("\n" + "="*80)
    print("REPORTE EXCEL Y CSV GENERADOS CON ÉXITO")
    print(f"Archivos: {OUTPUT_EXCEL} | {OUTPUT_CSV}")
    print("="*80)
    
    return final_models

if __name__ == "__main__":
    run_grid_search_and_validation()
