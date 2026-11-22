# -*- coding: utf-8 -*-
"""
Proyección de un embedding nuevo (384-d) al espacio refinado (UMAP) en el que
se construyó el grafo.

Camino principal  : reproducir el UMAP original (mismos parámetros, misma muestra,
                    misma semilla) y verificar que reproduce X_refined.
Camino alternativo: si no se reproduce, aproximar la proyección con los vecinos más
                    cercanos del corpus (ProyectorVecinos). Es una aproximación y debe
                    declararse como tal.
"""
import os

import numpy as np


def _nuevo_umap(n_components, random_state):
    import umap
    return umap.UMAP(n_components=n_components, n_neighbors=5, metric="cosine",
                     low_memory=True, n_jobs=1, random_state=random_state)


def indices_muestra_umap(n, n_components=42, muestra=0.4, random_state=42):
    """Mismos índices (y mismo orden) que usó reduce_embeddings para el fit."""
    rng = np.random.default_rng(random_state)
    n_sample = max(int(n * muestra), n_components + 2)
    return rng.choice(n, size=n_sample, replace=False)


def ajustar_reducer(emb_original, n_components=42, muestra=0.4, random_state=42):
    """Igual que reduce_embeddings (notebook 1) pero DEVOLVIENDO el reducer.
    emb_original: embeddings originales (n x 384) de TODAS las filas, alineados con el dataset."""
    reducer = _nuevo_umap(n_components, random_state)
    idx = indices_muestra_umap(emb_original.shape[0], n_components, muestra, random_state)
    reducer.fit(emb_original[idx])
    return reducer


def ajustar_reducer_con_muestra(emb_muestra, n_components=42, random_state=42):
    """emb_muestra = embeddings originales de df.iloc[indices_muestra_umap(...)], EN ESE ORDEN."""
    reducer = _nuevo_umap(n_components, random_state)
    reducer.fit(emb_muestra)
    return reducer


def verificar_reducer(reducer, emb_filas, R_filas):
    """Compara reducer.transform(emb_filas) con las filas correspondientes de X_refined."""
    Z = reducer.transform(np.asarray(emb_filas, dtype=np.float32))
    R = np.asarray(R_filas, dtype=np.float32)
    err_rel = float(np.linalg.norm(Z - R) / np.linalg.norm(R))
    cos = float(((Z * R).sum(1) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(R, axis=1) + 1e-12)).mean())
    res = {"error_relativo": err_rel, "coseno_medio": cos, "max_abs": float(np.abs(Z - R).max()),
           "reproducido": err_rel < 1e-3}
    print(res)
    return res


def guardar_proyeccion_umap(reducer, out_dir):
    import joblib
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(reducer, os.path.join(out_dir, "umap_reducer.joblib"))
    return {"tipo": "umap", "archivo": "umap_reducer.joblib"}


class ProyectorVecinos:
    """
    Aproximación: proyecta e como el promedio ponderado de las coordenadas refinadas de sus
    k vecinos más cercanos (coseno) en el espacio original. NO equivale a UMAP.transform.
    Con excluir_identicos=True se ignoran vecinos con similitud > 0.9999 (útil para validar
    con comentarios que ya están en el corpus).
    """
    def __init__(self, emb_original, X_refined, k=5, excluir_identicos=False):
        self.E, self.R, self.k = emb_original, X_refined, k
        self.excluir_identicos = excluir_identicos

    def _sims(self, e, bloque=200_000):
        out = np.empty(len(self.E), dtype=np.float32)
        for i in range(0, len(self.E), bloque):
            blk = np.asarray(self.E[i:i + bloque], dtype=np.float32)
            out[i:i + len(blk)] = blk @ e / np.maximum(np.linalg.norm(blk, axis=1), 1e-12)
        return out

    def __call__(self, e):
        e = np.asarray(e, dtype=np.float32)
        e = e / max(np.linalg.norm(e), 1e-12)
        s = self._sims(e)
        if self.excluir_identicos:
            s[s > 0.9999] = -1.0
        top = np.argpartition(-s, self.k)[:self.k]
        w = 1.0 / (1e-6 + (1.0 - np.clip(s[top], -1, 1)))
        return (np.asarray(self.R[top], dtype=np.float32) * w[:, None]).sum(0) / w.sum()


def cargar_proyector(meta_proy, artefactos_dir):
    """Devuelve un callable 384-d -> espacio refinado, según lo registrado en comunidades.json."""
    if not meta_proy:
        return lambda e: e
    if meta_proy["tipo"] == "umap":
        import joblib
        reducer = joblib.load(os.path.join(artefactos_dir, meta_proy["archivo"]))
        return lambda e: reducer.transform(np.asarray(e, dtype=np.float32)[None])[0].astype(np.float32)
    if meta_proy["tipo"] == "knn":
        def ruta(p):
            return p if os.path.isabs(p) else os.path.join(artefactos_dir, p)
        E = np.load(ruta(meta_proy["embeddings_originales"]), mmap_mode="r")
        R = np.load(ruta(meta_proy["embeddings_refinados"]))
        return ProyectorVecinos(E, R, k=meta_proy.get("k", 5))
    raise ValueError(f"Tipo de proyección desconocido: {meta_proy['tipo']}")


def embeber(textos, nombre_modelo, device=None, batch_size=128, modelo=None):
    """Embeddings normalizados (como en el pipeline original: max_seq_length=512)."""
    if modelo is None:
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer(nombre_modelo, device=device)
        modelo.max_seq_length = 512
    return modelo.encode(list(textos), batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True)


def evaluar_modelo_embedding(nombre_modelo, textos_ref, R_ref, textos_test, R_test,
                             k=5, device=None):
    """
    Diagnóstico SIN embeddings originales: proyecta filas de prueba (excluidas de la referencia)
    por vecinos y compara con su coordenada refinada real (coseno de vectores centrados).
    El modelo que generó el espacio refinado obtiene un valor claramente mayor.
    """
    E_ref = embeber(textos_ref, nombre_modelo, device=device)
    E_test = embeber(textos_test, nombre_modelo, device=device)
    proy = ProyectorVecinos(E_ref, np.asarray(R_ref, dtype=np.float32), k=k)
    P = np.stack([proy(e) for e in E_test])
    mu = np.asarray(R_ref, dtype=np.float32).mean(0)
    A, B = P - mu, np.asarray(R_test, dtype=np.float32) - mu
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    return float(cos.mean())
