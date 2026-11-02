# -*- coding: utf-8 -*-
"""
================================================================================
MÓDULO DE ANÁLISIS DE REDES COMPLEJAS (SNA) Y VISUALIZACIÓN MULTIESCALA (V3.0)
Versión de Alta Claridad Visual (Menos Hover, Etiquetas en Lienzo y Metadatos Ricos)
Redes Semántico-Afectivas en Salud Mental (Reddit / RMHD)
Subreddits: Anxiety, SuicideWatch, depression
================================================================================
Mejoras clave en V3.0 respecto a V2.1:
  1. Pestaña 1 (Micro Orgánico):
     - Etiquetas de centroides directamente sobre cada clúster en el lienzo
       (C# + tópico corto + % volumen) para identificar grupos sin hacer hover.
     - Customdata enriquecido por nodo y catálogo de comentarios representativos
       por comunidad/trimestre para lectura fija en el panel lateral.
  2. Pestañas 2 y 3 (Macro Agregado y Evolución Temporal):
     - Nodos con etiquetas completas (C# + tópico + % volumen).
     - Filtros rápidos de claridad en el propio gráfico ("Todas las Aristas",
       "Top 15 Puentes Más Fuertes (ρ)" con etiquetas de ρ en las líneas, y
       "Top 10 Mayor Cambio |Δ%|").
     - Metadatos estructurados de aristas por periodo para alimentar la Tabla
       Ranking de Puentes lateral en tiempo real.
  3. Pestaña 4 (Matriz de Burbujas):
     - Comunidades ordenadas de mayor a menor volumen total para lectura inmediata.
     - Tres modos: % Dominancia Emocional, % Volumen del Trimestre y Vista
       Combinada (% Dominancia + % Volumen en la misma etiqueta).
  4. Pestaña 5 (Evolución de Volumen Longitudinal):
     - Etiquetas directas al final derecho de cada línea para evitar cruzar la
       vista con la leyenda.
     - Filtros rápidos Anti-Spaghetti: "Todas (Volumen)", "Top 5 Más Grandes",
       "% de Cambio Trimestral" y "Top 5 MayorVariación".
  5. Extractor de Resumen Analítico (`build_subreddit_summary_meta`):
     - Genera los KPIs globales/trimestrales, directorio de comunidades,
       distribución afectiva, ranking de puentes y muestras de comentarios para
       el panel inspector lateral del HTML.
================================================================================
"""

import os
import re
import html
import math
import numpy as np
import igraph as ig
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc
from collections import Counter, defaultdict

FACTOR_COLS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

EMOTION_PALETTE = {
    "disgust":  {"name": "Disgust (Disgusto)",  "short": "Disgusto", "color": "rgba(40, 201, 105, 0.85)", "hex": "#28C969"},
    "joy":      {"name": "Joy (Alegría)",       "short": "Alegría",  "color": "rgba(255, 215, 0, 0.85)",  "hex": "#FFD700"},
    "surprise": {"name": "Surprise (Sorpresa)", "short": "Sorpresa", "color": "rgba(255, 140, 0, 0.85)",  "hex": "#FF8C00"},
    "anger":    {"name": "Anger (Ira/Enfado)",  "short": "Ira",      "color": "rgba(220, 20, 60, 0.85)",  "hex": "#DC143C"},
    "sadness":  {"name": "Sadness (Tristeza)",  "short": "Tristeza", "color": "rgba(30, 144, 255, 0.85)", "hex": "#1E90FF"},
    "fear":     {"name": "Fear (Miedo)",        "short": "Miedo",    "color": "rgba(148, 0, 211, 0.85)",  "hex": "#9400D3"},
    "neutral":  {"name": "Neutral",             "short": "Neutral",  "color": "rgba(194, 194, 194, 0.85)", "hex": "#C2C2C2"}
}

COMMUNITY_PALETTE = (
    pc.qualitative.Plotly +
    pc.qualitative.D3 +
    pc.qualitative.Set2 +
    pc.qualitative.Bold +
    pc.qualitative.Dark24
)


def sort_periods_chronologically(period_list):
    """Ordena periodos trimestrales cronológicamente (e.g. 2019Q1 ... 2022Q3)."""
    def key_fn(p):
        try:
            s = str(p).strip().upper()
            if "Q" in s:
                parts = s.split("Q")
                return (int(parts[0]), int(parts[1]))
            return (int(s[:4]), 0)
        except Exception:
            return (9999, 9999)
    return sorted(list(set(period_list)), key=key_fn)


def get_community_color(comm_id):
    """Color determinista por comunidad."""
    try:
        idx = int(comm_id)
    except (ValueError, TypeError):
        idx = hash(str(comm_id))
    return COMMUNITY_PALETTE[abs(idx) % len(COMMUNITY_PALETTE)]


def ensure_graph_attributes(G, subreddit_name=None):
    """Garantiza la presencia de atributos requeridos en los vértices del grafo."""
    if "factor" not in G.vertex_attributes() and all(f in G.vertex_attributes() for f in FACTOR_COLS):
        mat = np.column_stack([G.vs[f] for f in FACTOR_COLS])
        dom_idx = np.argmax(mat, axis=1)
        G.vs["factor"] = [FACTOR_COLS[i] for i in dom_idx]

    if "comentario_completo" not in G.vertex_attributes():
        if "text" in G.vertex_attributes():
            G.vs["comentario_completo"] = G.vs["text"]
        elif "comment" in G.vertex_attributes():
            G.vs["comentario_completo"] = G.vs["comment"]


def ensure_community_topics(G, community_topics=None):
    """Garantiza tópicos válidos por comunidad."""
    topics = {} if community_topics is None else dict(community_topics)
    if "topic" in G.vertex_attributes():
        for v in G.vs:
            c = v["community"]
            if c != -1 and c not in topics:
                t_val = v["topic"]
                if t_val:
                    topics[c] = t_val
    return topics


def get_community_topic(community_topics, comm, max_words=5):
    """Recupera la descripción del tópico formateada para etiquetas."""
    if not community_topics:
        return "N/A"
    candidates = [comm, str(comm), f"C{comm}"]
    val = None
    for k in candidates:
        if k in community_topics:
            val = community_topics[k]
            break
    if val is None or str(val).lower() in ["sin tópico", "n/a"]:
        return "N/A"
    if isinstance(val, list):
        clean = [str(w).strip(" ,;.") for w in val if str(w).strip(" ,;.")]
        return ", ".join(clean[:max_words]) if clean else "N/A"
    elif isinstance(val, str):
        words = [w.strip(" ,;.") for w in val.replace(",", " ").split() if w.strip(" ,;.")]
        return ", ".join(words[:max_words]) if words else "N/A"
    return str(val)


# ==============================================================================
# 1. PESTAÑA 1: VISTA MICRO ORGÁNICA CON ETIQUETAS DE CLÚSTER EN EL LIENZO
# ==============================================================================

def sample_representative_subgraph(G, community_topics, max_nodes_total=1400, min_nodes_per_comm=3, random_seed=42):
    """
    Muestreo estratificado representativo enfocado en ALINEACIÓN TEMÁTICA y DIVERSIDAD AFECTIVA.
    """
    rng = np.random.default_rng(random_seed)
    all_comms = [c for c in set(G.vs["community"]) if c != -1]
    if not all_comms:
        return G.subgraph([])

    comm_indices = {c: [v.index for v in G.vs if v["community"] == c] for c in all_comms}
    comm_counts = {c: len(comm_indices[c]) for c in all_comms}
    total_valid = sum(comm_counts.values())

    if total_valid <= max_nodes_total:
        all_sampled = [idx for indices in comm_indices.values() for idx in indices]
        G_sub = G.subgraph(all_sampled)
        G_sub.vs["orig_index"] = all_sampled
        return G_sub

    quotas = {c: (comm_counts[c] / total_valid) * max_nodes_total for c in all_comms}
    allocations = {}
    for c in all_comms:
        size = comm_counts[c]
        min_c = min(size, min_nodes_per_comm)
        allocations[c] = min(size, max(min_c, int(math.floor(quotas[c]))))

    remainder = max_nodes_total - sum(allocations.values())
    if remainder > 0:
        fractions = sorted(
            [(quotas[c] - math.floor(quotas[c]), c) for c in all_comms if allocations[c] < comm_counts[c]],
            key=lambda x: x[0], reverse=True
        )
        for _, c in fractions[:remainder]:
            allocations[c] += 1
    elif remainder < 0:
        excess = -remainder
        reducible = sorted(
            [(allocations[c], c) for c in all_comms if allocations[c] > min_nodes_per_comm],
            key=lambda x: x[0], reverse=True
        )
        for _, c in reducible[:excess]:
            allocations[c] -= 1

    sampled_nodes = []
    has_factors = all(f in G.vertex_attributes() for f in FACTOR_COLS)
    degrees = G.degree()

    for comm in all_comms:
        vs = comm_indices[comm]
        n_sample = allocations[comm]
        if n_sample <= 0 or not vs:
            continue

        raw_words = get_community_topic(community_topics, comm, max_words=8)
        kw_set = set([w.lower().strip() for w in raw_words.replace(",", " ").split() if len(w) > 2])

        scores = []
        for idx in vs:
            deg = degrees[idx]
            txt = str(G.vs[idx]["comentario_completo"] or "").lower()
            kw_hits = sum(1 for kw in kw_set if kw in txt)
            len_bonus = 1.0 if len(txt.split()) >= 8 else 0.4
            thematic_score = (deg * 0.4) + (kw_hits * 15.0) + (len_bonus * 2.0)
            scores.append(thematic_score)

        order_thematic = np.argsort(scores)[::-1]
        n_thematic = max(1, int(n_sample * 0.50))
        thematic_selected = [vs[i] for i in order_thematic[:n_thematic]]
        thematic_set = set(thematic_selected)

        rem_vs = [vs[i] for i in order_thematic[n_thematic:]]
        emo_selected = []
        n_emo = n_sample - len(thematic_selected)

        if has_factors and rem_vs and n_emo > 0:
            comm_emo_counts = Counter([G.vs[i]["factor"] for i in vs if G.vs[i]["factor"]])
            valid_emos = [e for e, count in comm_emo_counts.most_common() if count > 0]
            per_emo_quota = max(1, n_emo // max(1, len(valid_emos)))

            for emo in valid_emos:
                if len(emo_selected) >= n_emo:
                    break
                emo_pool = [i for i in rem_vs if G.vs[i]["factor"] == emo and i not in emo_selected]
                if emo_pool:
                    intensities = [G.vs[i][emo] for i in emo_pool]
                    best_pool = [emo_pool[idx] for idx in np.argsort(intensities)[::-1]]
                    emo_selected.extend(best_pool[:per_emo_quota])

        chosen = list(thematic_set.union(emo_selected))
        if len(chosen) < n_sample:
            still_rem = [i for i in vs if i not in chosen]
            if still_rem:
                fill = list(rng.choice(still_rem, size=min(n_sample - len(chosen), len(still_rem)), replace=False))
                chosen.extend(fill)

        sampled_nodes.extend(chosen[:n_sample])

    G_sub = G.subgraph(sampled_nodes)
    G_sub.vs["orig_index"] = sampled_nodes
    return G_sub


def compute_organic_cluster_separated_layout(G_sub, random_seed=42):
    """Disposición física Fruchterman-Reingold multinivel con separación de macrocomunidades."""
    rng = np.random.default_rng(random_seed)
    all_comms = sorted(list(set(G_sub.vs["community"])))
    n_comms = len(all_comms)
    comm_to_idx = {c: i for i, c in enumerate(all_comms)}

    cross_weights = defaultdict(float)
    for e in G_sub.es:
        cu = G_sub.vs[e.source]["community"]
        cv = G_sub.vs[e.target]["community"]
        if cu != cv:
            p = tuple(sorted([cu, cv]))
            w = e["weight"] if "weight" in G_sub.edge_attributes() else 1.0
            cross_weights[p] += w

    G_macro_temp = ig.Graph(n=n_comms)
    m_edges = [(comm_to_idx[p[0]], comm_to_idx[p[1]]) for p in cross_weights.keys()]
    G_macro_temp.add_edges(m_edges)
    G_macro_temp.es["weight"] = list(cross_weights.values())

    l_macro = G_macro_temp.layout_fruchterman_reingold(
        weights="weight" if G_macro_temp.ecount() > 0 else None,
        niter=400, grid="auto"
    )
    m_coords = np.array(l_macro.coords)
    m_coords -= m_coords.mean(axis=0)
    max_m = np.max(np.abs(m_coords)) if len(m_coords) > 0 and np.max(np.abs(m_coords)) > 0 else 1.0
    macro_pos = {all_comms[i]: (m_coords[i] / max_m) * 24.0 for i in range(n_comms)}

    init_coords = []
    for v in G_sub.vs:
        c = v["community"]
        base = macro_pos.get(c, np.zeros(2))
        disp = rng.normal(0, 1.8, size=2)
        init_coords.append([float(base[0] + disp[0]), float(base[1] + disp[1])])

    layout_weights = []
    for e in G_sub.es:
        cu = G_sub.vs[e.source]["community"]
        cv = G_sub.vs[e.target]["community"]
        base_w = e["weight"] if "weight" in G_sub.edge_attributes() else 1.0
        if cu == cv:
            layout_weights.append(base_w * 3.5)
        else:
            layout_weights.append(base_w * 0.22)

    G_sub.es["w_layout"] = layout_weights
    layout = G_sub.layout_fruchterman_reingold(
        weights="w_layout",
        niter=500,
        seed=init_coords,
        grid="auto"
    )
    coords = np.array(layout.coords)
    coords -= coords.mean(axis=0)
    std_val = np.std(coords)
    if std_val > 0:
        coords /= std_val
    return {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}


def build_micro_organic_figure(G_full, community_topics, subreddit_name="Subreddit", max_nodes_total=1400, random_seed=42):
    """
    Construye la Pestaña 1 (V3.0): Grafo Micro Orgánico con:
      - Etiquetas flotantes en los centroides de cada macro-comunidad (para no depender del hover).
      - Customdata estructurado para clic -> Lector de comentarios en el panel lateral.
      - Filtro interactivo por trimestre.
    """
    ensure_graph_attributes(G_full, subreddit_name=subreddit_name)
    community_topics = ensure_community_topics(G_full, community_topics)
    G_sub = sample_representative_subgraph(G_full, community_topics, max_nodes_total=max_nodes_total, random_seed=random_seed)
    pos = compute_organic_cluster_separated_layout(G_sub, random_seed=random_seed)

    N_sub_total = G_full.vcount()
    E_sub_total = G_full.ecount()
    n_sample_total = G_sub.vcount()
    e_sample_total = G_sub.ecount()

    comm_full_vols = Counter([v["community"] for v in G_full.vs if v["community"] != -1])
    total_valid_full = max(1, sum(comm_full_vols.values()))

    periods = sort_periods_chronologically(G_sub.vs["period"])
    period_full_vols = Counter(G_full.vs["period"]) if "period" in G_full.vertex_attributes() else {}

    fig = go.Figure()

    has_w = "weight" in G_sub.edge_attributes()
    weights = G_sub.es["weight"] if has_w else [1.0] * G_sub.ecount()
    w_th = float(np.percentile(weights, 25)) if weights else 0.0

    # Aristas globales
    edge_x, edge_y = [], []
    for edge in G_sub.es:
        w = edge["weight"] if has_w else 1.0
        if w >= w_th:
            u, v = edge.source, edge.target
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.55, color="rgba(148, 163, 184, 0.16)"),
        hoverinfo="skip", name="Enlaces Semánticos", showlegend=True, visible=True
    ))

    degrees = G_sub.degree()
    all_comms = sorted(list(set(G_sub.vs["community"])))

    def build_node_data(vs_subset):
        node_x, node_y, sizes, hover_texts, customdata_list = [], [], [], [], []
        for v in vs_subset:
            idx = v.index
            node_x.append(pos[idx][0])
            node_y.append(pos[idx][1])
            deg = degrees[idx]
            sizes.append(4.0 + np.sqrt(deg) * 1.8)

            cid = v["orig_index"] if "orig_index" in G_sub.vertex_attributes() else idx
            factor = v["factor"] if "factor" in G_sub.vertex_attributes() else "N/A"
            period = v["period"] if "period" in G_sub.vertex_attributes() else "N/A"
            comm = v["community"]
            topic_words = get_community_topic(community_topics, comm, max_words=5)
            emo_label = EMOTION_PALETTE.get(factor, {}).get("name", factor)

            raw_comment = str(v["comentario_completo"] or "No disponible").strip()
            words_c = raw_comment.split()
            wrapped_lines, cur_line, cur_len = [], [], 0
            for w in words_c:
                if cur_len + len(w) + 1 > 62:
                    wrapped_lines.append(" ".join(cur_line))
                    cur_line = [w]
                    cur_len = len(w)
                else:
                    cur_line.append(w)
                    cur_len += len(w) + 1
            if cur_line:
                wrapped_lines.append(" ".join(cur_line))

            if len(wrapped_lines) > 5:
                comment_html = "<br>".join([html.escape(l) for l in wrapped_lines[:5]]) + "<br><i>[Haz clic para leer completo en el panel →]</i>"
            else:
                comment_html = "<br>".join([html.escape(l) for l in wrapped_lines])

            hover_texts.append(
                f"<b>C{comm}: {topic_words}</b><br>"
                f"<b>ID:</b> #{cid} &nbsp;|&nbsp; <b>Periodo:</b> {period}<br>"
                f"<b>Emoción:</b> {emo_label} &nbsp;|&nbsp; <b>Grado:</b> {deg}<br>"
                f"<span style='color:#94a3b8'>────────────────────────────</span><br>"
                f"{comment_html}"
            )
            customdata_list.append({
                "type": "micro_node",
                "id": int(cid),
                "comm": int(comm),
                "topic": topic_words,
                "emotion": factor,
                "emotion_label": emo_label,
                "degree": int(deg),
                "period": str(period),
                "text": raw_comment,
                "color": get_community_color(comm)
            })
        return node_x, node_y, sizes, hover_texts, customdata_list

    # Centroides de cada comunidad para etiquetas fijas en el lienzo
    centroid_x, centroid_y, centroid_texts, centroid_hovers, centroid_cd = [], [], [], [], []
    for comm in all_comms:
        vs_comm = [v for v in G_sub.vs if v["community"] == comm]
        if not vs_comm:
            continue
        nx_c, ny_c, sz_c, hov_c, cd_c = build_node_data(vs_comm)
        comm_color = get_community_color(comm)
        topic_words = get_community_topic(community_topics, comm, max_words=5)
        short_topic = get_community_topic(community_topics, comm, max_words=2)
        pct_comm = (comm_full_vols[comm] / total_valid_full) * 100.0

        fig.add_trace(go.Scatter(
            x=nx_c, y=ny_c, mode="markers",
            name=f"C{comm} ({pct_comm:.1f}%): {short_topic}",
            marker=dict(size=sz_c, color=comm_color, line=dict(width=0.6, color="rgba(255,255,255,0.75)"), opacity=0.88),
            text=hov_c, customdata=cd_c, hovertemplate="%{text}<extra></extra>",
            legendgroup=f"C{comm}", showlegend=True, visible=True
        ))

        cx = float(np.median(nx_c))
        cy = float(np.median(ny_c))
        centroid_x.append(cx)
        centroid_y.append(cy)
        centroid_texts.append(f"<b>C{comm}: {short_topic}</b><br><span style='font-size:10px'>({pct_comm:.1f}%)</span>")
        centroid_hovers.append(
            f"<b>Centroide Macro-Comunidad C{comm}</b><br>"
            f"<b>Tópico Completo:</b> {topic_words}<br>"
            f"<b>Volumen Global:</b> {comm_full_vols[comm]:,} comentarios ({pct_comm:.1f}%)<br>"
            f"<i>Haz clic para inspeccionar esta comunidad en el panel lateral</i>"
        )
        centroid_cd.append({
            "type": "community_centroid",
            "comm": int(comm),
            "topic": topic_words,
            "vol": int(comm_full_vols[comm]),
            "pct": round(pct_comm, 2),
            "color": comm_color
        })

    # Capa de etiquetas de clúster sobre el lienzo (evita tener que hacer hover para saber qué es cada grupo)
    fig.add_trace(go.Scatter(
        x=centroid_x, y=centroid_y,
        mode="markers+text",
        text=centroid_texts,
        textposition="middle center",
        textfont=dict(color="#0f172a", size=15, family="Segoe UI, Arial, sans-serif"),
        marker=dict(
            size=48,
            color="rgba(255, 255, 255, 0.92)",
            symbol="square",
            line=dict(width=2.0, color="#64748b")
        ),
        hovertext=centroid_hovers,
        customdata=centroid_cd,
        hovertemplate="%{hovertext}<extra></extra>",
        name="🏷️ Etiquetas de Clúster",
        showlegend=True,
        visible=True
    ))

    base_trace_count = len(fig.data)
    quarter_trace_indices = {}
    quarter_metrics = {}

    # Trazas por trimestre
    for p in periods:
        q_start = len(fig.data)
        vs_p = [v for v in G_sub.vs if v["period"] == p]
        vs_p_set = set(v.index for v in vs_p)

        edge_x_p, edge_y_p = [], []
        q_edges_count = 0
        for edge in G_sub.es:
            if edge.source in vs_p_set and edge.target in vs_p_set:
                w = edge["weight"] if has_w else 1.0
                if w >= w_th:
                    u, v = edge.source, edge.target
                    edge_x_p.extend([pos[u][0], pos[v][0], None])
                    edge_y_p.extend([pos[u][1], pos[v][1], None])
                    q_edges_count += 1

        fig.add_trace(go.Scatter(
            x=edge_x_p, y=edge_y_p, mode="lines",
            line=dict(width=0.8, color="rgba(56, 189, 248, 0.42)"),
            hoverinfo="skip", name=f"Enlaces {p}", showlegend=False, visible=False
        ))

        for comm in all_comms:
            vs_comm_p = [v for v in vs_p if v["community"] == comm]
            if not vs_comm_p:
                continue
            nx_cp, ny_cp, sz_cp, hov_cp, cd_cp = build_node_data(vs_comm_p)
            comm_color = get_community_color(comm)
            short_topic = get_community_topic(community_topics, comm, max_words=2)
            fig.add_trace(go.Scatter(
                x=nx_cp, y=ny_cp, mode="markers",
                name=f"C{comm}: {short_topic}",
                marker=dict(size=sz_cp, color=comm_color, line=dict(width=0.7, color="rgba(255,255,255,0.85)"), opacity=0.92),
                text=hov_cp, customdata=cd_cp, hovertemplate="%{text}<extra></extra>",
                legendgroup=f"C{comm}", showlegend=False, visible=False
            ))

        # Añadir también las etiquetas de clúster de referencia en cada trimestre
        fig.add_trace(go.Scatter(
            x=centroid_x, y=centroid_y,
            mode="markers+text",
            text=centroid_texts,
            textposition="middle center",
            textfont=dict(color="#0f172a", size=15, family="Segoe UI, Arial, sans-serif"),
            marker=dict(
                size=48,
                color="rgba(255, 255, 255, 0.92)",
                symbol="square",
                line=dict(width=2.0, color="#64748b")
            ),
            hovertext=centroid_hovers,
            customdata=centroid_cd,
            hovertemplate="%{hovertext}<extra></extra>",
            name=f"Etiquetas {p}",
            showlegend=False,
            visible=False
        ))

        quarter_trace_indices[p] = list(range(q_start, len(fig.data)))
        quarter_metrics[p] = (len(vs_p), q_edges_count)

    total_traces = len(fig.data)

    title_base = (
        f"<b>1. Red Micro Orgánica con Etiquetas de Clúster</b> — r/{subreddit_name}<br>"
        f"<sup><b>Total Subreddit:</b> {N_sub_total:,} comentarios | {E_sub_total:,} aristas &nbsp;•&nbsp; "
        f"<b>Muestra Representativa:</b> {n_sample_total:,} nodos | Haz clic en cualquier punto o etiqueta para ver detalles a la derecha</sup>"
    )

    buttons = [
        dict(
            label="📅 Todos los Trimestres (Con Etiquetas)",
            method="update",
            args=[
                {"visible": [True] * base_trace_count + [False] * (total_traces - base_trace_count)},
                {"title.text": title_base}
            ]
        ),
        dict(
            label="👁️ Todos los Trimestres (Solo Puntos)",
            method="update",
            args=[
                {"visible": [True] * (base_trace_count - 1) + [False] + [False] * (total_traces - base_trace_count)},
                {"title.text": title_base}
            ]
        )
    ]

    for p in periods:
        n_q_nodes, n_q_edges = quarter_metrics[p]
        n_full_q = period_full_vols.get(p, 0)
        q_indices_set = set(quarter_trace_indices[p])
        vis_list = [False] * base_trace_count + [(i in q_indices_set) for i in range(base_trace_count, total_traces)]
        title_q = (
            f"<b>1. Red Micro Orgánica — r/{subreddit_name} [{p}]</b><br>"
            f"<sup><b>Trimestre {p}:</b> {n_full_q:,} comentarios totales ({n_q_nodes:,} en muestra | {n_q_edges:,} enlaces activos)</sup>"
        )
        buttons.append(dict(
            label=f"📅 {p} ({n_q_nodes} nodos muestra)",
            method="update",
            args=[
                {"visible": vis_list},
                {"title.text": title_q}
            ]
        ))

    fig.update_layout(
        title=dict(text=title_base, x=0.02, y=0.97),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, Arial, sans-serif"),
        hovermode="closest",
        margin=dict(l=16, r=210, t=95, b=20),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=1.0, y=1.11,
            xanchor="right", yanchor="top",
            bgcolor="#f8fafc",
            bordercolor="#cbd5e1",
            font=dict(color="#0284c7", size=11, family="Segoe UI, sans-serif"),
            buttons=buttons,
            active=0
        )],
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(
            title="Comunidades (% Vol)",
            orientation="v", y=1, yanchor="top", x=1.01, xanchor="left",
            font=dict(size=10, color="#0f172a"), bgcolor="rgba(255,255,255,0.9)"
        )
    )
    return fig


# ==============================================================================
# 2. PESTAÑA 2: VISTA MACRO AGREGADA CON FILTROS DE CLARIDAD Y ETIQUETAS RICAS
# ==============================================================================

def collapse_macro_graph(G, community_topics, pruning_percentile=0, min_cross_edges=0, ref_percentile=65, prev_cut_densities=None, prev_period_label=None):
    """
    Construye el grafo macro colapsado conservando todas las parejas de comunidades y sus métricas:
      - Densidad de corte (cut_density)
      - Cantidad de aristas compartidas (count)
      - % de diferencia en la densidad de corte respecto al trimestre anterior (pct_diff_cut_density / pct_diff_str)
    """
    ensure_graph_attributes(G)
    community_topics = ensure_community_topics(G, community_topics)
    all_comms = sorted([c for c in set(G.vs["community"]) if c != -1])
    n_comms = len(all_comms)
    if n_comms == 0:
        return ig.Graph()

    comm_to_idx = {c: i for i, c in enumerate(all_comms)}
    comm_sizes = {c: 0 for c in all_comms}
    comm_emotions = {c: Counter() for c in all_comms}
    has_f = "factor" in G.vertex_attributes()

    for v in G.vs:
        c = v["community"]
        if c in comm_sizes:
            comm_sizes[c] += 1
            if has_f and v["factor"]:
                comm_emotions[c][v["factor"]] += 1

    cross_w = defaultdict(float)
    cross_n = defaultdict(int)
    cross_emo = defaultdict(lambda: Counter())

    has_edge_weight = "weight" in G.edge_attributes()
    for e in G.es:
        c_u = G.vs[e.source]["community"]
        c_v = G.vs[e.target]["community"]
        if c_u == -1 or c_v == -1:
            continue
        if c_u != c_v:
            pair = tuple(sorted([c_u, c_v]))
            w = e["weight"] if has_edge_weight else 1.0
            cross_w[pair] += w
            cross_n[pair] += 1
            if has_f:
                f_u = G.vs[e.source]["factor"]
                f_v = G.vs[e.target]["factor"]
                emo = f_u if (f_u and f_u == f_v) else "NoSharedEmotion"
            else:
                emo = "NoSharedEmotion"
            cross_emo[pair][emo] += 1

    quarterly_pct_diffs = {}
    if prev_cut_densities is None and "period" in G.vertex_attributes():
        periods_all = sort_periods_chronologically(G.vs["period"])
        if len(periods_all) >= 2:
            p_comm_sizes = defaultdict(lambda: defaultdict(int))
            for v in G.vs:
                c = v["community"]
                p = v["period"]
                if c != -1 and p:
                    p_comm_sizes[p][c] += 1
            p_cross_w = defaultdict(lambda: defaultdict(float))
            for e in G.es:
                u_v, v_v = G.vs[e.source], G.vs[e.target]
                cu, cv = u_v["community"], v_v["community"]
                pu, pv = u_v["period"], v_v["period"]
                if cu != -1 and cv != -1 and cu != cv and pu == pv and pu:
                    pair = tuple(sorted([cu, cv]))
                    w = e["weight"] if has_edge_weight else 1.0
                    p_cross_w[pu][pair] += w

            for i in range(n_comms):
                for j in range(i + 1, n_comms):
                    pair = (all_comms[i], all_comms[j])
                    q_dens_list = []
                    for p in periods_all:
                        s1 = p_comm_sizes[p].get(pair[0], 0)
                        s2 = p_comm_sizes[p].get(pair[1], 0)
                        denom_p = s1 * s2
                        d_p = (p_cross_w[p].get(pair, 0.0) / denom_p) if denom_p > 0 else 0.0
                        q_dens_list.append((p, d_p))
                    diffs_pair = []
                    for k_idx in range(1, len(q_dens_list)):
                        d_prev = q_dens_list[k_idx - 1][1]
                        d_curr = q_dens_list[k_idx][1]
                        if d_prev > 0:
                            diffs_pair.append(((d_curr - d_prev) / d_prev) * 100.0)
                    last_p, last_d = q_dens_list[-1]
                    prev_p, prev_d = q_dens_list[-2]
                    if prev_d > 0:
                        last_diff = ((last_d - prev_d) / prev_d) * 100.0
                        last_str = f"{last_diff:+.2f}% ({last_p} vs {prev_p})"
                    elif last_d > 0:
                        last_diff = 100.0
                        last_str = f"+100.00% ({last_p} vs {prev_p})"
                    else:
                        last_diff = 0.0
                        last_str = f"0.00% ({last_p} vs {prev_p})"
                    mean_diff = float(np.mean(diffs_pair)) if diffs_pair else 0.0
                    quarterly_pct_diffs[pair] = (last_diff, f"{last_str} | Prom: {mean_diff:+.1f}%")

    G_macro = ig.Graph(n=n_comms)
    G_macro.vs["community"] = all_comms
    G_macro.vs["size"] = [comm_sizes[c] for c in all_comms]
    G_macro.vs["topic_words"] = [get_community_topic(community_topics, c, max_words=5) for c in all_comms]
    G_macro.vs["short_topic"] = [get_community_topic(community_topics, c, max_words=2) for c in all_comms]
    G_macro.vs["dominant_emotion"] = [comm_emotions[c].most_common(1)[0][0] if comm_emotions[c] else "N/A" for c in all_comms]

    candidates = []
    for i in range(n_comms):
        for j in range(i + 1, n_comms):
            c1, c2 = all_comms[i], all_comms[j]
            pair = (c1, c2)
            count = cross_n.get(pair, 0)
            tw = cross_w.get(pair, 0.0)
            denom = comm_sizes[c1] * comm_sizes[c2]
            dens = (tw / denom) if denom > 0 else 0.0
            clean_emo = Counter({k: v for k, v in cross_emo[pair].items() if k != "NoSharedEmotion"})
            dom_emo = clean_emo.most_common(1)[0][0] if clean_emo else "NoSharedEmotion"

            if prev_cut_densities is not None:
                if len(prev_cut_densities) == 0:
                    pct_diff_val = 0.0
                    pct_diff_str = "N/A (Trimestre inicial)"
                else:
                    prev_d = prev_cut_densities.get(pair, 0.0)
                    lbl = f" vs {prev_period_label}" if prev_period_label else ""
                    if prev_d > 0:
                        pct_diff_val = ((dens - prev_d) / prev_d) * 100.0
                        pct_diff_str = f"{pct_diff_val:+.2f}%{lbl}"
                    elif dens > 0:
                        pct_diff_val = 100.0
                        pct_diff_str = f"+100.00% (Nueva{lbl})"
                    else:
                        pct_diff_val = 0.0
                        pct_diff_str = f"0.00% (Sin cambio{lbl})"
            elif pair in quarterly_pct_diffs:
                pct_diff_val, pct_diff_str = quarterly_pct_diffs[pair]
            else:
                pct_diff_val = 0.0
                pct_diff_str = "N/A"

            candidates.append({
                "u": comm_to_idx[c1], "v": comm_to_idx[c2], "c1": c1, "c2": c2,
                "count": count, "total_w": tw, "cut_density": dens, "dom_emo": dom_emo,
                "pct_diff_cut_density": float(pct_diff_val),
                "pct_diff_str": pct_diff_str
            })

    if not candidates:
        return G_macro

    edges, attrs = [], defaultdict(list)
    for c in candidates:
        edges.append((c["u"], c["v"]))
        attrs["cut_density"].append(c["cut_density"])
        attrs["count"].append(c["count"])
        attrs["total_w"].append(c["total_w"])
        attrs["dominant_emotion"].append(c["dom_emo"])
        attrs["pct_diff_cut_density"].append(c["pct_diff_cut_density"])
        attrs["pct_diff_str"].append(c["pct_diff_str"])

    G_macro.add_edges(edges)
    for k, vals in attrs.items():
        G_macro.es[k] = vals
    return G_macro


def build_macro_figure(G_macro, subreddit_name="Subreddit"):
    """
    Construye la Pestaña 2 (V3.0): Macro Agregado con:
      - Etiquetas enriquecidas debajo de cada nodo (C# + Tópico corto + % Vol).
      - Botones rápidos en el gráfico para alternar entre:
        1) Todas las Aristas Conectadas
        2) Top 15 Puentes Más Fuertes (con el valor de ρ visible sobre las aristas)
        3) Top 10 Mayor Cambio Trimestral (|Δ%|)
    """
    fig = go.Figure()
    if G_macro.vcount() == 0:
        return fig

    n_comms = G_macro.vcount()
    # Disposición circular/fuerza limpia que evita solapamiento en el centro
    init_seed = [[float(np.cos(2 * np.pi * i / n_comms)), float(np.sin(2 * np.pi * i / n_comms))] for i in range(n_comms)]
    if G_macro.ecount() > 0 and "cut_density" in G_macro.edge_attributes():
        w_layout = [max(float(d), 1e-12) for d in G_macro.es["cut_density"]]
        layout = G_macro.layout_fruchterman_reingold(weights=w_layout, niter=500, seed=init_seed)
    elif G_macro.ecount() > 0:
        layout = G_macro.layout_fruchterman_reingold(niter=500, seed=init_seed)
    else:
        layout = G_macro.layout_circle()

    coords = np.array(layout.coords)
    coords -= coords.mean(axis=0)
    max_r = np.max(np.abs(coords)) if len(coords) > 0 and np.max(np.abs(coords)) > 0 else 1.0
    pos = {i: (coords[i, 0] / max_r * 10.0, coords[i, 1] / max_r * 10.0) for i in range(len(coords))}

    # Ordenar aristas para saber cuáles son Top 15 por ρ y Top 10 por |Δ%|
    edge_records = []
    if G_macro.ecount() > 0:
        dens = np.array(G_macro.es["cut_density"])
        d_min, d_max = np.min(dens), np.max(dens)
        d_range = (d_max - d_min) if (d_max - d_min) > 0 else 1.0

        for idx_e, e in enumerate(G_macro.es):
            edge_records.append((idx_e, float(e["cut_density"]), abs(float(e["pct_diff_cut_density"])), int(e["count"])))

        top_rho_set = set(x[0] for x in sorted(edge_records, key=lambda item: item[1], reverse=True)[:15] if x[3] > 0)
        top_chg_set = set(x[0] for x in sorted(edge_records, key=lambda item: item[2], reverse=True)[:10] if x[3] > 0)

        seen_emo = set()
        t_interp = np.linspace(0.0, 1.0, 15)
        n_pts = len(t_interp)

        edge_trace_meta = []  # (trace_idx, is_top_rho, is_top_chg)
        mid_x_rho, mid_y_rho, mid_txt_rho, mid_hov_rho = [], [], [], []
        mid_x_chg, mid_y_chg, mid_txt_chg, mid_hov_chg = [], [], [], []

        for idx_e, e in enumerate(G_macro.es):
            u, v = e.source, e.target
            c_u = G_macro.vs[u]["community"]
            c_v = G_macro.vs[v]["community"]
            t_u = G_macro.vs[u]["short_topic"] if "short_topic" in G_macro.vertex_attributes() else ""
            t_v = G_macro.vs[v]["short_topic"] if "short_topic" in G_macro.vertex_attributes() else ""
            emo = e["dominant_emotion"]
            cfg = EMOTION_PALETTE.get(emo, {"name": "Mixto / Sin emoción única", "short": "Mixto", "color": "rgba(148, 163, 184, 0.38)"})
            width = 1.3 + ((e["cut_density"] - d_min) / d_range) * 7.2
            pct_diff_str = e["pct_diff_str"] if "pct_diff_str" in G_macro.edge_attributes() else "N/A"
            pct_diff_val = e["pct_diff_cut_density"] if "pct_diff_cut_density" in G_macro.edge_attributes() else 0.0

            hover = (
                f"<b>Puente Semántico: C{c_u} ({t_u}) ↔ C{c_v} ({t_v})</b><br>"
                f"• <b>Densidad de Corte (ρ):</b> {e['cut_density']:.4e}<br>"
                f"• <b>Aristas Compartidas:</b> {e['count']:,}<br>"
                f"• <b>Emoción del Puente:</b> {cfg['name']}<br>"
                f"• <b>Cambio Trimestral (Δ% ρ):</b> {pct_diff_str}"
            )

            edge_customdata = {
                "type": "macro_edge",
                "c1": int(c_u),
                "c2": int(c_v),
                "t1": t_u,
                "t2": t_v,
                "emotion": cfg["name"],
                "cut_density": float(e["cut_density"]),
                "count": int(e["count"]),
                "pct_diff_cut_density": float(pct_diff_val),
                "pct_diff_str": str(pct_diff_str)
            }

            x_pts = list(pos[u][0] + t_interp * (pos[v][0] - pos[u][0])) + [None]
            y_pts = list(pos[u][1] + t_interp * (pos[v][1] - pos[u][1])) + [None]

            t_idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=x_pts, y=y_pts, mode="lines",
                line=dict(width=width, color=cfg["color"]),
                hoverinfo="text",
                hovertext=[hover] * n_pts + [None],
                customdata=[edge_customdata] * n_pts + [None],
                hovertemplate="%{hovertext}<extra></extra>",
                name=f"Puente: {cfg['short']}", legendgroup=f"Emo_{emo}", showlegend=False,
                visible=True
            ))
            seen_emo.add(emo)
            edge_trace_meta.append((t_idx, idx_e in top_rho_set, idx_e in top_chg_set))

            mx = (pos[u][0] + pos[v][0]) / 2.0
            my = (pos[u][1] + pos[v][1]) / 2.0
            if idx_e in top_rho_set:
                mid_x_rho.append(mx)
                mid_y_rho.append(my)
                mid_txt_rho.append(f"ρ={e['cut_density']:.1e}")
                mid_hov_rho.append(hover)
            if idx_e in top_chg_set:
                mid_x_chg.append(mx)
                mid_y_chg.append(my)
                mid_txt_chg.append(f"{pct_diff_val:+.1f}%")
                mid_hov_chg.append(hover)

        # Traza de etiquetas en el punto medio para Top 15 ρ (oculta por defecto, visible al activar Top 15)
        rho_lbl_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=mid_x_rho, y=mid_y_rho, mode="text",
            text=mid_txt_rho,
            textfont=dict(color="#0f172a", size=13.5, family="Consolas, monospace"),
            hovertext=mid_hov_rho, hovertemplate="%{hovertext}<extra></extra>",
            name="Valores ρ", showlegend=False, visible=False
        ))

        chg_lbl_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=mid_x_chg, y=mid_y_chg, mode="text",
            text=mid_txt_chg,
            textfont=dict(color="#b45309", size=13.5, family="Consolas, monospace"),
            hovertext=mid_hov_chg, hovertemplate="%{hovertext}<extra></extra>",
            name="Valores Δ%", showlegend=False, visible=False
        ))
    else:
        edge_trace_meta = []
        rho_lbl_idx = None
        chg_lbl_idx = None

    sizes = np.array(G_macro.vs["size"])
    total_vol = max(1, int(np.sum(sizes)))
    s_min, s_max = np.min(sizes), np.max(sizes)
    s_range = (np.sqrt(s_max) - np.sqrt(s_min)) if (np.sqrt(s_max) - np.sqrt(s_min)) > 0 else 1.0

    x_n, y_n, s_vis, colors, inside_labels, bottom_labels, hovers, node_cd = [], [], [], [], [], [], [], []
    for v in G_macro.vs:
        idx = v.index
        x_n.append(pos[idx][0])
        y_n.append(pos[idx][1])
        c = v["community"]
        vol = int(v["size"])
        pct = (vol / total_vol) * 100.0
        s_vis.append(30.0 + ((np.sqrt(vol) - np.sqrt(s_min)) / s_range) * 38.0)
        colors.append(get_community_color(c))
        inside_labels.append(f"<b>C{c}</b>")
        short_t = v["short_topic"] if "short_topic" in G_macro.vertex_attributes() else v["topic_words"]
        bottom_labels.append(f"<b>{short_t}</b> ({pct:.1f}%)")
        dom_e = v["dominant_emotion"]
        emo_full = EMOTION_PALETTE.get(dom_e, {}).get("name", dom_e)

        hovers.append(
            f"<b>Macro-Comunidad C{c} ({pct:.1f}% del total)</b><br>"
            f"• <b>Volumen Total:</b> {vol:,} comentarios<br>"
            f"• <b>Tópico Semántico:</b> {v['topic_words']}<br>"
            f"• <b>Emoción Dominante:</b> {emo_full}"
        )
        node_cd.append({
            "type": "macro_node",
            "comm": int(c),
            "vol": vol,
            "pct": round(pct, 2),
            "topic": v["topic_words"],
            "emotion": dom_e,
            "emotion_label": emo_full,
            "color": get_community_color(c)
        })

    # Nodos principales con ID C# dentro del círculo
    node_trace_idx = len(fig.data)
    fig.add_trace(go.Scatter(
        x=x_n, y=y_n, mode="markers+text", text=inside_labels, textposition="middle center",
        textfont=dict(color="white", size=16, family="Arial Black"),
        marker=dict(size=s_vis, color=colors, line=dict(width=2.2, color="#f8fafc"), opacity=0.96),
        hovertext=hovers, customdata=node_cd, hovertemplate="%{hovertext}<extra></extra>",
        name="Macro-Comunidades", showlegend=False, visible=True
    ))

    # Etiquetas descriptivas debajo de cada nodo (Tópico corto + %) para entender el grafo sin hover
    label_trace_idx = len(fig.data)
    y_lbl = [y - 1.15 for y in y_n]
    fig.add_trace(go.Scatter(
        x=x_n, y=y_lbl, mode="text", text=bottom_labels, textposition="bottom center",
        textfont=dict(color="#0f172a", size=14.5, family="Segoe UI, Arial, sans-serif"),
        hoverinfo="skip", showlegend=False, visible=True
    ))

    total_traces = len(fig.data)
    vis_all = [True] * total_traces
    if rho_lbl_idx is not None:
        vis_all[rho_lbl_idx] = False
    if chg_lbl_idx is not None:
        vis_all[chg_lbl_idx] = False

    vis_top_rho = [False] * total_traces
    vis_top_chg = [False] * total_traces
    for t_idx, is_rho, is_chg in edge_trace_meta:
        vis_top_rho[t_idx] = is_rho
        vis_top_chg[t_idx] = is_chg
    if rho_lbl_idx is not None:
        vis_top_rho[rho_lbl_idx] = True
    if chg_lbl_idx is not None:
        vis_top_chg[chg_lbl_idx] = True
    vis_top_rho[node_trace_idx] = True
    vis_top_rho[label_trace_idx] = True
    vis_top_chg[node_trace_idx] = True
    vis_top_chg[label_trace_idx] = True

    buttons = [
        dict(
            label="🔥 Top 15 Puentes Más Fuertes (Con ρ)",
            method="update",
            args=[
                {"visible": vis_top_rho},
                {"title.text": f"<b>2. Grafo Macro Agregado (Top 15 Puentes por Densidad ρ)</b> — r/{subreddit_name}<br><sup>Vista limpia con los 15 puentes semánticos más intensos y sus valores de ρ | Consulta la tabla completa a la derecha</sup>"}
            ]
        ),
        dict(
            label="🌐 Todas las Aristas (Grafo Completo)",
            method="update",
            args=[
                {"visible": vis_all},
                {"title.text": f"<b>2. Grafo Macro Agregado (Todas las Comunidades Conectadas)</b> — r/{subreddit_name}<br><sup>Red completa sin filtrar | Haz clic en una comunidad o usa la tabla derecha para inspeccionar ρ y Δ%</sup>"}
            ]
        ),
        dict(
            label="⚡ Top 10 Mayor Cambio Trimestral (|Δ%|)",
            method="update",
            args=[
                {"visible": vis_top_chg},
                {"title.text": f"<b>2. Grafo Macro Agregado (Top 10 Mayor Cambio Trimestral Δ%)</b> — r/{subreddit_name}<br><sup>Aristas con mayor variación porcentual en su densidad de corte</sup>"}
            ]
        )
    ]

    # Por defecto dejamos visible "Todas las Aristas" pero con el botón de Top 15 listo al lado,
    # o iniciamos en Todas las Aristas para cumplir estrictamente con mostrar todas conectadas.
    fig.update_layout(
        title=dict(
            text=f"<b>2. Grafo Macro Agregado (Todas las Comunidades Conectadas)</b> — r/{subreddit_name}<br>"
                 f"<sup>Cada nodo muestra su tópico y % de volumen | Usa los botones superiores para limpiar aristas o consulta el Ranking a la derecha</sup>",
            x=0.02, y=0.97
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, Arial, sans-serif"),
        hovermode="closest",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.0, y=1.11,
            xanchor="right", yanchor="top",
            bgcolor="#f8fafc",
            bordercolor="#cbd5e1",
            font=dict(color="#0f172a", size=10.5),
            buttons=buttons,
            active=1
        )],
        margin=dict(l=30, r=40, t=95, b=25),
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-12.5, 12.5]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-13.0, 12.0]),
        showlegend=False
    )
    return fig


# ==============================================================================
# 3. PESTAÑA 3: EVOLUCIÓN TEMPORAL DEL GRAFO CON ETIQUETAS CLARAS
# ==============================================================================

def build_macro_temporal_figure(G_full, community_topics, subreddit_name="Subreddit",
                                pruning_percentile=0, min_cross_edges=0, ref_percentile=60):
    """
    Construye la Pestaña 3 (V3.0): Evolución Temporal del Grafo Macro (Snapshots Trimestrales)
    con etiquetas permanentes de tópico y % del trimestre en cada nodo.
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    periods = sort_periods_chronologically(G_full.vs["period"])
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    if not periods or not all_comms:
        return go.Figure()

    G_macro_g = collapse_macro_graph(
        G_full, community_topics,
        pruning_percentile=pruning_percentile,
        min_cross_edges=min_cross_edges,
        ref_percentile=ref_percentile
    )
    n_comms = len(G_macro_g.vs)
    if n_comms == 0:
        return go.Figure()

    init_seed = [[float(np.cos(2 * np.pi * i / n_comms)), float(np.sin(2 * np.pi * i / n_comms))] for i in range(n_comms)]
    w_layout_g = [max(float(d), 1e-12) for d in G_macro_g.es["cut_density"]] if G_macro_g.ecount() > 0 else None
    layout = G_macro_g.layout_fruchterman_reingold(
        weights=w_layout_g,
        niter=600,
        seed=init_seed
    )
    coords_g = np.array(layout.coords)
    coords_g -= coords_g.mean(axis=0)
    max_r = np.max(np.abs(coords_g)) if len(coords_g) > 0 and np.max(np.abs(coords_g)) > 0 else 1.0
    anchor_pos = {c: (coords_g[i, 0] / max_r * 10.0, coords_g[i, 1] / max_r * 10.0)
                  for i, c in enumerate(G_macro_g.vs["community"])}

    fig = go.Figure()
    trace_map = {}
    seen_emo = set()

    period_nodes_map = {p: [v.index for v in G_full.vs if v["period"] == p] for p in periods}
    max_period_vol = 1
    for p in periods:
        c_counts = Counter([G_full.vs[idx]["community"] for idx in period_nodes_map[p] if G_full.vs[idx]["community"] != -1])
        if c_counts:
            max_period_vol = max(max_period_vol, max(c_counts.values()))

    period_total_vols = {p: len(period_nodes_map[p]) for p in periods}
    period_edge_counts = {}
    valid_periods = []
    prev_cut_dens_map = {}
    prev_p_label = None
    t_interp = np.linspace(0.0, 1.0, 13)
    n_pts = len(t_interp)

    for p in periods:
        nodes_p = period_nodes_map[p]
        if len(nodes_p) < 15:
            continue

        valid_periods.append(p)
        start_idx = len(fig.data)

        G_sub_p = G_full.subgraph(nodes_p)
        G_macro_p = collapse_macro_graph(
            G_sub_p, community_topics,
            pruning_percentile=pruning_percentile,
            min_cross_edges=min_cross_edges,
            ref_percentile=ref_percentile,
            prev_cut_densities=prev_cut_dens_map,
            prev_period_label=prev_p_label
        )
        period_edge_counts[p] = G_macro_p.ecount()

        curr_cut_dens_map = {}
        for e in G_macro_p.es:
            cu = G_macro_p.vs[e.source]["community"]
            cv = G_macro_p.vs[e.target]["community"]
            pair = tuple(sorted([cu, cv]))
            curr_cut_dens_map[pair] = float(e["cut_density"])
        prev_cut_dens_map = curr_cut_dens_map
        prev_p_label = p

        if G_macro_p.ecount() > 0:
            dens = np.array(G_macro_p.es["cut_density"])
            d_min, d_max = np.min(dens), np.max(dens)
            d_range = (d_max - d_min) if (d_max - d_min) > 0 else 1.0

            edges_by_emo = defaultdict(list)
            for e in G_macro_p.es:
                c1 = G_macro_p.vs[e.source]["community"]
                c2 = G_macro_p.vs[e.target]["community"]
                if c1 in anchor_pos and c2 in anchor_pos:
                    emo = e["dominant_emotion"]
                    edges_by_emo[emo].append((
                        c1, c2, e["cut_density"], e["count"],
                        e["pct_diff_cut_density"], e["pct_diff_str"]
                    ))

            for emo, edge_list in edges_by_emo.items():
                cfg = EMOTION_PALETTE.get(emo, {"name": "Mixto", "short": "Mixto", "color": "rgba(148, 163, 184, 0.35)"})
                for c1, c2, c_dens, cross_cnt, pct_diff_val, pct_diff_str in edge_list:
                    width = 1.2 + ((c_dens - d_min) / d_range) * 6.5
                    t1 = get_community_topic(community_topics, c1, max_words=2)
                    t2 = get_community_topic(community_topics, c2, max_words=2)
                    hover = (
                        f"<b>[{p}] Puente: C{c1} ({t1}) ↔ C{c2} ({t2})</b><br>"
                        f"• <b>Densidad de Corte (ρ):</b> {c_dens:.4e}<br>"
                        f"• <b>Aristas Compartidas:</b> {cross_cnt:,}<br>"
                        f"• <b>Cambio vs Trimestre Anterior:</b> {pct_diff_str}"
                    )
                    edge_cd = {
                        "type": "temporal_edge",
                        "period": str(p),
                        "c1": int(c1),
                        "c2": int(c2),
                        "t1": t1,
                        "t2": t2,
                        "emotion": cfg["name"],
                        "cut_density": float(c_dens),
                        "count": int(cross_cnt),
                        "pct_diff_cut_density": float(pct_diff_val),
                        "pct_diff_str": str(pct_diff_str)
                    }
                    x0, y0 = anchor_pos[c1]
                    x1, y1 = anchor_pos[c2]
                    x_pts = list(x0 + t_interp * (x1 - x0)) + [None]
                    y_pts = list(y0 + t_interp * (y1 - y0)) + [None]

                    fig.add_trace(go.Scatter(
                        x=x_pts,
                        y=y_pts,
                        mode="lines",
                        line=dict(width=width, color=cfg["color"]),
                        visible=False,
                        name=f"Puente: {cfg['short']}",
                        legendgroup=f"Emo_{emo}",
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=[hover] * n_pts + [None],
                        customdata=[edge_cd] * n_pts + [None],
                        hovertemplate="%{hovertext}<extra></extra>"
                    ))
                    seen_emo.add(emo)

        x_n, y_n, s_n, col_n, txt_n, bot_n, hov_n, cd_n = [], [], [], [], [], [], [], []
        total_q_vol = period_total_vols.get(p, 0)
        for v in G_macro_p.vs:
            c = v["community"]
            if c in anchor_pos:
                x_n.append(anchor_pos[c][0])
                y_n.append(anchor_pos[c][1])
                vol_c = int(v["size"])
                s_n.append(20.0 + (np.sqrt(vol_c) / np.sqrt(max_period_vol)) * 34.0)
                col_n.append(get_community_color(c))
                txt_n.append(f"<b>C{c}</b>")
                top_str = get_community_topic(community_topics, c, max_words=5)
                short_str = get_community_topic(community_topics, c, max_words=2)
                pct_of_q = (vol_c / total_q_vol * 100.0) if total_q_vol > 0 else 0.0
                bot_n.append(f"<b>{short_str}</b> ({pct_of_q:.1f}%)")
                dom_e = v["dominant_emotion"]
                emo_full = EMOTION_PALETTE.get(dom_e, {}).get("name", dom_e)

                hov_n.append(
                    f"<b>Trimestre {p} — Macro-Comunidad C{c}</b><br>"
                    f"• <b>Tópico:</b> {top_str}<br>"
                    f"• <b>Comentarios en {p}:</b> {vol_c:,} de {total_q_vol:,} ({pct_of_q:.1f}%)<br>"
                    f"• <b>Emoción Dominante:</b> {emo_full}"
                )
                cd_n.append({
                    "type": "temporal_node",
                    "period": str(p),
                    "comm": int(c),
                    "vol": vol_c,
                    "pct": round(pct_of_q, 2),
                    "topic": top_str,
                    "emotion": dom_e,
                    "emotion_label": emo_full,
                    "color": get_community_color(c)
                })

        fig.add_trace(go.Scatter(
            x=x_n, y=y_n, mode="markers+text", text=txt_n, textposition="middle center",
            textfont=dict(color="white", size=16, family="Arial Black"),
            marker=dict(size=[max(30, s * 1.12) for s in s_n], color=col_n, line=dict(width=2.2, color="white"), opacity=0.95),
            hovertext=hov_n, customdata=cd_n, hovertemplate="%{hovertext}<extra></extra>",
            visible=False, showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=x_n, y=[y - 1.15 for y in y_n], mode="text", text=bot_n, textposition="bottom center",
            textfont=dict(color="#0f172a", size=14.5, family="Segoe UI, Arial, sans-serif"),
            hoverinfo="skip", visible=False, showlegend=False
        ))

        trace_map[p] = list(range(start_idx, len(fig.data)))

    if valid_periods:
        for idx in trace_map[valid_periods[0]]:
            fig.data[idx].visible = True

    steps = [
        dict(
            label=p, method="update",
            args=[
                {"visible": [i in trace_map[p] for i in range(len(fig.data))]},
                {"title.text": (
                    f"<b>3. Evolución Temporal del Grafo Macro</b> — r/{subreddit_name} [{p}]<br>"
                    f"<sup><b>Total en {p}:</b> {period_total_vols.get(p, 0):,} comentarios &nbsp;•&nbsp; "
                    f"<b>Puentes activos:</b> {period_edge_counts.get(p, 0)} macroaristas &nbsp;•&nbsp; La tabla derecha se actualiza con el trimestre</sup>"
                )}
            ]
        )
        for p in valid_periods
    ]

    title_init = (
        f"<b>3. Evolución Temporal del Grafo Macro</b> — r/{subreddit_name} [{valid_periods[0]}]<br>"
        f"<sup><b>Total en {valid_periods[0]}:</b> {period_total_vols.get(valid_periods[0], 0):,} comentarios &nbsp;•&nbsp; "
        f"<b>Puentes activos:</b> {period_edge_counts.get(valid_periods[0], 0)} macroaristas &nbsp;•&nbsp; La tabla derecha se actualiza con el trimestre</sup>"
    )

    fig.update_layout(
        title=dict(text=title_init, x=0.02, y=0.97),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, Arial, sans-serif"),
        hovermode="closest",
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="📅 Trimestre Activo: ",
                font=dict(size=16, color="#0284c7", family="Segoe UI, sans-serif"),
                visible=True,
                xanchor="left"
            ),
            pad=dict(t=20, b=10),
            len=0.92, x=0.04, y=-0.02,
            bgcolor="#f1f5f9",
            bordercolor="#cbd5e1",
            borderwidth=1.5,
            activebgcolor="#2563eb",
            tickcolor="#64748b",
            ticklen=5,
            font=dict(color="#334155", size=13.5, family="Segoe UI, sans-serif"),
            steps=steps
        )],
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-12.5, 12.5]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-13.0, 12.0]),
        margin=dict(l=30, r=40, t=95, b=85),
        showlegend=False
    )
    return fig


# ==============================================================================
# 4. PESTAÑA 4: TRAYECTORIAS DE CUOTA RELATIVA (%), CAMBIO ESTRUCTURAL (Δ pp),
#    ÁREAS APILADAS 100% Y EVOLUCIÓN EMOCIONAL (CON SELECTOR DE COMUNIDADES)
# ==============================================================================

AVOID_COMMS_BY_SUB = {
    "Anxiety": {9},
    "depression": {2},
    "SuicideWatch": {0, 9}
}

KEY_FINDINGS_COMMS_BY_SUB = {
    "Anxiety": [2, 0, 1, 4, 6],
    "depression": [6, 10, 7, 9, 1, 11],
    "SuicideWatch": [8, 2, 1, 7, 3, 4]
}

MISSING_QUARTERS_NOTES = {
    "depression": {
        "2019Q4": "Sin Dic 2019",
        "2021Q2": "Sin May 2021"
    },
    "SuicideWatch": {
        "2022Q2": "Sin Jun 2022"
    }
}


def build_community_period_bubble_grid(G_full, community_topics, subreddit_name="Subreddit"):
    """
    Construye la Pestaña 4: Tendencias de Cuota (%)
    Enfocada 100% en volumen y participación estructural:
      - Bloque 0 (mode='share'): Líneas de Cuota Relativa (% del Trimestre).
      - Bloque 1 (mode='delta_pp'): Líneas de Cambio Estructural Acumulado (Δ pp vs 2019Q1).
    Eje X multiíndice agrupado por año y trimestre (2019Q1 – 2022Q3), con etiquetas y ejes ampliados.
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    periods = sort_periods_chronologically(G_full.vs["period"])
    if not periods or not all_comms:
        return go.Figure()

    comm_totals = Counter([v["community"] for v in G_full.vs if v["community"] != -1])
    total_all = max(1, sum(comm_totals.values()))
    comms_by_vol = sorted(all_comms, key=lambda c: comm_totals[c], reverse=True)

    cell_vols = defaultdict(int)
    cell_emo_counts = defaultdict(lambda: Counter())
    for v in G_full.vs:
        c, p, f = v["community"], v["period"], v["factor"]
        if c != -1 and p:
            cell_vols[(c, p)] += 1
            if f:
                cell_emo_counts[(c, p)][f] += 1

    quarter_totals = defaultdict(int)
    for (c, p), v_count in cell_vols.items():
        quarter_totals[p] += v_count

    default_Selected = set(KEY_FINDINGS_COMMS_BY_SUB.get(subreddit_name, comms_by_vol[:5]))
    key_periods_set = {periods[0], "2020Q2", "2021Q3", "2021Q4", periods[-1]}
    missing_notes = MISSING_QUARTERS_NOTES.get(subreddit_name, {})

    years = [str(p)[:4] for p in periods]
    quarters = [str(p)[4:] for p in periods]
    multi_x = [years, quarters]

    fig = go.Figure()

    comm_series = {}
    for c in comms_by_vol:
        c_color = get_community_color(c)
        topic_full = get_community_topic(community_topics, c, max_words=5).replace("_", " ")
        topic_short = get_community_topic(community_topics, c, max_words=2).replace("_", " ")
        global_pct = (comm_totals[c] / total_all) * 100.0

        y_share = []
        y_delta_pp = []
        y_dom_emo_pct = []
        marker_emo_colors = []
        marker_sizes_share = []
        texts_share = []
        texts_delta = []
        hovers = []
        cds = []

        base_share = None

        for idx_p, p in enumerate(periods):
            vol = cell_vols.get((c, p), 0)
            q_tot = quarter_totals.get(p, 0)
            share_p = (vol / q_tot * 100.0) if q_tot > 0 else 0.0
            if base_share is None:
                base_share = share_p
            delta_pp = share_p - base_share

            counts = cell_emo_counts.get((c, p), Counter())
            if counts and vol > 0:
                dom_emo, dom_cnt = counts.most_common(1)[0]
                dom_pct = (dom_cnt / vol) * 100.0
            else:
                dom_emo, dom_cnt, dom_pct = "neutral", 0, 0.0

            emo_cfg = EMOTION_PALETTE.get(dom_emo, EMOTION_PALETTE["neutral"])
            emo_pcts = {
                k: round((counts.get(k, 0) / vol * 100.0) if vol > 0 else 0.0, 1)
                for k in FACTOR_COLS
            }

            is_key_q = (p in key_periods_set)
            is_last_q = (idx_p == len(periods) - 1)

            y_share.append(round(share_p, 2))
            y_delta_pp.append(round(delta_pp, 2))
            y_dom_emo_pct.append(round(dom_pct, 1))
            marker_emo_colors.append(emo_cfg["hex"])
            marker_sizes_share.append(11.0 if not is_key_q else 12.0)

            if is_last_q:
                texts_share.append(f"  <b>C{c}</b> ({topic_short}): <b>{share_p:.1f}%</b> ({delta_pp:+.1f} pp)")
                texts_delta.append(f"  <b>C{c}</b> ({topic_short}): <b>{delta_pp:+.2f} pp</b>")
            elif p in {"2019Q1", "2020Q2", "2021Q4"}:
                texts_share.append(f"{share_p:.1f}%")
                texts_delta.append(f"{delta_pp:+.1f}pp")
            else:
                texts_share.append("")
                texts_delta.append("")

            counts_str = "<br>".join([
                f"  • {EMOTION_PALETTE.get(k, {}).get('name', k)}: {v:,} ({v/vol*100:.1f}%)"
                for k, v in counts.most_common()
            ]) if vol > 0 else "  • Sin registros"

            miss_tag = f" &nbsp;<span style='color:#fbbf24'>⚠️ [{missing_notes[p]}]</span>" if p in missing_notes else ""
            hover_str = (
                f"<b>C{c} ({global_pct:.1f}% global): {topic_full}</b><br>"
                f"<b>Trimestre:</b> {p}{miss_tag} &nbsp;|&nbsp; <b>Volumen:</b> {vol:,} de {q_tot:,} com.<br>"
                f"<b>Cuota Relativa en {p}:</b> <b>{share_p:.2f}%</b> del subreddit<br>"
                f"<b>Cambio Estructural vs 2019Q1:</b> <b>{delta_pp:+.2f} puntos porcentuales (pp)</b><br>"
                f"<b>Emoción Dominante en {p}:</b> <span style='color:{emo_cfg['hex']}'>● {emo_cfg['name']} ({dom_pct:.1f}%)</span><br>"
                f"<span style='color:#94a3b8'>────────────────────────────────────</span><br>"
                f"<b>Desglose Afectivo en {p}:</b><br>{counts_str}"
            )
            hovers.append(hover_str)

            cds.append({
                "type": "evo_point",
                "comm": int(c),
                "topic": topic_full,
                "topic_short": topic_short,
                "color": c_color,
                "period": str(p),
                "vol": int(vol),
                "q_total": int(q_tot),
                "pct_quarter": round(share_p, 2),
                "delta_pp": round(delta_pp, 2),
                "pct_dom": round(dom_pct, 1),
                "dom_emo_key": dom_emo,
                "emotion": emo_cfg["name"],
                "emotion_short": emo_cfg["short"],
                "emotion_hex": emo_cfg["hex"],
                "emo_pcts": emo_pcts,
                "counts": dict(counts)
            })

        comm_series[c] = {
            "color": c_color,
            "topic_full": topic_full,
            "topic_short": topic_short,
            "global_pct": global_pct,
            "y_share": y_share,
            "y_delta_pp": y_delta_pp,
            "marker_emo_colors": marker_emo_colors,
            "marker_sizes_share": marker_sizes_share,
            "texts_share": texts_share,
            "texts_delta": texts_delta,
            "hovers": hovers,
            "cds": cds
        }

    # BLOQUE 0: LÍNEAS DE CUOTA RELATIVA (% DEL TRIMESTRE)
    for c in comms_by_vol:
        s = comm_series[c]
        is_vis = (c in default_Selected)
        fig.add_trace(go.Scatter(
            x=multi_x,
            y=s["y_share"],
            mode="lines+markers+text",
            text=s["texts_share"],
            textposition=["top center" if i < len(periods) - 1 else "middle right" for i in range(len(periods))],
            textfont=dict(color=s["color"], size=13.5, family="Segoe UI, Arial, sans-serif"),
            cliponaxis=False,
            name=f"C{c} ({s['global_pct']:.1f}%): {s['topic_short']}",
            legendgroup=f"C{c}",
            line=dict(color=s["color"], width=3.0, shape="spline", smoothing=0.35),
            marker=dict(
                size=s["marker_sizes_share"],
                color=s["marker_emo_colors"],
                line=dict(width=2.2, color=s["color"]),
                opacity=0.96
            ),
            hovertext=s["hovers"],
            customdata=s["cds"],
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(bgcolor="#ffffff", bordercolor=s["color"], font=dict(color="#0f172a", size=11.5)),
            visible=is_vis
        ))

    # BLOQUE 1: LÍNEAS DE CAMBIO ESTRUCTURAL ACUMULADO (Δ PUNTOS PORCENTUALES VS 2019Q1)
    for c in comms_by_vol:
        s = comm_series[c]
        fig.add_trace(go.Scatter(
            x=multi_x,
            y=s["y_delta_pp"],
            mode="lines+markers+text",
            text=s["texts_delta"],
            textposition=["top center" if i < len(periods) - 1 else "middle right" for i in range(len(periods))],
            textfont=dict(color=s["color"], size=13.5, family="Segoe UI, Arial, sans-serif"),
            cliponaxis=False,
            name=f"C{c} ({s['global_pct']:.1f}%): {s['topic_short']}",
            legendgroup=f"C{c}",
            line=dict(color=s["color"], width=3.0, shape="spline", smoothing=0.35),
            marker=dict(
                size=s["marker_sizes_share"],
                color=s["marker_emo_colors"],
                line=dict(width=2.2, color=s["color"]),
                opacity=0.96
            ),
            hovertext=s["hovers"],
            customdata=s["cds"],
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(bgcolor="#ffffff", bordercolor=s["color"], font=dict(color="#0f172a", size=11.5)),
            visible=False
        ))

    shapes = []
    annotations = []
    for mp, mnote in missing_notes.items():
        if mp in periods:
            idx_mp = periods.index(mp)
            shapes.append(dict(
                type="line", x0=idx_mp, x1=idx_mp, y0=0, y1=1, yref="paper",
                line=dict(color="rgba(217, 119, 6, 0.55)", width=1.5, dash="dot")
            ))
            annotations.append(dict(
                x=idx_mp, y=1.02, yref="paper", text=f"⚠️ {mnote} (Cuota % neutraliza sesgo)",
                showarrow=False, font=dict(size=10, color="#b45309"), bgcolor="rgba(254, 243, 199, 0.95)"
            ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>4. Tendencias de Cuota y Cambio Estructural</b> — r/{subreddit_name}<br>"
                f"<sup>Línea = Trayectoria de participación (%) | Color interior del nodo = Emoción Dominante | Eje X multiíndice agrupado por año y trimestre</sup>"
            ),
            x=0.02, y=0.97,
            font=dict(size=16, color="#0f172a", family="Segoe UI, Arial, sans-serif")
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, Arial, sans-serif"),
        height=720,
        hovermode="closest",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            title=dict(text="<b>Periodo Cronológico (Año y Trimestre)</b>", font=dict(size=15, color="#0f172a", family="Segoe UI, sans-serif")),
            tickfont=dict(size=13.0, color="#0f172a", family="Segoe UI, sans-serif"),
            type="multicategory",
            showgrid=True,
            gridcolor="rgba(203, 213, 225, 0.7)",
            zeroline=False,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(2, 132, 199, 0.55)", spikethickness=1.3, spikedash="dot"
        ),
        yaxis=dict(
            title=dict(text="<b>Cuota de Participación en el Trimestre (%)</b>", font=dict(size=15, color="#0f172a", family="Segoe UI, sans-serif")),
            tickfont=dict(size=13.5, color="#0f172a", family="Segoe UI, sans-serif"),
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(203, 213, 225, 0.7)",
            zeroline=True,
            zerolinecolor="rgba(100, 116, 139, 0.45)"
        ),
        legend=dict(
            title=dict(text="Comunidades (Clic en barra superior)", font=dict(size=11, color="#0f172a")),
            orientation="v", y=1, yanchor="top", x=1.02, xanchor="left", font=dict(size=10.5, color="#0f172a")
        ),
        margin=dict(l=65, r=40, t=90, b=55)
    )
    return fig

build_quota_trends_figure = build_community_period_bubble_grid


def build_affective_climate_figure(G_full, community_topics, subreddit_name="Subreddit"):
    """
    Construye la Pestaña 5: Clima Emocional por Trimestre (Analítica Afectiva) con:
      1. Matriz de Emoción Predominante (Vista principal):
         - Eje X: Trimestres cronológicos (2019Q1 – 2022Q3).
         - Eje Y: Macro-Comunidades temáticas ordenadas de mayor a menor volumen (C0, C1, etc.).
         - Celdas: Cada celda es un bloque (square size 44) con el color oficial de la emoción predominante.
         - Etiqueta interna clara: Muestra la emoción y % de dominancia (ej. Miedo 53%, Tristeza 64%).
         - Tooltip: Desglose completo de las 7 emociones y volumen de comentarios en ese trimestre.
      2. Curvas de Evolución Emocional (%):
         - Trayectorias temporales de la emoción dominante o aislar una emoción específica con dropdown.
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    periods = sort_periods_chronologically(G_full.vs["period"])
    if not periods or not all_comms:
        return go.Figure()

    comm_totals = Counter([v["community"] for v in G_full.vs if v["community"] != -1])
    total_all = max(1, sum(comm_totals.values()))
    comms_by_vol = sorted(all_comms, key=lambda c: comm_totals[c], reverse=True)

    cell_vols = defaultdict(int)
    cell_emo_counts = defaultdict(lambda: Counter())
    for v in G_full.vs:
        c, p, f = v["community"], v["period"], v["factor"]
        if c != -1 and p:
            cell_vols[(c, p)] += 1
            if f:
                cell_emo_counts[(c, p)][f] += 1

    quarter_totals = defaultdict(int)
    for (c, p), v_count in cell_vols.items():
        quarter_totals[p] += v_count

    default_Selected = set(KEY_FINDINGS_COMMS_BY_SUB.get(subreddit_name, comms_by_vol[:5]))

    fig = go.Figure()
    comm_labels_ordered = []

    # BLOQUE 0: MATRIZ DE EMOCIÓN PREDOMINANTE
    for c in comms_by_vol:
        c_color = get_community_color(c)
        topic_full = get_community_topic(community_topics, c, max_words=5).replace("_", " ")
        topic_short = get_community_topic(community_topics, c, max_words=2).replace("_", " ")
        c_label = f"C{c} ({topic_short})"
        comm_labels_ordered.append(c_label)

        x_pts, y_pts, txt_pts, col_pts, hov_pts, cds = [], [], [], [], [], []

        for p in periods:
            vol = cell_vols.get((c, p), 0)
            q_tot = quarter_totals.get(p, 0)
            share_p = (vol / q_tot * 100.0) if q_tot > 0 else 0.0

            counts = cell_emo_counts.get((c, p), Counter())
            if counts and vol > 0:
                dom_emo, dom_cnt = counts.most_common(1)[0]
                dom_pct = (dom_cnt / vol) * 100.0
            else:
                dom_emo, dom_cnt, dom_pct = "neutral", 0, 0.0

            emo_cfg = EMOTION_PALETTE.get(dom_emo, EMOTION_PALETTE["neutral"])
            emo_pcts = {
                k: round((counts.get(k, 0) / vol * 100.0) if vol > 0 else 0.0, 1)
                for k in FACTOR_COLS
            }

            x_pts.append(p)
            y_pts.append(c_label)
            txt_pts.append(f"<b>{emo_cfg['short']}</b><br>{dom_pct:.0f}%")
            col_pts.append(emo_cfg["hex"])

            emo_items = sorted(emo_pcts.items(), key=lambda x: x[1], reverse=True)
            emo_lines = "<br>".join([f"  ● <b>{k.capitalize()}:</b> {v:.1f}%" for k, v in emo_items if v > 0])

            hov_str = (
                f"<b>Macro-Comunidad C{c}: {topic_full}</b><br>"
                f"<b>Trimestre:</b> {p} &nbsp;|&nbsp; <b>Volumen:</b> {vol:,} comentarios ({share_p:.1f}% del trimestre)<br>"
                f"<b>Emoción Predominante:</b> <span style='color:{emo_cfg['hex']}; font-size:1.05em;'>● <b>{emo_cfg['name']} ({dom_pct:.1f}%)</b></span><br>"
                f"<span style='color:#cbd5e1;'>─────────────────────────────</span><br>"
                f"<b>Desglose Afectivo Completo:</b><br>{emo_lines}"
            )
            hov_pts.append(hov_str)

            cds.append({
                "type": "emotion_matrix_point",
                "comm": int(c),
                "topic": topic_full,
                "topic_short": topic_short,
                "color": c_color,
                "period": str(p),
                "vol": int(vol),
                "pct_quarter": round(share_p, 2),
                "pct_dom": round(dom_pct, 1),
                "dom_emo_key": dom_emo,
                "emotion": emo_cfg["name"],
                "emotion_short": emo_cfg["short"],
                "emotion_hex": emo_cfg["hex"],
                "emo_pcts": emo_pcts,
                "counts": dict(counts)
            })

        is_vis = (c in default_Selected)
        fig.add_trace(go.Scatter(
            x=x_pts,
            y=y_pts,
            mode="markers+text",
            text=txt_pts,
            textposition="middle center",
            textfont=dict(size=11.5, color="#ffffff", family="Segoe UI, Arial, sans-serif"),
            marker=dict(
                symbol="square",
                size=44,
                color=col_pts,
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.95
            ),
            hovertext=hov_pts,
            customdata=cds,
            hovertemplate="%{hovertext}<extra></extra>",
            name=c_label,
            legendgroup=f"C{c}",
            visible=is_vis
        ))

    # BLOQUE 1: CURVAS DE EVOLUCIÓN EMOCIONAL (%)
    for c in comms_by_vol:
        c_color = get_community_color(c)
        topic_full = get_community_topic(community_topics, c, max_words=5).replace("_", " ")
        topic_short = get_community_topic(community_topics, c, max_words=2).replace("_", " ")
        global_pct = (comm_totals[c] / total_all) * 100.0

        y_dom_emo_pct = []
        marker_emo_colors = []
        texts_emo = []
        hovers_curve = []
        cds_curve = []

        for idx_p, p in enumerate(periods):
            vol = cell_vols.get((c, p), 0)
            counts = cell_emo_counts.get((c, p), Counter())
            if counts and vol > 0:
                dom_emo, dom_cnt = counts.most_common(1)[0]
                dom_pct = (dom_cnt / vol) * 100.0
            else:
                dom_emo, dom_cnt, dom_pct = "neutral", 0, 0.0

            emo_cfg = EMOTION_PALETTE.get(dom_emo, EMOTION_PALETTE["neutral"])
            emo_pcts = {
                k: round((counts.get(k, 0) / vol * 100.0) if vol > 0 else 0.0, 1)
                for k in FACTOR_COLS
            }

            y_dom_emo_pct.append(round(dom_pct, 1))
            marker_emo_colors.append(emo_cfg["hex"])

            is_last_q = (idx_p == len(periods) - 1)
            if is_last_q:
                texts_emo.append(f"  <b>C{c}</b>: {emo_cfg['short']} ({dom_pct:.1f}%)")
            elif p in {"2019Q1", "2020Q2", "2021Q4"}:
                texts_emo.append(f"{dom_pct:.0f}%")
            else:
                texts_emo.append("")

            emo_items = sorted(emo_pcts.items(), key=lambda x: x[1], reverse=True)
            emo_lines = "<br>".join([f"  ● <b>{k.capitalize()}:</b> {v:.1f}%" for k, v in emo_items if v > 0])
            hover_str = (
                f"<b>Macro-Comunidad C{c} ({global_pct:.1f}%): {topic_full}</b><br>"
                f"<b>Trimestre:</b> {p} &nbsp;|&nbsp; <b>Volumen:</b> {vol:,} comentarios<br>"
                f"<b>Emoción Dominante:</b> <span style='color:{emo_cfg['hex']}'>● {emo_cfg['name']} ({dom_pct:.1f}%)</span><br>"
                f"<span style='color:#94a3b8'>────────────────────────────────────</span><br>"
                f"<b>Desglose Afectivo:</b><br>{emo_lines}"
            )
            hovers_curve.append(hover_str)

            cds_curve.append({
                "type": "emotion_curve_point",
                "comm": int(c),
                "topic": topic_full,
                "topic_short": topic_short,
                "color": c_color,
                "period": str(p),
                "vol": int(vol),
                "pct_dom": round(dom_pct, 1),
                "dom_emo_key": dom_emo,
                "emotion": emo_cfg["name"],
                "emotion_short": emo_cfg["short"],
                "emotion_hex": emo_cfg["hex"],
                "emo_pcts": emo_pcts,
                "counts": dict(counts)
            })

        fig.add_trace(go.Scatter(
            x=periods,
            y=y_dom_emo_pct,
            mode="lines+markers+text",
            text=texts_emo,
            textposition=["top center" if i < len(periods) - 1 else "middle right" for i in range(len(periods))],
            textfont=dict(color=c_color, size=13.5, family="Segoe UI, Arial, sans-serif"),
            cliponaxis=False,
            name=f"C{c} ({global_pct:.1f}%): {topic_short}",
            legendgroup=f"C{c}",
            line=dict(color=c_color, width=2.8, shape="spline", smoothing=0.3),
            marker=dict(
                size=10.0,
                color=marker_emo_colors,
                line=dict(width=2.2, color=c_color),
                opacity=0.96
            ),
            hovertext=hovers_curve,
            customdata=cds_curve,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(bgcolor="#ffffff", bordercolor=c_color, font=dict(color="#0f172a", size=11.5)),
            visible=False
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>5. Clima Emocional por Trimestre (Emoción Predominante)</b> — r/{subreddit_name}<br><sup>Cada celda indica la emoción dominante en el trimestre y su % de dominancia | Paleta oficial de colores</sup>",
            x=0.02, y=0.97,
            font=dict(size=16, color="#0f172a", family="Segoe UI, Arial, sans-serif")
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, Arial, sans-serif"),
        height=740,
        hovermode="closest",
        xaxis=dict(
            title=dict(text="<b>Trimestre Cronológico (2019Q1 – 2022Q3)</b>", font=dict(size=15, color="#0f172a", family="Segoe UI, sans-serif")),
            tickfont=dict(size=13.5, color="#0f172a", family="Segoe UI, sans-serif"),
            type="category",
            categoryorder="array",
            categoryarray=periods,
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="<b>Macro-Comunidades Temáticas</b>", font=dict(size=15, color="#0f172a", family="Segoe UI, sans-serif")),
            tickfont=dict(size=13.5, color="#0f172a", family="Segoe UI, sans-serif"),
            type="category",
            categoryorder="array",
            categoryarray=comm_labels_ordered[::-1],
            showgrid=False,
            automargin=True
        ),
        margin=dict(l=340, r=40, t=90, b=60)
    )
    return fig



# ==============================================================================
# VOLUMEN LONGITUDINAL CON FILTROS ANTI-SPAGHETTI Y ETIQUETAS DIRECTAS
# ==============================================================================

def build_longitudinal_volume_figure(G_full, community_topics, subreddit_name="Subreddit"):
    """
    Construye la Pestaña 5 (V3.0): Evolución Temporal del Volumen por Macro-Comunidad con:
      - Etiquetas directas al final derecho de cada línea (`C#: tópico`) para lectura inmediata.
      - Botones rápidos Anti-Spaghetti:
        1) Top 5 Comunidades Más Grandes (Volumen Absoluto)
        2) Todas las Comunidades (Volumen Absoluto)
        3) % de Cambio Trimestral (Todas)
        4) Top 5 Mayor Crecimiento (% de Cambio)
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    periods = sort_periods_chronologically(G_full.vs["period"])
    if not periods or not all_comms:
        return go.Figure()

    vol_comm_period = defaultdict(lambda: defaultdict(int))
    total_period = defaultdict(int)
    comm_totals = Counter()

    for v in G_full.vs:
        c = v["community"]
        p = v["period"]
        if c != -1 and p:
            vol_comm_period[c][p] += 1
            total_period[p] += 1
            comm_totals[c] += 1

    top5_vol_comms = set(c for c, _ in comm_totals.most_common(5))

    fig = go.Figure()
    growth_scores = {}

    # 1. TRAZAS DE VOLUMEN ABSOLUTO POR COMUNIDAD
    for c in all_comms:
        comm_y_abs = [vol_comm_period[c][p] for p in periods]
        comm_color = get_community_color(c)
        topic_words = get_community_topic(community_topics, c, max_words=4)
        short_topic = get_community_topic(community_topics, c, max_words=2)

        # Etiqueta directa solo en el último punto para identificar la línea sin usar la leyenda
        end_texts = [""] * (len(periods) - 1) + [f"  <b>C{c}</b> ({short_topic})"]

        hover_abs = [
            f"<b>Macro-Comunidad C{c}: {topic_words}</b><br>"
            f"<b>Periodo:</b> {p}<br>"
            f"<b>Volumen Absoluto:</b> {cnt:,} comentarios<br>" +
            (f"<b>Participación en {p}:</b> {(cnt / total_period[p] * 100):.1f}%" if total_period[p] > 0 else "0%")
            for p, cnt in zip(periods, comm_y_abs)
        ]

        cds = [{
            "type": "volume_point",
            "comm": int(c),
            "topic": topic_words,
            "period": str(p),
            "vol": int(cnt),
            "color": comm_color
        } for p, cnt in zip(periods, comm_y_abs)]

        fig.add_trace(go.Scatter(
            x=periods, y=comm_y_abs, mode="lines+markers+text",
            text=end_texts,
            textposition="middle right",
            textfont=dict(color=comm_color, size=13.5, family="Segoe UI, Arial, sans-serif"),
            cliponaxis=False,
            name=f"C{c}: {short_topic}",
            legendgroup=f"C{c}",
            line=dict(color=comm_color, width=2.6),
            marker=dict(size=7, color=comm_color, line=dict(width=1.5, color=comm_color)),
            hovertext=hover_abs,
            customdata=cds,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(
                bgcolor="#0f172a",
                bordercolor=comm_color,
                font=dict(color="#f1f5f9", size=11.5, family="Segoe UI, Arial, sans-serif")
            ),
            visible=True
        ))

    # 2. TRAZAS DE % DE CAMBIO RESPECTO AL TRIMESTRE ANTERIOR
    for c in all_comms:
        comm_y_abs = [vol_comm_period[c][p] for p in periods]
        comm_color = get_community_color(c)
        topic_words = get_community_topic(community_topics, c, max_words=4)
        short_topic = get_community_topic(community_topics, c, max_words=2)

        comm_y_pct_change = []
        hover_pct = []
        for i, (p, cnt) in enumerate(zip(periods, comm_y_abs)):
            if i == 0:
                comm_y_pct_change.append(0.0)
                hover_pct.append(
                    f"<b>Macro-Comunidad C{c}: {topic_words}</b><br>"
                    f"<b>Periodo:</b> {p} (Línea Base)<br>"
                    f"<b>Volumen:</b> {cnt:,} comentarios<br>"
                    f"<b>% de Cambio:</b> 0.0% (Inicio)"
                )
            else:
                prev_cnt = comm_y_abs[i - 1]
                p_prev = periods[i - 1]
                if prev_cnt > 0:
                    pct_chg = ((cnt - prev_cnt) / prev_cnt) * 100.0
                else:
                    pct_chg = 0.0 if cnt == 0 else 100.0
                comm_y_pct_change.append(round(pct_chg, 2))
                sign = "+" if pct_chg > 0 else ""
                hover_pct.append(
                    f"<b>Macro-Comunidad C{c}: {topic_words}</b><br>"
                    f"<b>Periodo:</b> {p}<br>"
                    f"<b>Volumen Actual:</b> {cnt:,} &nbsp;|&nbsp; <b>Anterior ({p_prev}):</b> {prev_cnt:,}<br>"
                    f"<b>% de Cambio Trimestral:</b> {sign}{pct_chg:.1f}%"
                )

        growth_scores[c] = float(np.mean(np.abs(comm_y_pct_change[1:]))) if len(comm_y_pct_change) > 1 else 0.0
        end_texts_pct = [""] * (len(periods) - 1) + [f"  <b>C{c}</b> ({comm_y_pct_change[-1]:+.1f}%)"]

        fig.add_trace(go.Scatter(
            x=periods, y=comm_y_pct_change, mode="lines+markers+text",
            text=end_texts_pct,
            textposition="middle right",
            textfont=dict(color=comm_color, size=13.5, family="Segoe UI, Arial, sans-serif"),
            cliponaxis=False,
            name=f"C{c}: {short_topic}",
            legendgroup=f"C{c}",
            showlegend=True,
            line=dict(color=comm_color, width=2.5),
            marker=dict(size=7, color=comm_color, line=dict(width=1.5, color=comm_color)),
            hovertext=hover_pct,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(
                bgcolor="#0f172a",
                bordercolor=comm_color,
                font=dict(color="#f1f5f9", size=11.5, family="Segoe UI, Arial, sans-serif")
            ),
            visible=False
        ))

    top5_chg_comms = set(c for c, _ in sorted(growth_scores.items(), key=lambda x: x[1], reverse=True)[:5])

    n_c = len(all_comms)
    vis_all_abs = [True] * n_c + [False] * n_c
    vis_top5_abs = [(c in top5_vol_comms) for c in all_comms] + [False] * n_c
    vis_all_pct = [False] * n_c + [True] * n_c
    vis_top5_pct = [False] * n_c + [(c in top5_chg_comms) for c in all_comms]

    buttons = [
        dict(
            label="📊 Todas (Volumen)",
            method="update",
            args=[
                {"visible": vis_all_abs},
                {
                    "title.text": (
                        f"<b>5. Evolución de Volumen por Macro-Comunidad (Todas)</b> — r/{subreddit_name}<br>"
                        f"<sup>Etiquetas directas al final de cada serie | Haz clic en cualquier comunidad de la lista derecha para iluminarla (Spotlight)</sup>"
                    ),
                    "yaxis.title.text": "<b>Número de Comentarios</b>"
                }
            ]
        ),
        dict(
            label="🏆 Top 5 Más Grandes",
            method="update",
            args=[
                {"visible": vis_top5_abs},
                {
                    "title.text": (
                        f"<b>5. Evolución de Volumen (Top 5 Comunidades Más Grandes)</b> — r/{subreddit_name}<br>"
                        f"<sup>Vista limpia sin solapamiento con las 5 comunidades de mayor volumen acumulado</sup>"
                    ),
                    "yaxis.title.text": "<b>Número de Comentarios</b>"
                }
            ]
        ),
        dict(
            label="📈 % Cambio Trimestral (Todas)",
            method="update",
            args=[
                {"visible": vis_all_pct},
                {
                    "title.text": (
                        f"<b>5. Variación Porcentual Trimestral por Comunidad (% Cambio)</b> — r/{subreddit_name}<br>"
                        f"<sup>Tasa de cambio respecto al trimestre inmediatamente anterior ((V_t - V_(t-1)) / V_(t-1))</sup>"
                    ),
                    "yaxis.title.text": "<b>% de Cambio vs Trimestre Anterior</b>"
                }
            ]
        ),
        dict(
            label="🚀 Top 5 Mayor Variación (%)",
            method="update",
            args=[
                {"visible": vis_top5_pct},
                {
                    "title.text": (
                        f"<b>5. Variación Porcentual (Top 5 Comunidades Más Dinámicas)</b> — r/{subreddit_name}<br>"
                        f"<sup>Las 5 macro-comunidades con mayor tasa de cambio trimestral promedio</sup>"
                    ),
                    "yaxis.title.text": "<b>% de Cambio vs Trimestre Anterior</b>"
                }
            ]
        )
    ]

    fig.update_layout(
        title=dict(
            text=f"<b>5. Evolución de Volumen por Macro-Comunidad (Todas)</b> — r/{subreddit_name}<br>"
                 f"<sup>Etiquetas directas al final de cada serie | Haz clic en cualquier comunidad de la lista derecha para iluminarla (Spotlight)</sup>",
            x=0.02, y=0.97
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, Arial, sans-serif"),
        hovermode="closest",
        margin=dict(l=45, r=230, t=95, b=45),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=1.0, y=1.11,
            xanchor="right", yanchor="top",
            buttons=buttons,
            bgcolor="#f8fafc",
            bordercolor="#cbd5e1",
            font=dict(color="#0f172a", size=10.5),
            active=0
        )],
        xaxis=dict(
            title="<b>Trimestre Cronológico</b>",
            type="category",
            categoryorder="array",
            categoryarray=periods,
            tickmode="array", tickvals=periods, ticktext=periods,
            showgrid=True, gridcolor="rgba(203, 213, 225, 0.7)",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(2, 132, 199, 0.55)",
            spikethickness=1.5, spikedash="dot"
        ),
        yaxis=dict(
            title=dict(text="<b>Número de Comentarios</b>"),
            showgrid=True, gridcolor="rgba(203, 213, 225, 0.7)",
            zeroline=True, zerolinecolor="rgba(100, 116, 139, 0.45)",
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(2, 132, 199, 0.55)",
            spikethickness=1.5, spikedash="dot"
        ),
        legend=dict(
            title="Macro-Comunidades",
            orientation="v", y=1, yanchor="top", x=1.12, xanchor="left",
            font=dict(size=9.5, color="#0f172a")
        )
    )
    return fig


# ==============================================================================
# 6. EXTRACTOR DE METADATOS ANALÍTICOS PARA EL PANEL LATERAL Y KPIs (SIN HOVER)
# ==============================================================================

def build_subreddit_summary_meta(G_full, community_topics, subreddit_name="Subreddit"):
    """
    Extrae toda la información estructurada para las tarjetas KPI superiores,
    el Directorio de Comunidades, el Ranking de Puentes (global y por trimestre),
    las barras de distribución emocional y el Lector de Comentarios.
    """
    ensure_graph_attributes(G_full, subreddit_name=subreddit_name)
    community_topics = ensure_community_topics(G_full, community_topics)

    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    periods = sort_periods_chronologically(G_full.vs["period"]) if "period" in G_full.vertex_attributes() else []

    total_nodes = G_full.vcount()
    total_edges = G_full.ecount()

    comm_counts = Counter()
    comm_emos = defaultdict(lambda: Counter())
    comm_period_counts = defaultdict(lambda: defaultdict(int))
    global_emos = Counter()

    for v in G_full.vs:
        c = v["community"]
        p = v["period"] if "period" in G_full.vertex_attributes() else "N/A"
        f = v["factor"] if "factor" in G_full.vertex_attributes() else "neutral"
        if c != -1:
            comm_counts[c] += 1
            if f:
                comm_emos[c][f] += 1
                global_emos[f] += 1
            if p:
                comm_period_counts[c][p] += 1

    total_valid = max(1, sum(comm_counts.values()))

    # 1. Directorio de Comunidades (ordenado por volumen descendente)
    communities_list = []
    for c in sorted(all_comms, key=lambda x: comm_counts[x], reverse=True):
        vol = comm_counts[c]
        pct = (vol / total_valid) * 100.0
        dom_e, dom_cnt = comm_emos[c].most_common(1)[0] if comm_emos[c] else ("neutral", 0)
        dom_pct = (dom_cnt / vol * 100.0) if vol > 0 else 0.0

        emo_breakdown = {}
        for emo_k in FACTOR_COLS:
            cnt_e = comm_emos[c].get(emo_k, 0)
            emo_breakdown[emo_k] = {
                "count": int(cnt_e),
                "pct": round((cnt_e / vol * 100.0) if vol > 0 else 0.0, 1)
            }

        # Crecimiento entre primer y último trimestre
        q_series = [comm_period_counts[c].get(p, 0) for p in periods]
        if len(q_series) >= 2 and q_series[-2] > 0:
            last_q_chg = ((q_series[-1] - q_series[-2]) / q_series[-2]) * 100.0
        else:
            last_q_chg = 0.0

        communities_list.append({
            "comm": int(c),
            "color": get_community_color(c),
            "topic_full": get_community_topic(community_topics, c, max_words=6),
            "topic_short": get_community_topic(community_topics, c, max_words=3),
            "vol": int(vol),
            "pct": round(pct, 2),
            "dom_emo": dom_e,
            "dom_emo_label": EMOTION_PALETTE.get(dom_e, {}).get("name", dom_e),
            "dom_emo_short": EMOTION_PALETTE.get(dom_e, {}).get("short", dom_e),
            "dom_emo_hex": EMOTION_PALETTE.get(dom_e, {}).get("hex", "#94a3b8"),
            "dom_emo_pct": round(dom_pct, 1),
            "emotions": emo_breakdown,
            "last_q_chg": round(last_q_chg, 1),
            "q_series": q_series
        })

    # 2. Ranking de Puentes Global
    G_macro_global = collapse_macro_graph(G_full, community_topics)
    global_bridges = []
    for e in G_macro_global.es:
        c1 = int(G_macro_global.vs[e.source]["community"])
        c2 = int(G_macro_global.vs[e.target]["community"])
        emo = e["dominant_emotion"]
        cfg = EMOTION_PALETTE.get(emo, {"name": "Mixto", "short": "Mixto", "hex": "#94a3b8"})
        global_bridges.append({
            "c1": c1,
            "c2": c2,
            "t1": get_community_topic(community_topics, c1, max_words=2),
            "t2": get_community_topic(community_topics, c2, max_words=2),
            "cut_density": float(e["cut_density"]),
            "count": int(e["count"]),
            "pct_diff_val": round(float(e["pct_diff_cut_density"]), 2),
            "pct_diff_str": str(e["pct_diff_str"]),
            "emotion": cfg["short"],
            "emotion_hex": cfg["hex"]
        })
    global_bridges.sort(key=lambda b: b["cut_density"], reverse=True)

    # 3. Ranking de Puentes por Trimestre (para sincronizar con el slider de la Pestaña 3)
    temporal_bridges = {}
    period_totals = {}
    prev_cut_map = {}
    prev_lbl = None
    period_nodes_map = {p: [v.index for v in G_full.vs if v["period"] == p] for p in periods}

    for p in periods:
        nodes_p = period_nodes_map.get(p, [])
        period_totals[p] = len(nodes_p)
        if len(nodes_p) < 15:
            continue
        G_sub_p = G_full.subgraph(nodes_p)
        G_mp = collapse_macro_graph(G_sub_p, community_topics, prev_cut_densities=prev_cut_map, prev_period_label=prev_lbl)
        curr_map = {}
        p_list = []
        for e in G_mp.es:
            c1 = int(G_mp.vs[e.source]["community"])
            c2 = int(G_mp.vs[e.target]["community"])
            pair = tuple(sorted([c1, c2]))
            curr_map[pair] = float(e["cut_density"])
            emo = e["dominant_emotion"]
            cfg = EMOTION_PALETTE.get(emo, {"name": "Mixto", "short": "Mixto", "hex": "#94a3b8"})
            p_list.append({
                "c1": c1,
                "c2": c2,
                "t1": get_community_topic(community_topics, c1, max_words=2),
                "t2": get_community_topic(community_topics, c2, max_words=2),
                "cut_density": float(e["cut_density"]),
                "count": int(e["count"]),
                "pct_diff_val": round(float(e["pct_diff_cut_density"]), 2),
                "pct_diff_str": str(e["pct_diff_str"]),
                "emotion": cfg["short"],
                "emotion_hex": cfg["hex"]
            })
        p_list.sort(key=lambda b: b["cut_density"], reverse=True)
        temporal_bridges[p] = p_list
        prev_cut_map = curr_map
        prev_lbl = p

    # 4. Comentarios destacados por comunidad (Top 6 por comunidad para el Lector lateral)
    degrees = G_full.degree()
    sample_comments_by_comm = {}
    for c in all_comms:
        vs_c = [v.index for v in G_full.vs if v["community"] == c]
        raw_words = get_community_topic(community_topics, c, max_words=6)
        kw_set = set([w.lower().strip() for w in raw_words.replace(",", " ").split() if len(w) > 2])

        scored = []
        for idx in vs_c[:4000]:  # Evaluación rápida sobre hasta 4,000 nodos de la comunidad
            txt = str(G_full.vs[idx]["comentario_completo"] or "").strip()
            if len(txt) < 35:
                continue
            txt_low = txt.lower()
            kw_hits = sum(1 for kw in kw_set if kw in txt_low)
            sc = degrees[idx] * 0.5 + kw_hits * 18.0 + (2.0 if 60 <= len(txt) <= 450 else 0.5)
            scored.append((sc, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_comments = []
        for _, idx in scored[:6]:
            v = G_full.vs[idx]
            emo = v["factor"] if "factor" in G_full.vertex_attributes() else "neutral"
            txt = str(v["comentario_completo"] or "").strip()
            if len(txt) > 150:
                txt = txt[:150] + "..."
            top_comments.append({
                "id": int(idx),
                "comm": int(c),
                "period": str(v["period"]) if "period" in G_full.vertex_attributes() else "N/A",
                "emotion": EMOTION_PALETTE.get(emo, {}).get("short", emo),
                "emotion_hex": EMOTION_PALETTE.get(emo, {}).get("hex", "#94a3b8"),
                "degree": int(degrees[idx]),
                "text": txt
            })
        sample_comments_by_comm[str(c)] = top_comments

    # Emociones globales
    top_emos = global_emos.most_common()
    dom_global_emo = top_emos[0][0] if top_emos else "neutral"
    dom_global_pct = (top_emos[0][1] / total_valid * 100.0) if top_emos else 0.0
    sec_global_emo = top_emos[1][0] if len(top_emos) > 1 else "neutral"
    sec_global_pct = (top_emos[1][1] / total_valid * 100.0) if len(top_emos) > 1 else 0.0

    top_comm = communities_list[0] if communities_list else {}
    top_bridge = global_bridges[0] if global_bridges else {}

    return {
        "subreddit": subreddit_name,
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "n_communities": len(all_comms),
        "periods": periods,
        "period_totals": period_totals,
        "kpi": {
            "top_comm_id": top_comm.get("comm", 0),
            "top_comm_topic": top_comm.get("topic_short", "N/A"),
            "top_comm_pct": top_comm.get("pct", 0.0),
            "top_comm_vol": top_comm.get("vol", 0),
            "dom_emo_label": EMOTION_PALETTE.get(dom_global_emo, {}).get("short", dom_global_emo),
            "dom_emo_hex": EMOTION_PALETTE.get(dom_global_emo, {}).get("hex", "#38bdf8"),
            "dom_emo_pct": round(dom_global_pct, 1),
            "sec_emo_label": EMOTION_PALETTE.get(sec_global_emo, {}).get("short", sec_global_emo),
            "sec_emo_pct": round(sec_global_pct, 1),
            "top_bridge_pair": f"C{top_bridge.get('c1', 0)} ↔ C{top_bridge.get('c2', 0)}",
            "top_bridge_topics": f"{top_bridge.get('t1', '')} / {top_bridge.get('t2', '')}",
            "top_bridge_rho": top_bridge.get("cut_density", 0.0),
            "top_bridge_count": top_bridge.get("count", 0)
        },
        "communities": communities_list,
        "global_bridges": global_bridges,
        "temporal_bridges": temporal_bridges,
        "sample_comments": sample_comments_by_comm
    }
