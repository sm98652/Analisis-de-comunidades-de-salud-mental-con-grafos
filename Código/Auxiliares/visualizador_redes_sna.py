# -*- coding: utf-8 -*-
"""
================================================================================
MÓDULO DE ANÁLISIS DE REDES COMPLEJAS (SNA) Y VISUALIZACIÓN MULTIESCALA (V2.1)
Redes Semántico-Afectivas en Salud Mental (Reddit / RMHD)
Subreddits: Anxiety, SuicideWatch, depression
================================================================================
Arquitectura de 5 Pestañas Analíticas:
  1. Micro Orgánico: Disposición física Fruchterman-Reingold con separación multinivel,
     muestreo estratificado temáticamente alineado (alta afinidad semántica con el tópico
     de la macrocomunidad + diversidad afectiva balanceada) y comentarios textuales completos.
  2. Macro Agregado: Grafo colapsado por macrocomunidades (K = 6 a 15).
     Nodos = √Volumen. Macroaristas ponderadas por Densidad de Corte Semántica real (ρ).
     *** 100% LIBRE DE ARISTAS ARTIFICIALES DE MST (Árbol de Expansión Mínima). ***
  3. Evolución Temporal del Grafo (Macro Temporal Longitudinal):
     Snapshots por trimestre cronológico (2019Q1 a 2022Q3) con anclaje espacial fijo
     para visualizar qué comunidades se conectaban en cada periodo sin saltos visuales.
  4. Evolución Temporal de Emociones (Matriz de Burbujas):
     Comunidades vs Periodos ordenados cronológicamente (2019Q1 a 2022Q3).
     Toggle entre % de Dominancia Emocional y % de Volumen del Trimestre (SOLO PORCENTAJE).
  5. Evolución de Volumen por Macro-Comunidad (Líneas):
     Gráfico de líneas con conteos longitudinales exactos (2019Q1 a 2022Q3).
     *** EXCLUSIVAMENTE POR COMUNIDAD (SIN TOTAL) Y CON TOGGLE ENTRE VOLUMEN ABSOLUTO Y % DE CAMBIO. ***
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
    "disgust":  {"name": "Disgust (Disgusto)",  "color": "rgba(40, 201, 105, 0.85)", "hex": "#28C969"},
    "joy":      {"name": "Joy (Alegría)",       "color": "rgba(255, 215, 0, 0.85)",  "hex": "#FFD700"},
    "surprise": {"name": "Surprise (Sorpresa)", "color": "rgba(255, 140, 0, 0.85)",  "hex": "#FF8C00"},
    "anger":    {"name": "Anger (Ira/Enfado)",  "color": "rgba(220, 20, 60, 0.85)",  "hex": "#DC143C"},
    "sadness":  {"name": "Sadness (Tristeza)",  "color": "rgba(30, 144, 255, 0.85)", "hex": "#1E90FF"},
    "fear":     {"name": "Fear (Miedo)",        "color": "rgba(148, 0, 211, 0.85)",  "hex": "#9400D3"},
    "neutral":  {"name": "Neutral",             "color": "rgba(194, 194, 194, 0.85)",  "hex": "#C2C2C2"}
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
# 1. PESTAÑA 1: VISTA MICRO ORGÁNICA CON MUESTREO TEMÁTICO-AFECTIVO REFINADO
# ==============================================================================

def sample_representative_subgraph(G, community_topics, max_nodes_total=1400, min_nodes_per_comm=3, random_seed=42):
    """
    Muestreo estratificado representativo enfocado en ALINEACIÓN TEMÁTICA y DIVERSIDAD AFECTIVA:
      1. Cuota proporcional estricta por tamaño de macrocomunidad (Hare-Niemeyer).
      2. Puntuación de afinidad temática: se priorizan comentarios que reflejan directamente
         el núcleo semántico y los términos distintivos del tópico del clúster.
      3. 50% Comentarios núcleo temático (máxima representatividad y claridad con el tópico).
      4. 50% Diversidad emocional balanceada condicionada a representatividad del grupo.
      5. Todos los comentarios conservan su texto íntegro, periodo y etiquetas originales.
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

    # Cuotas proporcionales
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

        # Obtener palabras clave del tópico de esta comunidad
        raw_words = get_community_topic(community_topics, comm, max_words=8)
        kw_set = set([w.lower().strip() for w in raw_words.replace(",", " ").split() if len(w) > 2])

        # Calcular score de representatividad temática por nodo
        scores = []
        for idx in vs:
            deg = degrees[idx]
            txt = str(G.vs[idx]["comentario_completo"] or "").lower()
            # Coincidencia de términos clave del tópico en el texto del comentario
            kw_hits = sum(1 for kw in kw_set if kw in txt)
            # Longitud informativa mínima (evitar comentarios de una sola palabra que no explican el tema)
            len_bonus = 1.0 if len(txt.split()) >= 8 else 0.4
            thematic_score = (deg * 0.4) + (kw_hits * 15.0) + (len_bonus * 2.0)
            scores.append(thematic_score)

        order_thematic = np.argsort(scores)[::-1]
        n_thematic = max(1, int(n_sample * 0.50))
        thematic_selected = [vs[i] for i in order_thematic[:n_thematic]]
        thematic_set = set(thematic_selected)

        # 50% Diversidad emocional entre comentarios con buena alineación temática
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
                    # Ordenar por puntuación de esa emoción específica
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
    G_sub.vs["sub_node_id"] = sampled_nodes
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
    if std_val > 0: coords /= std_val
    return {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}


def build_micro_organic_figure(G_full, community_topics, subreddit_name="Subreddit", max_nodes_total=1400, random_seed=42):
    """Construye la Pestaña 1: Micro Orgánico con muestreo cualitativamente alineado al tópico."""
    ensure_graph_attributes(G_full, subreddit_name=subreddit_name)
    community_topics = ensure_community_topics(G_full, community_topics)
    G_sub = sample_representative_subgraph(G_full, community_topics, max_nodes_total=max_nodes_total, random_seed=random_seed)
    pos = compute_organic_cluster_separated_layout(G_sub, random_seed=random_seed)

    fig = go.Figure()

    # Aristas
    edge_x, edge_y = [], []
    has_w = "weight" in G_sub.edge_attributes()
    weights = G_sub.es["weight"] if has_w else [1.0] * G_sub.ecount()
    w_th = float(np.percentile(weights, 25)) if weights else 0.0

    for edge in G_sub.es:
        w = edge["weight"] if has_w else 1.0
        if w >= w_th:
            u, v = edge.source, edge.target
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.55, color="rgba(165, 180, 205, 0.18)"),
        hoverinfo="skip", name="Enlaces Semánticos", showlegend=True
    ))

    # Nodos por Comunidad
    degrees = G_sub.degree()
    for comm in sorted(list(set(G_sub.vs["community"]))):
        vs_comm = [v for v in G_sub.vs if v["community"] == comm]
        if not vs_comm: continue

        node_x, node_y, sizes, hover_texts, customdata_list = [], [], [], [], []
        comm_color = get_community_color(comm)
        topic_words = get_community_topic(community_topics, comm, max_words=5)

        for v in vs_comm:
            idx = v.index
            node_x.append(pos[idx][0])
            node_y.append(pos[idx][1])
            deg = degrees[idx]
            sizes.append(3.2 + np.sqrt(deg) * 1.7)

            cid = v["orig_index"] if "orig_index" in G_sub.vertex_attributes() else idx
            factor = v["factor"] if "factor" in G_sub.vertex_attributes() else "N/A"
            period = v["period"] if "period" in G_sub.vertex_attributes() else "N/A"

            raw_comment = str(v["comentario_completo"] or "No disponible").strip()
            words_c = raw_comment.split()
            wrapped_lines, cur_line, cur_len = [], [], 0
            for w in words_c:
                if cur_len + len(w) + 1 > 65:
                    wrapped_lines.append(" ".join(cur_line))
                    cur_line = [w]
                    cur_len = len(w)
                else:
                    cur_line.append(w)
                    cur_len += len(w) + 1
            if cur_line: wrapped_lines.append(" ".join(cur_line))

            if len(wrapped_lines) > 7:
                comment_html = "<br>".join([html.escape(l) for l in wrapped_lines[:7]]) + "<br><i>[...]</i>"
            else:
                comment_html = "<br>".join([html.escape(l) for l in wrapped_lines])

            hover_texts.append(
                f"<b>ID Comentario:</b> #{cid}<br>"
                f"<b>Macro-Comunidad:</b> C{comm}<br>"
                f"<b>Tópico Semántico:</b> {topic_words}<br>"
                f"<b>Emoción Detectada:</b> {factor}<br>"
                f"<b>Grado Local:</b> {deg} conexiones<br>"
                f"<b>Periodo:</b> {period}<br>"
                f"<b>Comentario:</b><br>{comment_html}"
            )
            customdata_list.append(raw_comment)

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            name=f"C{comm}: {topic_words[:28]}...",
            marker=dict(size=sizes, color=comm_color, line=dict(width=0.5, color="rgba(255,255,255,0.75)"), opacity=0.9),
            text=hover_texts, customdata=customdata_list, hovertemplate="%{text}<extra></extra>",
            legendgroup=f"C{comm}"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>Pestaña 1: Grafo Micro Orgánico (Nivel Comentarios)</b> — r/{subreddit_name}<br>"
                 f"<sup>Muestreo Representativo (Alineación Temática y Diversidad Afectiva) | {G_sub.vcount():,} comentarios y {G_sub.ecount():,} enlaces reales</sup>",
            x=0.02, y=0.96
        ),
        template="plotly_dark", hovermode="closest",
        margin=dict(l=20, r=260, t=90, b=30),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(title="Macro-Comunidades", orientation="v", y=1, yanchor="top", x=1.02, xanchor="left", font=dict(size=10))
    )
    return fig


# ==============================================================================
# 2. PESTAÑA 2: VISTA MACRO AGREGADA (GRAFO COLAPSADO SIN MST)
# ==============================================================================

def collapse_macro_graph(G, community_topics, pruning_percentile=65, min_cross_edges=3):
    """
    Construye el grafo macro colapsado con Densidad de Corte Semántica (ρ).
    *** COMPLETAMENTE LIBRE DE ARISTAS ARTIFICIALES DE MST (Árbol de Expansión Mínima). ***
    """
    ensure_graph_attributes(G)
    community_topics = ensure_community_topics(G, community_topics)
    all_comms = sorted([c for c in set(G.vs["community"]) if c != -1])
    n_comms = len(all_comms)
    if n_comms == 0: return ig.Graph()

    comm_to_idx = {c: i for i, c in enumerate(all_comms)}
    comm_sizes = {c: 0 for c in all_comms}
    comm_emotions = {c: Counter() for c in all_comms}
    has_f = "factor" in G.vertex_attributes()

    for v in G.vs:
        c = v["community"]
        if c in comm_sizes:
            comm_sizes[c] += 1
            if has_f and v["factor"]: comm_emotions[c][v["factor"]] += 1

    cross_w = defaultdict(float)
    cross_n = defaultdict(int)
    cross_emo = defaultdict(lambda: Counter())

    has_edge_weight = "weight" in G.edge_attributes()
    for e in G.es:
        c_u = G.vs[e.source]["community"]
        c_v = G.vs[e.target]["community"]
        if c_u != -1 and c_v != -1 and c_u != c_v:
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

    G_macro = ig.Graph(n=n_comms)
    G_macro.vs["community"] = all_comms
    G_macro.vs["size"] = [comm_sizes[c] for c in all_comms]
    G_macro.vs["topic_words"] = [get_community_topic(community_topics, c, max_words=5) for c in all_comms]
    G_macro.vs["dominant_emotion"] = [comm_emotions[c].most_common(1)[0][0] if comm_emotions[c] else "N/A" for c in all_comms]

    candidates = []
    for (c1, c2), count in cross_n.items():
        if count < min_cross_edges: continue
        tw = cross_w[(c1, c2)]
        dens = tw / (comm_sizes[c1] * comm_sizes[c2])
        clean_emo = Counter({k: v for k, v in cross_emo[(c1, c2)].items() if k != "NoSharedEmotion"})
        dom_emo = clean_emo.most_common(1)[0][0] if clean_emo else "NoSharedEmotion"
        candidates.append({
            "u": comm_to_idx[c1], "v": comm_to_idx[c2], "c1": c1, "c2": c2,
            "count": count, "total_w": tw, "cut_density": dens, "dom_emo": dom_emo
        })

    if not candidates: return G_macro

    # Poda puramente estadística basada en percentil de densidad de corte (SIN MST)
    all_dens = [c["cut_density"] for c in candidates]
    th = float(np.percentile(all_dens, pruning_percentile)) if all_dens else 0.0

    edges, attrs = [], defaultdict(list)
    for c in candidates:
        if c["cut_density"] >= th:
            edges.append((c["u"], c["v"]))
            attrs["cut_density"].append(c["cut_density"])
            attrs["count"].append(c["count"])
            attrs["total_w"].append(c["total_w"])
            attrs["dominant_emotion"].append(c["dom_emo"])

    G_macro.add_edges(edges)
    for k, vals in attrs.items(): G_macro.es[k] = vals
    return G_macro


def build_macro_figure(G_macro, subreddit_name="Subreddit"):
    """Construye la Pestaña 2: Macro Agregado sin MST."""
    fig = go.Figure()
    if G_macro.vcount() == 0: return fig

    w_attr = "cut_density" if "cut_density" in G_macro.edge_attributes() else None
    if G_macro.ecount() > 0:
        layout = G_macro.layout_fruchterman_reingold(weights=w_attr, niter=500, grid="auto")
    else:
        layout = G_macro.layout_circle()

    coords = np.array(layout.coords)
    coords -= coords.mean(axis=0)
    max_r = np.max(np.abs(coords)) if len(coords) > 0 and np.max(np.abs(coords)) > 0 else 1.0
    pos = {i: (coords[i, 0] / max_r * 10.0, coords[i, 1] / max_r * 10.0) for i in range(len(coords))}

    if G_macro.ecount() > 0:
        dens = np.array(G_macro.es["cut_density"])
        d_min, d_max = np.min(dens), np.max(dens)
        d_range = (d_max - d_min) if (d_max - d_min) > 0 else 1.0
        seen_emo = set()

        for e in G_macro.es:
            u, v = e.source, e.target
            emo = e["dominant_emotion"]
            cfg = EMOTION_PALETTE.get(emo, {"name": emo, "color": "rgba(180, 180, 180, 0.45)"})
            width = 1.6 + ((e["cut_density"] - d_min) / d_range) * 7.0
            total_w = e["total_w"] if "total_w" in G_macro.edge_attributes() else 0.0

            hover = (
                f"<b>Puente Semántico Real: C{G_macro.vs[u]['community']} ↔ C{G_macro.vs[v]['community']}</b><br>"
                f"<b>Emoción Predominante:</b> {cfg['name']}<br>"
                f"<b>Densidad de Corte (ρ):</b> {e['cut_density']:.6e}<br>"
                f"<b>Comentarios Cruzados:</b> {e['count']:,}<br>"
                f"<b>Fuerza Agregada:</b> {total_w:.1f}"
            )

            fig.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0], None], y=[pos[u][1], pos[v][1], None], mode="lines",
                line=dict(width=width, color=cfg["color"]), hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                name=f"Puente: {cfg['name']}", legendgroup=f"Emo_{emo}", showlegend=(emo not in seen_emo)
            ))
            seen_emo.add(emo)

    sizes = np.array(G_macro.vs["size"])
    s_min, s_max = np.min(sizes), np.max(sizes)
    s_range = (np.sqrt(s_max) - np.sqrt(s_min)) if (np.sqrt(s_max) - np.sqrt(s_min)) > 0 else 1.0

    x_n, y_n, s_vis, colors, labels, hovers = [], [], [], [], [], []
    for v in G_macro.vs:
        idx = v.index
        x_n.append(pos[idx][0])
        y_n.append(pos[idx][1])
        c = v["community"]
        s_vis.append(20.0 + ((np.sqrt(v["size"]) - np.sqrt(s_min)) / s_range) * 36.0)
        colors.append(get_community_color(c))
        labels.append(f"<b>C{c}</b>")
        hovers.append(
            f"<b>Macro-Comunidad C{c}</b><br>"
            f"<b>Volumen Total:</b> {v['size']:,} comentarios<br>"
            f"<b>Tópico Semántico:</b> {v['topic_words']}<br>"
            f"<b>Emoción Dominante:</b> {v['dominant_emotion']}<br>"
            f"<b>Puentes Activos:</b> {G_macro.degree(idx)}"
        )

    fig.add_trace(go.Scatter(
        x=x_n, y=y_n, mode="markers+text", text=labels, textposition="middle center",
        textfont=dict(color="white", size=11, family="Arial Black"),
        marker=dict(size=s_vis, color=colors, line=dict(width=2, color="white"), opacity=0.95),
        hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>", showlegend=False
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Pestaña 2: Grafo Macro Agregado (Nivel Comunidades)</b> — r/{subreddit_name}<br>"
                 f"<sup>Macroaristas con Densidad de Corte Real (ρ) | Sin conexiones artificiales por MST</sup>",
            x=0.02, y=0.96
        ),
        template="plotly_dark", hovermode="closest",
        margin=dict(l=20, r=260, t=90, b=30),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(title="Emoción de Puentes", orientation="v", y=1, yanchor="top", x=1.02, xanchor="left", font=dict(size=10))
    )
    return fig


# ==============================================================================
# 3. PESTAÑA 3: EVOLUCIÓN TEMPORAL DEL GRAFO (SNAPSHOTS CON SLIDER Y ANCLAJE ESPACIAL)
# ==============================================================================

def build_macro_temporal_figure(G_full, community_topics, subreddit_name="Subreddit",
                                pruning_percentile=60, min_cross_edges=2):
    """
    Construye la Pestaña 3: Evolución Temporal del Grafo Macro (Snapshots Trimestrales).
    Muestra qué comunidades se conectaban en cada trimestre, usando anclaje espacial fijo
    para que las macrocomunidades no salten de posición, y aristas basadas en
    la Densidad de Corte Semántica real (ρ).
    *** 100% LIBRE DE CONEXIONES ARTIFICIALES POR MST. ***
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    periods = sort_periods_chronologically(G_full.vs["period"])
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    if not periods or not all_comms:
        return go.Figure()

    # 1. Calcular grafo macro global para definir el ANCLAJE ESPACIAL fijo
    G_macro_g = collapse_macro_graph(G_full, community_topics, pruning_percentile=pruning_percentile, min_cross_edges=min_cross_edges)
    n_comms = len(G_macro_g.vs)
    if n_comms == 0:
        return go.Figure()

    # Semilla circular determinista para disposición Fruchterman-Reingold estable
    init_seed = [[float(np.cos(2 * np.pi * i / n_comms)), float(np.sin(2 * np.pi * i / n_comms))] for i in range(n_comms)]
    layout = G_macro_g.layout_fruchterman_reingold(
        weights="cut_density" if G_macro_g.ecount() > 0 else None,
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

    # Volumen máximo trimestral para escalamiento relativo de nodos
    period_nodes_map = {p: [v.index for v in G_full.vs if v["period"] == p] for p in periods}
    max_period_vol = 1
    for p in periods:
        c_counts = Counter([G_full.vs[idx]["community"] for idx in period_nodes_map[p] if G_full.vs[idx]["community"] != -1])
        if c_counts:
            max_period_vol = max(max_period_vol, max(c_counts.values()))

    valid_periods = []

    for p in periods:
        nodes_p = period_nodes_map[p]
        if len(nodes_p) < 15:
            continue

        valid_periods.append(p)
        start_idx = len(fig.data)

        # Subgrafo colapsado de este trimestre con densidad de corte real (sin MST)
        G_sub_p = G_full.subgraph(nodes_p)
        G_macro_p = collapse_macro_graph(G_sub_p, community_topics, pruning_percentile=pruning_percentile, min_cross_edges=min_cross_edges)

        # 1. Aristas por emoción para este periodo
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
                    edges_by_emo[emo].append((c1, c2, e["cut_density"], e["count"]))

            for emo, edge_list in edges_by_emo.items():
                cfg = EMOTION_PALETTE.get(emo, {"name": emo, "color": "rgba(180, 180, 180, 0.45)"})
                for c1, c2, c_dens, cross_cnt in edge_list:
                    width = 1.5 + ((c_dens - d_min) / d_range) * 6.5
                    hover = (
                        f"<b>Trimestre: {p}</b><br>"
                        f"<b>Puente Semántico:</b> C{c1} ↔ C{c2}<br>"
                        f"<b>Emoción Predominante:</b> {cfg['name']}<br>"
                        f"<b>Densidad de Corte (ρ):</b> {c_dens:.6e}<br>"
                        f"<b>Comentarios Cruzados en Trimestre:</b> {cross_cnt:,}"
                    )
                    fig.add_trace(go.Scatter(
                        x=[anchor_pos[c1][0], anchor_pos[c2][0], None],
                        y=[anchor_pos[c1][1], anchor_pos[c2][1], None],
                        mode="lines",
                        line=dict(width=width, color=cfg["color"]),
                        visible=False,
                        name=f"Puente: {cfg['name']}",
                        legendgroup=f"Emo_{emo}",
                        showlegend=(emo not in seen_emo),
                        hovertext=hover,
                        hovertemplate="%{hovertext}<extra></extra>"
                    ))
                    seen_emo.add(emo)

        # 2. Nodos activos en este periodo
        x_n, y_n, s_n, col_n, txt_n, hov_n = [], [], [], [], [], []
        for v in G_macro_p.vs:
            c = v["community"]
            if c in anchor_pos:
                x_n.append(anchor_pos[c][0])
                y_n.append(anchor_pos[c][1])
                s_n.append(16.0 + (np.sqrt(v["size"]) / np.sqrt(max_period_vol)) * 34.0)
                col_n.append(get_community_color(c))
                txt_n.append(f"<b>C{c}</b>")
                top_str = get_community_topic(community_topics, c, max_words=5)
                hov_n.append(
                    f"<b>Trimestre: {p}</b><br>"
                    f"<b>Macro-Comunidad:</b> C{c}<br>"
                    f"<b>Tópico Semántico:</b> {top_str}<br>"
                    f"<b>Volumen en Trimestre:</b> {v['size']:,} comentarios<br>"
                    f"<b>Emoción Dominante:</b> {v['dominant_emotion']}<br>"
                    f"<b>Puentes Semánticos Activos:</b> {G_macro_p.degree(v.index)}"
                )

        fig.add_trace(go.Scatter(
            x=x_n, y=y_n, mode="markers+text", text=txt_n, textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            marker=dict(size=s_n, color=col_n, line=dict(width=2, color="white"), opacity=0.95),
            hovertext=hov_n, hovertemplate="%{hovertext}<extra></extra>",
            visible=False, showlegend=False
        ))

        trace_map[p] = list(range(start_idx, len(fig.data)))

    # Hacer visibles por defecto las trazas del primer periodo válido
    if valid_periods:
        for idx in trace_map[valid_periods[0]]:
            fig.data[idx].visible = True

    # Pasos del slider por trimestre
    steps = [
        dict(
            label=p, method="update",
            args=[
                {"visible": [i in trace_map[p] for i in range(len(fig.data))]},
                {"title.text": f"<b>Pestaña 3: Evolución Temporal del Grafo Macro (Snapshots)</b> — r/{subreddit_name} [{p}]<br><sup>Red de macrocomunidades y puentes semánticos reales en el trimestre | Anclaje espacial estable (Sin MST)</sup>"}
            ]
        )
        for p in valid_periods
    ]

    fig.update_layout(
        title=dict(
            text=f"<b>Pestaña 3: Evolución Temporal del Grafo Macro (Snapshots)</b> — r/{subreddit_name} [{valid_periods[0]}]<br><sup>Red de macrocomunidades y puentes semánticos reales en el trimestre | Anclaje espacial estable (Sin MST)</sup>",
            x=0.02, y=0.96
        ),
        template="plotly_dark",
        hovermode="closest",
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Trimestre Cronológico: ", "font": {"size": 13, "color": "white"}},
            steps=steps,
            pad={"t": 35, "b": 10},
            len=0.92, x=0.04, y=0.0
        )],
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=260, t=90, b=90),
        legend=dict(title="Puentes Emocionales", orientation="v", y=1, yanchor="top", x=1.02, xanchor="left", font=dict(size=10))
    )
    return fig


# ==============================================================================
# 4. PESTAÑA 4: MATRIZ DE BURBUJAS (VOLUMEN EXCLUSIVAMENTE EN PORCENTAJE)
# ==============================================================================

def build_community_period_bubble_grid(G_full, community_topics, subreddit_name="Subreddit"):
    """
    Construye la Pestaña 4: Matriz de Burbujas en Cuadrícula (Comunidades vs Periodos).
    *** AJUSTE SOLICITADO: EL MODO DE VOLUMEN MUESTRA EXCLUSIVAMENTE EL PORCENTAJE (%). ***
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    periods = sort_periods_chronologically(G_full.vs["period"])
    if not periods or not all_comms: return go.Figure()

    comm_labels = {c: f"C{c}: {get_community_topic(community_topics, c, max_words=5)}" for c in all_comms}

    cell_vols = defaultdict(int)
    cell_emo_counts = defaultdict(lambda: Counter())
    for v in G_full.vs:
        c, p, f = v["community"], v["period"], v["factor"]
        if c != -1 and p:
            cell_vols[(c, p)] += 1
            if f: cell_emo_counts[(c, p)][f] += 1

    quarter_totals = defaultdict(int)
    for (c, p), v_count in cell_vols.items():
        quarter_totals[p] += v_count

    cells_by_emo = {m: [] for m in FACTOR_COLS}
    all_vols = []

    for c in all_comms:
        for p in periods:
            vol = cell_vols.get((c, p), 0)
            if vol == 0: continue
            all_vols.append(vol)
            counts = cell_emo_counts.get((c, p), Counter())
            if not counts: continue
            dom_emo, dom_cnt = counts.most_common(1)[0]
            pct = (dom_cnt / vol) * 100.0
            q_total = quarter_totals.get(p, 0)
            pct_quarter = (vol / q_total * 100.0) if q_total > 0 else 0.0
            cells_by_emo[dom_emo].append({
                "comm": c, "comm_label": comm_labels[c], "period": p,
                "vol": vol, "pct": pct, "q_total": q_total, "pct_quarter": pct_quarter,
                "dom_emo": dom_emo, "counts": dict(counts)
            })

    fig = go.Figure()

    # MODO 1: % DOMINANCIA EMOCIONAL (Visible por defecto)
    for m in FACTOR_COLS:
        pts = cells_by_emo[m]
        x = [pt["period"] for pt in pts]
        y = [pt["comm_label"] for pt in pts]
        sizes = [18.0 + (pt["pct"] / 100.0) * 28.0 for pt in pts]
        texts = [f"{pt['pct']:.0f}%" for pt in pts]

        hovers = []
        for pt in pts:
            counts_str = "<br>".join([f"  • {EMOTION_PALETTE.get(k, {}).get('name', k)}: {v:,} ({v/pt['vol']*100:.1f}%)" for k, v in Counter(pt["counts"]).most_common()])
            hovers.append(
                f"<b>Macro-Comunidad:</b> {pt['comm_label']}<br>"
                f"<b>Periodo Cronológico:</b> {pt['period']}<br>"
                f"<b>Emoción Dominante:</b> {EMOTION_PALETTE[m]['name']}<br>"
                f"<b>% Dominancia Emocional:</b> {pt['pct']:.1f}%<br>"
                f"<b>Participación Trimestral:</b> {pt['pct_quarter']:.1f}% del total del trimestre<br>"
                f"<b>Desglose Afectivo:</b><br>{counts_str}"
            )

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text", text=texts, textposition="middle center",
            textfont=dict(color="white", size=9.5, family="Arial, sans-serif"),
            marker=dict(size=sizes, color=EMOTION_PALETTE[m]["color"], line=dict(width=1.5, color="rgba(255,255,255,0.85)"), opacity=0.92),
            hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
            name=EMOTION_PALETTE[m]["name"], legendgroup=m, showlegend=True, visible=True
        ))

    # MODO 2: VOLUMEN DE PUBLICACIONES (SOLO EN PORCENTAJE SEGÚN REQUERIMIENTO)
    max_pct_q = max([pt["pct_quarter"] for pts in cells_by_emo.values() for pt in pts] + [1.0])
    for m in FACTOR_COLS:
        pts = cells_by_emo[m]
        x = [pt["period"] for pt in pts]
        y = [pt["comm_label"] for pt in pts]
        # Tamaño proporcional al porcentaje de participación
        sizes = [16.0 + (pt["pct_quarter"] / max_pct_q) * 32.0 for pt in pts]
        # ETIQUETA: EXCLUSIVAMENTE PORCENTAJE (%)
        texts = [f"{pt['pct_quarter']:.1f}%" for pt in pts]

        hovers = []
        for pt in pts:
            counts_str = "<br>".join([f"  • {EMOTION_PALETTE.get(k, {}).get('name', k)}: {v:,} ({v/pt['vol']*100:.1f}%)" for k, v in Counter(pt["counts"]).most_common()])
            hovers.append(
                f"<b>Macro-Comunidad:</b> {pt['comm_label']}<br>"
                f"<b>Periodo Cronológico:</b> {pt['period']}<br>"
                f"<b>Emoción Dominante:</b> {EMOTION_PALETTE[m]['name']}<br>"
                f"<b>% del Total del Trimestre:</b> {pt['pct_quarter']:.1f}%<br>"
                f"<b>% Dominancia Emocional:</b> {pt['pct']:.1f}%<br>"
                f"<b>Desglose Afectivo:</b><br>{counts_str}"
            )

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text", text=texts, textposition="middle center",
            textfont=dict(color="white", size=9.0, family="Arial, sans-serif"),
            marker=dict(size=sizes, color=EMOTION_PALETTE[m]["color"], line=dict(width=1.5, color="rgba(255,255,255,0.85)"), opacity=0.92),
            hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
            name=EMOTION_PALETTE[m]["name"], legendgroup=m, showlegend=True, visible=False
        ))

    n_m = len(FACTOR_COLS)
    buttons = [
        dict(label="🔘 % Dominancia Emocional", method="update",
             args=[{"visible": [True]*n_m + [False]*n_m},
                   {"title.text": f"<b>Pestaña 3: Matriz Cuadrícula de Comunidades vs Periodos</b> — r/{subreddit_name}<br><sup>Tamaño y Etiqueta = % Dominancia Afectiva | Color = Emoción Dominante</sup>"}]),
        dict(label="🔘 % Volumen del Trimestre", method="update",
             args=[{"visible": [False]*n_m + [True]*n_m},
                   {"title.text": f"<b>Pestaña 3: Matriz Cuadrícula de Comunidades vs Periodos</b> — r/{subreddit_name}<br><sup>Tamaño y Etiqueta = % de Participación en el Trimestre (Solo %) | Color = Emoción Dominante</sup>"}])
    ]

    dynamic_height = max(580, len(all_comms) * 58)
    fig.update_layout(
        title=dict(
            text=f"<b>Pestaña 3: Matriz Cuadrícula de Comunidades vs Periodos</b> — r/{subreddit_name}<br><sup>Tamaño y Etiqueta = % Dominancia Afectiva | Secuencia cronológica estricta</sup>",
            x=0.02, y=0.98
        ),
        template="plotly_dark", height=dynamic_height,
        updatemenus=[dict(type="buttons", direction="right", x=0.01, y=1.13, buttons=buttons, bgcolor="#1e293b", font=dict(color="white", size=11), active=0)],
        xaxis=dict(title="<b>Periodo Cronológico (Trimestre)</b>", tickmode="array", tickvals=periods, ticktext=periods, showgrid=True, gridcolor="rgba(255, 255, 255, 0.12)", zeroline=False),
        yaxis=dict(title="<b>Macro-Comunidad Temática</b>", categoryorder="array", categoryarray=[comm_labels[c] for c in all_comms[::-1]], showgrid=True, gridcolor="rgba(255, 255, 255, 0.12)", zeroline=False),
        legend=dict(title="Emoción Dominante", orientation="v", y=1, yanchor="top", x=1.02, xanchor="left", font=dict(size=10)),
        margin=dict(l=40, r=260, t=110, b=50)
    )
    return fig


# ==============================================================================
# 5. PESTAÑA 5: VOLUMEN LONGITUDINAL POR COMUNIDAD (ABSOLUTO Y % DE CAMBIO)
# ==============================================================================

def build_longitudinal_volume_figure(G_full, community_topics, subreddit_name="Subreddit"):
    """
    Construye la Pestaña 5: Evolución Temporal del Volumen Longitudinal por Macro-Comunidad.
    *** AJUSTES REQUERIDOS:
      1. EXCLUSIVAMENTE VOLUMEN ABSOLUTO (SIN TOTAL DEL SUBREDDIT Y SIN SPAGHETTI).
      2. RESALTADO PERSONALIZADO EN CADA COMUNIDAD (SIN BORDES NI LÍNEAS DE RESALTADO EN BLANCO).
    ***
    """
    ensure_graph_attributes(G_full)
    community_topics = ensure_community_topics(G_full, community_topics)
    all_comms = sorted([c for c in set(G_full.vs["community"]) if c != -1])
    periods = sort_periods_chronologically(G_full.vs["period"])
    if not periods or not all_comms: return go.Figure()

    vol_comm_period = defaultdict(lambda: defaultdict(int))
    total_period = defaultdict(int)

    for v in G_full.vs:
        c = v["community"]
        p = v["period"]
        if c != -1 and p:
            vol_comm_period[c][p] += 1
            total_period[p] += 1

    fig = go.Figure()

    # TRAZAS EXCLUSIVAS DE VOLUMEN ABSOLUTO POR COMUNIDAD
    for c in all_comms:
        comm_y_abs = [vol_comm_period[c][p] for p in periods]
        comm_color = get_community_color(c)
        topic_words = get_community_topic(community_topics, c, max_words=4)
        
        hover_abs = [
            f"<b>Macro-Comunidad C{c}</b><br>"
            f"<b>Tópico:</b> {topic_words}<br>"
            f"<b>Periodo:</b> {p}<br>"
            f"<b>Volumen Absoluto:</b> {cnt:,} comentarios<br>" +
            (f"<b>% del Trimestre:</b> {(cnt / total_period[p] * 100):.1f}%" if total_period[p] > 0 else "0%")
            for p, cnt in zip(periods, comm_y_abs)
        ]

        fig.add_trace(go.Scatter(
            x=periods, y=comm_y_abs, mode="lines+markers",
            name=f"C{c}: {topic_words[:24]}...",
            legendgroup=f"C{c}",
            line=dict(color=comm_color, width=2.4),
            marker=dict(
                size=7,
                color=comm_color,
                line=dict(width=1.5, color=comm_color)  # Borde con color de la comunidad (no blanco)
            ),
            hovertext=hover_abs,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverlabel=dict(
                bgcolor="#0f172a",          # Fondo oscuro elegante
                bordercolor=comm_color,     # Resaltado con el color único de la comunidad (no blanco)
                font=dict(color="#f1f5f9", size=11.5, family="Arial, sans-serif")
            ),
            visible=True
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>Pestaña 5: Evolución Temporal del Volumen por Macro-Comunidad</b> — r/{subreddit_name}<br>"
                 f"<sup>Recuentos longitudinales exactos en volumen absoluto (2019Q1 a 2022Q3) | Haz clic en la leyenda para aislar</sup>",
            x=0.02, y=0.96
        ),
        template="plotly_dark",
        hovermode="closest",
        margin=dict(l=40, r=260, t=90, b=50),
        xaxis=dict(
            title="<b>Trimestre Cronológico</b>",
            tickmode="array", tickvals=periods, ticktext=periods,
            showgrid=True, gridcolor="rgba(51, 65, 85, 0.4)",      # Pizarra oscuro (no blanco)
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(56, 189, 248, 0.6)",                  # Resaltado cian eléctrico (no blanco)
            spikethickness=1.5, spikedash="dot"
        ),
        yaxis=dict(
            title="<b>Número de Comentarios</b>",
            showgrid=True, gridcolor="rgba(51, 65, 85, 0.4)",      # Pizarra oscuro (no blanco)
            zeroline=True, zerolinecolor="rgba(59, 130, 246, 0.4)", # Azul acento (no blanco)
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(56, 189, 248, 0.6)",                  # Resaltado cian eléctrico (no blanco)
            spikethickness=1.5, spikedash="dot"
        ),
        legend=dict(
            title="Macro-Comunidades",
            orientation="v", y=1, yanchor="top", x=1.02, xanchor="left",
            font=dict(size=10)
        )
    )
    return fig


# ==============================================================================
# 6. FUNCIÓN INTEGRADORA DEL DASHBOARD (5 PESTAÑAS EXACTAS)
# ==============================================================================

def create_unified_sna_dashboard(subreddit, graph_data):
    """
    Retorna el diccionario con las 5 figuras analíticas optimizadas:
      - 'micro': Pestaña 1 (Micro Orgánico con muestreo alineado al tópico y emociones)
      - 'macro': Pestaña 2 (Macro Agregado sin MST con corte semántico real)
      - 'temporal': Pestaña 3 (Macro Temporal Longitudinal con snapshots y anclaje espacial)
      - 'evolution': Pestaña 4 (Matriz de Burbujas en Cuadrícula con volumen en %)
      - 'volume': Pestaña 5 (Evolución de Volumen Absoluto por Macro-Comunidad sin blanco)
    """
    G = graph_data["graph"]
    topics = graph_data["topics"]

    print(f"[*] Construyendo las 5 Vistas Analíticas para r/{subreddit}...")
    fig_micro = build_micro_organic_figure(G, topics, subreddit_name=subreddit)
    fig_macro = build_macro_figure(collapse_macro_graph(G, topics), subreddit_name=subreddit)
    fig_temporal = build_macro_temporal_figure(G, topics, subreddit_name=subreddit)
    fig_evolution = build_community_period_bubble_grid(G, topics, subreddit_name=subreddit)
    fig_volume = build_longitudinal_volume_figure(G, topics, subreddit_name=subreddit)

    return {
        "micro": fig_micro,
        "macro": fig_macro,
        "temporal": fig_temporal,
        "evolution": fig_evolution,
        "volume": fig_volume
    }
