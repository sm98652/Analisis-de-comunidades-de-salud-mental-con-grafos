# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DEL DASHBOARD ANALÍTICO INTERACTIVO SNA V3.0 (5 PESTAÑAS)
================================================================================
Archivo de salida por defecto:
  - dashboard_grafos_interactivos_sna_v3.html
Carpeta de caché:
  - dashboard_cache_sna_v3/

Estructura de 5 Pestañas:
  1. 🔬 1. Red de Comentarios: Agrupamiento semántico y desglose micro de comentarios.
  2. 🌐 2. Red de Comunidades: Macro-red y puentes semánticos de densidad ρ.
  3. ⏳ 3. Evolución por Trimestre: Snapshots trimestrales interactivos con slider cronológico.
  4. 📈 4. Tendencias de Cuota (%): Enfocada 100% en volumen y participación estructural:
       1. Cuota Relativa (% Trimestre)
       2. Cambio Estructural (Δ pp vs 2019Q1)
  5. 🎭 5. Clima Emocional por Trimestre (Analítica afectiva):
       1. Matriz de Emoción Predominante (Vista principal):
          Eje X: Trimestres cronológicos (2019Q1 – 2022Q3).
          Eje Y: Macro-Comunidades temáticas ordenadas de mayor a menor volumen (C0, C1, etc.).
          Celdas: Bloques (square 44) con el color oficial de la emoción predominante.
          Etiqueta interna clara: Texto legible con la emoción y % dominancia (ej. Miedo 53%, Tristeza 64%).
          Tooltip al pasar cursor: Desglose completo de las 7 emociones y volumen.
       2. Curvas de Evolución Emocional (%):
          Trayectorias temporales de emoción dominante o aislar emoción específica con menú desplegable.
================================================================================
"""

import time
import json
import gc
import os
import sys
import igraph as ig
import numpy as np

from visualizador_redes_sna_v3 import (
    ensure_graph_attributes,
    ensure_community_topics,
    build_micro_organic_figure,
    collapse_macro_graph,
    build_macro_figure,
    build_macro_temporal_figure,
    build_community_period_bubble_grid,
    build_quota_trends_figure,
    build_affective_climate_figure,
    build_subreddit_summary_meta,
    FACTOR_COLS,
    EMOTION_PALETTE
)

SUBREDDITS = ["Anxiety", "SuicideWatch", "depression"]


def procesar_un_subreddit(sub, cache_dir="dashboard_cache_sna_v3", force_recalc=False):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cache_{sub}_v3.json")

    pkl_name = f"Grafo_{sub}_best.pkl"
    json_name = f"topics_{sub}_best.json"

    if not force_recalc and os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        print(f"[*] r/{sub}: Utilizando caché V3 existente desde '{cache_file}'.")
        return

    if not os.path.exists(pkl_name):
        # Si no existe el .pkl, verificar si ya existe el caché
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
            print(f"[*] r/{sub}: Utilizando caché V3 existente desde '{cache_file}'.")
            return
        # Extracción automática desde el HTML base para inicializar el caché
        template_path = os.path.join(os.path.dirname(__file__), "dashboard_grafos_interactivos_sna_v3.html")
        if os.path.exists(template_path):
            try:
                print(f"[*] r/{sub}: Inicializando caché desde '{template_path}'...")
                with open(template_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "const DATA = " in line:
                            idx = line.find("const DATA = ")
                            s = line[idx + len("const DATA = "):].strip().rstrip(";")
                            data_all = json.loads(s)
                            payload = {
                                "meta": data_all.get("meta", {}).get(sub, {}),
                                "micro": data_all.get("micro", {}).get(sub, {}),
                                "macro": data_all.get("macro", {}).get(sub, {}),
                                "temporal": data_all.get("temporal", {}).get(sub, {}),
                                "evolution": data_all.get("evolution", {}).get(sub, {}),
                                "emotion": data_all.get("emotion", {}).get(sub, {})
                            }
                            with open(cache_file, "w", encoding="utf-8") as cf:
                                json.dump(payload, cf)
                            print(f"    [OK] r/{sub}: Caché creado con éxito en '{cache_file}'.")
                            return
            except Exception as e:
                print(f"[!] Error al extraer datos de '{template_path}': {e}")
        print(f"[!] Omitiendo r/{sub}: No existe '{pkl_name}' ni archivo de caché.")
        return

    print(f"\n[*] Procesando r/{sub} para Dashboard SNA V3.0...")
    t0 = time.time()

    G = ig.Graph.Read_Pickle(pkl_name)
    ensure_graph_attributes(G, subreddit_name=sub)
    topics = {}
    if os.path.exists(json_name):
        with open(json_name, "r", encoding="utf-8") as f:
            topics = {int(k): v for k, v in json.load(f).items()}
    topics = ensure_community_topics(G, topics)

    # 0. Metadatos estructurados
    print("    -> Extrayendo Metadatos Analíticos...")
    meta_json = build_subreddit_summary_meta(G, topics, subreddit_name=sub)

    # 1. Pestaña 1: Micro Orgánico
    print("    -> Generando Pestaña 1: Grafo Micro Orgánico...")
    fig_micro = build_micro_organic_figure(G, topics, subreddit_name=sub, max_nodes_total=1400)
    fig_micro.update_layout(margin=dict(l=30, r=40, t=95, b=20))
    micro_json = json.loads(fig_micro.to_json())
    del fig_micro; gc.collect()

    # 2. Pestaña 2: Macro Agregado
    print("    -> Generando Pestaña 2: Grafo Macro Agregado...")
    G_macro = collapse_macro_graph(G, topics, pruning_percentile=0, min_cross_edges=0, ref_percentile=65)
    fig_macro = build_macro_figure(G_macro, subreddit_name=sub)
    fig_macro.update_layout(showlegend=False, margin=dict(l=30, r=40, t=95, b=25))
    for tr in fig_macro.data:
        tr.showlegend = False
    macro_json = json.loads(fig_macro.to_json())
    del fig_macro, G_macro; gc.collect()

    # 3. Pestaña 3: Evolución Temporal Macro
    print("    -> Generando Pestaña 3: Evolución Temporal del Grafo Macro...")
    fig_temporal = build_macro_temporal_figure(G, topics, subreddit_name=sub, pruning_percentile=0, min_cross_edges=0, ref_percentile=60)
    fig_temporal.update_layout(showlegend=False, margin=dict(l=30, r=40, t=95, b=85))
    for tr in fig_temporal.data:
        tr.showlegend = False
    temporal_json = json.loads(fig_temporal.to_json())
    del fig_temporal; gc.collect()

    # 4. Pestaña 4: Tendencias de Cuota (%)
    print("    -> Generando Pestaña 4: Tendencias de Cuota (%): Cuota Relativa y Cambio Estructural...")
    fig_evolution = build_quota_trends_figure(G, topics, subreddit_name=sub)
    fig_evolution.update_layout(margin=dict(l=65, r=40, t=90, b=55))
    evolution_json = json.loads(fig_evolution.to_json())
    del fig_evolution; gc.collect()

    # 5. Pestaña 5: Clima Emocional por Trimestre
    print("    -> Generando Pestaña 5: Clima Emocional por Trimestre: Matriz y Curvas...")
    fig_emotion = build_affective_climate_figure(G, topics, subreddit_name=sub)
    fig_emotion.update_layout(margin=dict(l=340, r=40, t=90, b=60))
    emotion_json = json.loads(fig_emotion.to_json())
    del fig_emotion, G; gc.collect()

    payload = {
        "meta": meta_json,
        "micro": micro_json,
        "macro": macro_json,
        "temporal": temporal_json,
        "evolution": evolution_json,
        "emotion": emotion_json
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    del payload; gc.collect()
    print(f"    [OK] r/{sub} procesado y guardado en '{cache_file}' en {time.time()-t0:.1f}s.")


def generar_dashboard_completo_v3(output_html="dashboard_grafos_interactivos_sna_v3.html", force_recalc=False):
    cache_dir = "dashboard_cache_sna_v3"
    os.makedirs(cache_dir, exist_ok=True)

    print("\n" + "="*80)
    print("GENERADOR DEL DASHBOARD ANALÍTICO INTERACTIVO SNA V3.0 (5 PESTAÑAS)")
    print("="*80)

    for sub in SUBREDDITS:
        procesar_un_subreddit(sub, cache_dir=cache_dir, force_recalc=force_recalc)

    dashboard_data = {"meta": {}, "micro": {}, "macro": {}, "temporal": {}, "evolution": {}, "emotion": {}}
    for sub in SUBREDDITS:
        cache_file = os.path.join(cache_dir, f"cache_{sub}_v3.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                dashboard_data["meta"][sub] = data.get("meta", {})
                dashboard_data["micro"][sub] = data.get("micro", {})
                dashboard_data["macro"][sub] = data.get("macro", {})
                dashboard_data["temporal"][sub] = data.get("temporal", {})

                # 4. Pestaña 4: Tendencias de Cuota (%)
                evo_raw = data.get("evolution", {})
                block0_share = []
                block1_delta = []
                raw_block3_emo = None

                meta_sub = data.get("meta", {})
                n_comms = len(meta_sub.get("communities", []))

                if evo_raw and "data" in evo_raw:
                    evo_data = evo_raw["data"]
                    if n_comms > 0:
                        n_blocks = len(evo_data) // n_comms
                        if n_blocks >= 2:
                            block0_share = evo_data[:n_comms]
                            block1_delta = evo_data[n_comms:2*n_comms]
                            if n_blocks >= 4:
                                raw_block3_emo = evo_data[3*n_comms:4*n_comms]
                        else:
                            block0_share = evo_data
                            block1_delta = []
                    else:
                        block0_share = evo_data
                        block1_delta = []

                    years = ['2019', '2019', '2019', '2019', '2020', '2020', '2020', '2020', '2021', '2021', '2021', '2021', '2022', '2022', '2022']
                    quarters = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3', 'Q4', 'Q1', 'Q2', 'Q3']

                    kept_traces = block0_share + block1_delta
                    for tr in kept_traces:
                        tr["x"] = [years, quarters]

                    layout4 = evo_raw.get("layout", {})
                    layout4["xaxis"] = {
                        "title": {"text": "<b>Periodo Cronológico (Año y Trimestre)</b>", "font": {"size": 15, "color": "#0f172a", "family": "Segoe UI, sans-serif"}},
                        "tickfont": {"size": 13, "color": "#0f172a", "family": "Segoe UI, sans-serif"},
                        "type": "multicategory",
                        "showgrid": True,
                        "gridcolor": "rgba(203, 213, 225, 0.7)",
                        "zeroline": False,
                        "showspikes": True,
                        "spikemode": "across",
                        "spikesnap": "cursor",
                        "spikecolor": "rgba(2, 132, 199, 0.55)",
                        "spikethickness": 1.3,
                        "spikedash": "dot"
                    }
                    if "yaxis" in layout4:
                        layout4["yaxis"]["title"] = {"text": "<b>Cuota de Participación en el Trimestre (%)</b>", "font": {"size": 15, "color": "#0f172a", "family": "Segoe UI, sans-serif"}}
                        layout4["yaxis"]["tickfont"] = {"size": 13.5, "color": "#0f172a", "family": "Segoe UI, sans-serif"}
                    if "shapes" in layout4:
                        for sh in layout4["shapes"]:
                            if sh.get("x0") == "2022Q2":
                                sh["x0"] = 13
                                sh["x1"] = 13
                    if "annotations" in layout4:
                        for an in layout4["annotations"]:
                            if an.get("x") == "2022Q2":
                                an["x"] = 13

                    dashboard_data["evolution"][sub] = {
                        "data": kept_traces,
                        "layout": layout4
                    }

                # 5. Pestaña 5: Clima Emocional por Trimestre
                emo_raw = data.get("emotion", {})
                if emo_raw and "data" in emo_raw:
                    emo_layout = emo_raw.get("layout", {})
                    emo_layout["margin"] = {"l": 340, "r": 40, "t": 90, "b": 60}
                    if "yaxis" in emo_layout:
                        emo_layout["yaxis"]["automargin"] = True
                        emo_layout["yaxis"]["title"] = {"text": "<b>Macro-Comunidades Temáticas</b>", "font": {"size": 15, "color": "#0f172a", "family": "Segoe UI, sans-serif"}}
                        emo_layout["yaxis"]["tickfont"] = {"size": 13.5, "color": "#0f172a", "family": "Segoe UI, sans-serif"}
                    if "xaxis" in emo_layout:
                        emo_layout["xaxis"]["title"] = {"text": "<b>Trimestre Cronológico (2019Q1 – 2022Q3)</b>", "font": {"size": 15, "color": "#0f172a", "family": "Segoe UI, sans-serif"}}
                        emo_layout["xaxis"]["tickfont"] = {"size": 13.5, "color": "#0f172a", "family": "Segoe UI, sans-serif"}
                    dashboard_data["emotion"][sub] = {
                        "data": emo_raw["data"],
                        "layout": emo_layout
                    }
                elif block0_share and raw_block3_emo:
                    periods = [cd["period"] for cd in block0_share[0]["customdata"]]
                    comm_labels_ordered = []
                    matrix_traces = []

                    for tr in block0_share:
                        cds = tr["customdata"]
                        c_id = cds[0]["comm"]
                        topic_short = cds[0]["topic_short"].replace("_", " ")
                        c_label = f"C{c_id} ({topic_short})"
                        comm_labels_ordered.append(c_label)

                        x_pts, y_pts, txt_pts, col_pts, hov_pts = [], [], [], [], []
                        for cd in cds:
                            p = cd["period"]
                            dom_short = cd["emotion_short"]
                            dom_pct = cd["pct_dom"]
                            dom_hex = cd["emotion_hex"]

                            x_pts.append(p)
                            y_pts.append(c_label)
                            txt_pts.append(f"<b>{dom_short}</b><br>{dom_pct:.0f}%")
                            col_pts.append(dom_hex)

                            emo_items = sorted(cd.get("emo_pcts", {}).items(), key=lambda x: x[1], reverse=True)
                            emo_lines = "<br>".join([f"  ● <b>{k.capitalize()}:</b> {v:.1f}%" for k, v in emo_items if v > 0])

                            hov_str = (
                                f"<b>Macro-Comunidad C{c_id}: {cd['topic'].replace('_', ' ')}</b><br>"
                                f"<b>Trimestre:</b> {p} &nbsp;|&nbsp; <b>Volumen:</b> {cd['vol']:,} comentarios<br>"
                                f"<b>Emoción Predominante:</b> <span style='color:{dom_hex}; font-size:1.05em;'>● <b>{cd['emotion']} ({dom_pct:.1f}%)</b></span><br>"
                                f"<span style='color:#cbd5e1;'>─────────────────────────────</span><br>"
                                f"<b>Desglose Afectivo Completo:</b><br>{emo_lines}"
                            )
                            hov_pts.append(hov_str)

                        matrix_traces.append({
                            "type": "scatter",
                            "mode": "markers+text",
                            "x": x_pts,
                            "y": y_pts,
                            "text": txt_pts,
                            "textposition": "middle center",
                            "textfont": {"size": 11.5, "color": "#ffffff", "family": "Segoe UI, Arial, sans-serif"},
                            "marker": {
                                "symbol": "square",
                                "size": 44,
                                "color": col_pts,
                                "line": {"width": 1.5, "color": "#ffffff"},
                                "opacity": 0.95
                            },
                            "hovertext": hov_pts,
                            "customdata": cds,
                            "hovertemplate": "%{hovertext}<extra></extra>",
                            "name": c_label,
                            "legendgroup": f"C{c_id}",
                            "visible": tr.get("visible", True)
                        })

                    curve_traces = []
                    for tr in raw_block3_emo:
                        tr_copy = dict(tr)
                        tr_copy["visible"] = False
                        curve_traces.append(tr_copy)

                    layout_emotion = {
                        "title": {
                            "text": f"<b>5. Clima Emocional por Trimestre (Emoción Predominante)</b> — r/{sub}<br><sup>Cada celda indica la emoción dominante en el trimestre y su % de dominancia | Paleta oficial de colores</sup>",
                            "x": 0.02, "y": 0.97,
                            "font": {"size": 16, "color": "#0f172a", "family": "Segoe UI, Arial, sans-serif"}
                        },
                        "template": "plotly_white",
                        "paper_bgcolor": "#ffffff",
                        "plot_bgcolor": "#ffffff",
                        "font": {"color": "#0f172a", "family": "Segoe UI, Arial, sans-serif"},
                        "height": 740,
                        "hovermode": "closest",
                        "xaxis": {
                            "title": {"text": "<b>Trimestre Cronológico (2019Q1 – 2022Q3)</b>", "font": {"size": 15, "color": "#0f172a", "family": "Segoe UI, sans-serif"}},
                            "tickfont": {"size": 13.5, "color": "#0f172a", "family": "Segoe UI, sans-serif"},
                            "type": "category",
                            "categoryorder": "array",
                            "categoryarray": periods,
                            "showgrid": False
                        },
                        "yaxis": {
                            "title": {"text": "<b>Macro-Comunidades Temáticas</b>", "font": {"size": 15, "color": "#0f172a", "family": "Segoe UI, sans-serif"}},
                            "tickfont": {"size": 13.5, "color": "#0f172a", "family": "Segoe UI, sans-serif"},
                            "type": "category",
                            "categoryorder": "array",
                            "categoryarray": comm_labels_ordered[::-1],
                            "showgrid": False,
                            "automargin": True
                        },
                        "margin": {"l": 340, "r": 40, "t": 90, "b": 60}
                    }

                    dashboard_data["emotion"][sub] = {
                        "data": matrix_traces + curve_traces,
                        "layout": layout_emotion
                    }

    # Leer template base
    template_path = os.path.join(os.path.dirname(__file__), "dashboard_grafos_interactivos_sna_v3.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            full_html = f.read()

        idx_start = full_html.find("const DATA = ")
        if idx_start != -1:
            idx_end = full_html.find(";\n    let currentTab", idx_start)
            if idx_end == -1:
                idx_end = full_html.find(";\r\n    let currentTab", idx_start)
            if idx_end != -1:
                full_html = (
                    full_html[:idx_start + len("const DATA = ")] +
                    json.dumps(dashboard_data) +
                    full_html[idx_end:]
                )

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"\n[OK] Dashboard SNA V3.0 (5 Pestañas) generado con éxito: '{output_html}'")


if __name__ == "__main__":
    generar_dashboard_completo_v3(force_recalc=False)
