# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR DEL DASHBOARD ANALÍTICO INTERACTIVO SNA V2.1
(5 PESTAÑAS x 3 SUBREDDITS CON FÍSICA MULTINIVEL, SIN MST Y FLUJO TEMPORAL)
================================================================================
Pestañas:
  1. Micro Orgánico: Fruchterman-Reingold con separación multinivel,
     muestreo cualitativo representativo altamente alineado al tópico
     de cada comunidad y con diversidad afectiva balanceada.
  2. Macro Agregado: Grafo colapsado por macrocomunidades (K = 6 a 15)
     con Densidad de Corte Semántica real (ρ). *** 100% LIBRE DE MST ***.
  3. Evolución Temporal del Grafo: Snapshots por trimestre con slider
     y anclaje espacial para ver qué comunidades se conectaban en cada periodo.
  4. Matriz de Burbujas: Comunidades vs Periodos cronológicos (2019Q1 a 2022Q3).
     Toggle entre % Dominancia Emocional y % Volumen del Trimestre (SOLO PORCENTAJE).
  5. Volumen por Comunidad: Gráfico de líneas cronológico exacto
     *** EXCLUSIVAMENTE POR COMUNIDAD (SIN TOTAL) Y CON TOGGLE ABSOLUTO VS % CAMBIO. ***
================================================================================
"""

import time
import json
import gc
import os
import sys
import igraph as ig
import numpy as np

from visualizador_redes_sna import (
    ensure_graph_attributes,
    ensure_community_topics,
    build_micro_organic_figure,
    collapse_macro_graph,
    build_macro_figure,
    build_macro_temporal_figure,
    build_community_period_bubble_grid,
    build_longitudinal_volume_figure,
    FACTOR_COLS
)

SUBREDDITS = ["Anxiety", "SuicideWatch", "depression"]

def procesar_un_subreddit(sub, cache_dir="dashboard_cache_sna", force_recalc=False, update_volume_only=True):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cache_{sub}.json")

    existing_payload = {}
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
        except Exception:
            existing_payload = {}

    has_other_tabs = all(k in existing_payload for k in ["micro", "macro", "temporal", "evolution"])
    pkl_name = f"Grafo_{sub}_best.pkl"
    json_name = f"topics_{sub}_best.json"
    if not os.path.exists(pkl_name):
        print(f"[!] Omitiendo r/{sub}: No existe '{pkl_name}'.")
        return

    # Si ya tenemos las 4 primeras pestañas válidas, solo recalculamos la Pestaña 5 (toma pocos segundos)
    if update_volume_only and has_other_tabs and not force_recalc:
        print(f"\n[*] Actualizando Pestaña 5 (Volumen Absoluto sin blanco) para r/{sub}...")
        t0 = time.time()
        G = ig.Graph.Read_Pickle(pkl_name)
        ensure_graph_attributes(G, subreddit_name=sub)
        topics = {}
        if os.path.exists(json_name):
            with open(json_name, "r", encoding="utf-8") as f:
                topics = {int(k): v for k, v in json.load(f).items()}
        topics = ensure_community_topics(G, topics)

        fig_volume = build_longitudinal_volume_figure(G, topics, subreddit_name=sub)
        existing_payload["volume"] = json.loads(fig_volume.to_json())
        del fig_volume, G; gc.collect()

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(existing_payload, f)
        print(f"    [OK] Pestaña 5 actualizada en caché para r/{sub} en {time.time()-t0:.1f}s.")
        return

    print(f"\n[*] Procesando r/{sub} para Dashboard SNA (5 pestañas completas)...")
    t0 = time.time()

    G = ig.Graph.Read_Pickle(pkl_name)
    ensure_graph_attributes(G, subreddit_name=sub)
    topics = {}
    if os.path.exists(json_name):
        with open(json_name, "r", encoding="utf-8") as f:
            topics = {int(k): v for k, v in json.load(f).items()}
    topics = ensure_community_topics(G, topics)

    # 1. Pestaña 1: Micro Orgánico con Muestreo Alineado al Tópico
    print("    -> Generando Pestaña 1: Grafo Micro Orgánico (Alineación Temática y Afectiva)...")
    fig_micro = build_micro_organic_figure(G, topics, subreddit_name=sub, max_nodes_total=1400)
    micro_json = json.loads(fig_micro.to_json())
    del fig_micro; gc.collect()

    # 2. Pestaña 2: Macro Agregado (Sin MST)
    print("    -> Generando Pestaña 2: Grafo Macro Agregado (Sin MST)...")
    G_macro = collapse_macro_graph(G, topics, pruning_percentile=65, min_cross_edges=3)
    fig_macro = build_macro_figure(G_macro, subreddit_name=sub)
    macro_json = json.loads(fig_macro.to_json())
    del fig_macro, G_macro; gc.collect()

    # 3. Pestaña 3: Evolución Temporal del Grafo (Snapshots Trimestrales con Slider)
    print("    -> Generando Pestaña 3: Evolución Temporal del Grafo Macro (Snapshots con Slider)...")
    fig_temporal = build_macro_temporal_figure(G, topics, subreddit_name=sub, pruning_percentile=60, min_cross_edges=2)
    temporal_json = json.loads(fig_temporal.to_json())
    del fig_temporal; gc.collect()

    # 4. Pestaña 4: Matriz de Burbujas en Cuadrícula (Volumen SOLO en %)
    print("    -> Generando Pestaña 4: Matriz de Burbujas (Volumen SOLO en %)...")
    fig_evolution = build_community_period_bubble_grid(G, topics, subreddit_name=sub)
    evolution_json = json.loads(fig_evolution.to_json())
    del fig_evolution; gc.collect()

    # 5. Pestaña 5: Volumen Longitudinal por Comunidad (Absoluto y % de Cambio)
    print("    -> Generando Pestaña 5: Volumen Longitudinal por Comunidad (Absoluto y % de Cambio)...")
    fig_volume = build_longitudinal_volume_figure(G, topics, subreddit_name=sub)
    volume_json = json.loads(fig_volume.to_json())
    del fig_volume, G; gc.collect()

    payload = {
        "micro": micro_json,
        "macro": macro_json,
        "temporal": temporal_json,
        "evolution": evolution_json,
        "volume": volume_json
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    del payload; gc.collect()
    print(f"    [OK] r/{sub} procesado y guardado en caché en {time.time()-t0:.1f}s.")

def generar_dashboard_completo(output_html="dashboard_grafos_interactivos_sna.html"):
    cache_dir = "dashboard_cache_sna"
    os.makedirs(cache_dir, exist_ok=True)

    print("\n" + "="*80)
    print("GENERADOR DEL DASHBOARD ANALÍTICO INTERACTIVO SNA V2.1 (5 PESTAÑAS)")
    print("="*80)

    for sub in SUBREDDITS:
        procesar_un_subreddit(sub, cache_dir=cache_dir, force_recalc=False, update_volume_only=True)

    dashboard_data = {"micro": {}, "macro": {}, "temporal": {}, "evolution": {}, "volume": {}}
    for sub in SUBREDDITS:
        cache_file = os.path.join(cache_dir, f"cache_{sub}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                dashboard_data["micro"][sub] = data["micro"]
                dashboard_data["macro"][sub] = data["macro"]
                dashboard_data["temporal"][sub] = data.get("temporal", {})
                dashboard_data["evolution"][sub] = data.get("evolution", {})
                dashboard_data["volume"][sub] = data.get("volume", {})

    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Dashboard SNA: Redes Semántico-Afectivas (Reddit RMHD)</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg-dark: #0f141c;
      --bg-card: #182232;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --text: #f1f5f9;
      --text-dim: #94a3b8;
      --border: #334155;
    }}
    body {{
      margin: 0; padding: 0; background-color: var(--bg-dark);
      color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    header {{
      background-color: var(--bg-card); padding: 14px 28px;
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid var(--border);
    }}
    .title-area h1 {{ margin: 0; font-size: 1.25rem; font-weight: 700; color: #60a5fa; }}
    .title-area p {{ margin: 3px 0 0; font-size: 0.8rem; color: var(--text-dim); }}
    .sub-selector {{
      display: flex; align-items: center; gap: 10px; font-size: 0.88rem;
    }}
    select {{
      background-color: #0b0f17; color: #38bdf8; border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 6px; font-weight: 600; cursor: pointer; outline: none;
    }}
    nav.tabs {{
      display: flex; background-color: #131b27; border-bottom: 1px solid var(--border); padding: 0 28px;
      gap: 6px;
    }}
    .tab-btn {{
      background: none; border: none; color: var(--text-dim);
      padding: 12px 18px; font-size: 0.88rem; font-weight: 600; cursor: pointer;
      border-bottom: 3px solid transparent; transition: all 0.2s ease;
    }}
    .tab-btn:hover {{ color: #38bdf8; }}
    .tab-btn.active {{ color: #60a5fa; border-bottom-color: #60a5fa; background-color: rgba(96, 165, 250, 0.06); }}
    main {{ padding: 16px 28px; }}
    .info-banner {{
      background-color: rgba(37, 99, 235, 0.1); border-left: 4px solid var(--accent);
      padding: 10px 16px; margin-bottom: 16px; border-radius: 4px; font-size: 0.85rem; color: #cbd5e1;
    }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}
    .graph-card {{ display: none; }}
    .graph-card.active {{ display: block; }}
    .plot-container {{ width: 100%; height: 750px; }}
  </style>
</head>
<body>
  <header>
    <div class="title-area">
      <h1>Análisis de Redes Complejas (SNA) — Comunidades de Salud Mental</h1>
      <p>Pipeline Semántico-Afectivo Optimizado | BGE Embeddings + FAISS + Leiden (K = 6 a 15) | Sin MST</p>
    </div>
    <div class="sub-selector">
      <span>Subreddit:</span>
      <select id="subSelect" onchange="onSubredditChange(this.value)">
        {''.join([f'<option value="{s}">r/{s}</option>' for s in SUBREDDITS])}
      </select>
    </div>
  </header>

  <nav class="tabs">
    <button class="tab-btn active" data-tab="micro" onclick="onTabChange('micro')">
      🔬 1. Micro Orgánico
    </button>
    <button class="tab-btn" data-tab="macro" onclick="onTabChange('macro')">
      🌐 2. Macro Agregado (Sin MST)
    </button>
    <button class="tab-btn" data-tab="temporal" onclick="onTabChange('temporal')">
      ⏳ 3. Evolución Temporal del Grafo
    </button>
    <button class="tab-btn" data-tab="evolution" onclick="onTabChange('evolution')">
      📊 4. Matriz de Burbujas (% y Emociones)
    </button>
    <button class="tab-btn" data-tab="volume" onclick="onTabChange('volume')">
      📈 5. Evolución de Volumen por Comunidad
    </button>
  </nav>

  <main>
    <!-- PESTAÑA 1: MICRO ORGÁNICO -->
    <div id="pane_micro" class="tab-pane active">
      <div class="info-banner">
        <b>Pestaña 1: Visualización Micro Orgánica</b> — Muestreo cualitativo representativo altamente alineado al tópico de cada macrocomunidad y con diversidad afectiva balanceada.
        Disposición física Fruchterman-Reingold con separación multinivel. 
        <b>Atributos en tooltips:</b> Comentario textual completo de Reddit, ID original, grado local y emoción detectada.
      </div>
      {''.join([f'<div id="card_micro_{s}" class="graph-card"><div id="plot_micro_{s}" class="plot-container"></div></div>' for s in SUBREDDITS])}
    </div>

    <!-- PESTAÑA 2: MACRO AGREGADO -->
    <div id="pane_macro" class="tab-pane">
      <div class="info-banner">
        <b>Pestaña 2: Visualización Macro Agregada</b> — Red colapsada en macrocomunidades temáticas interpretables (K = 10 a 12).
        Nodos proporcionales a &radic;Volumen. <b>Aristas con Densidad de Corte Semántica real (&rho;):</b> <i>Se eliminaron totalmente las conexiones artificiales por MST</i>, reflejando exclusivamente puentes respaldados por los datos.
      </div>
      {''.join([f'<div id="card_macro_{s}" class="graph-card"><div id="plot_macro_{s}" class="plot-container"></div></div>' for s in SUBREDDITS])}
    </div>

    <!-- PESTAÑA 3: EVOLUCIÓN TEMPORAL DEL GRAFO (SNAPSHOTS CON SLIDER) -->
    <div id="pane_temporal" class="tab-pane">
      <div class="info-banner">
        <b>Pestaña 3: Evolución Temporal del Grafo Macro (Snapshots Trimestrales)</b> — 
        Visualización longitudinal que muestra qué comunidades se conectaban en cada trimestre (2019Q1 a 2022Q3). 
        <b>Anclaje espacial estable:</b> Las macrocomunidades conservan su ubicación física para evitar desorientación visual al desplazarse en el tiempo. 
        <b>Sin MST:</b> Los puentes activos corresponden exclusivamente a relaciones observadas con densidad de corte real (&rho;).
      </div>
      {''.join([f'<div id="card_temporal_{s}" class="graph-card"><div id="plot_temporal_{s}" class="plot-container"></div></div>' for s in SUBREDDITS])}
    </div>

    <!-- PESTAÑA 4: MATRIZ DE BURBUJAS -->
    <div id="pane_evolution" class="tab-pane">
      <div class="info-banner">
        <b>Pestaña 4: Matriz de Burbujas en Cuadrícula (Comunidades vs Periodos)</b> — 
        Filas = Macro-Comunidades, Columnas = Trimestres ordenados cronológicamente (2019Q1 a 2022Q3). 
        Color = Emoción dominante. Usa los botones superiores para alternar entre <b>% Dominancia Emocional</b> y <b>% Volumen del Trimestre (exclusivamente en porcentaje)</b>.
      </div>
      {''.join([f'<div id="card_evolution_{s}" class="graph-card" style="height: 860px;"><div id="plot_evolution_{s}" class="plot-container" style="height: 840px;"></div></div>' for s in SUBREDDITS])}
    </div>

    <!-- PESTAÑA 5: VOLUMEN LONGITUDINAL POR COMUNIDAD -->
    <div id="pane_volume" class="tab-pane">
      <div class="info-banner">
        <b>Pestaña 5: Evolución Longitudinal de Volumen Absoluto por Macro-Comunidad</b> — 
        Recuentos cronológicos exactos (2019Q1 a 2022Q3) desagregados exclusivamente por macrocomunidad (sin distorsión por la línea agregada del subreddit).
        <b>Resaltado cromático:</b> Cada serie interactiva, etiqueta y guía de seguimiento utiliza un tono cian eléctrico y la paleta identificativa de su propia comunidad, eliminando por completo los destellos y líneas blancas.
      </div>
      {''.join([f'<div id="card_volume_{s}" class="graph-card" style="height: 820px;"><div id="plot_volume_{s}" class="plot-container" style="height: 800px;"></div></div>' for s in SUBREDDITS])}
    </div>
  </main>

  <script>
    const DATA = {json.dumps(dashboard_data)};
    let currentTab = 'micro';
    let currentSub = '{SUBREDDITS[0]}';
    const rendered = {{}};

    function renderPlot(tab, sub) {{
      const key = `${{tab}}_${{sub}}`;
      if (rendered[key]) return;
      const card = document.getElementById(`card_${{tab}}_${{sub}}`);
      const container = document.getElementById(`plot_${{tab}}_${{sub}}`);
      if (!card || !container) return;

      const plotData = DATA[tab]?.[sub];
      if (plotData && plotData.data) {{
        Plotly.newPlot(container, plotData.data, plotData.layout, {{ responsive: true }});
        rendered[key] = true;
      }} else {{
        container.innerHTML = '<p style="padding: 40px; color: #94a3b8; text-align: center;">Gráfica no disponible para r/' + sub + '</p>';
      }}
    }}

    function updateVisibility() {{
      const allSubs = {json.dumps(SUBREDDITS)};
      const tabs = ['micro', 'macro', 'temporal', 'evolution', 'volume'];

      tabs.forEach(t => {{
        allSubs.forEach(s => {{
          const card = document.getElementById(`card_${{t}}_${{s}}`);
          if (card) {{
            if (t === currentTab && s === currentSub) {{
              card.classList.add('active');
              renderPlot(t, s);
              setTimeout(() => {{
                const container = document.getElementById(`plot_${{t}}_${{s}}`);
                if (container) Plotly.Plots.resize(container);
              }}, 80);
            }} else {{
              card.classList.remove('active');
            }}
          }}
        }});
      }});
    }}

    function onTabChange(tab) {{
      currentTab = tab;
      document.querySelectorAll('.tab-btn').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.tab === tab);
      }});
      document.querySelectorAll('.tab-pane').forEach(p => {{
        p.classList.toggle('active', p.id === `pane_${{tab}}`);
      }});
      updateVisibility();
    }}

    function onSubredditChange(sub) {{
      currentSub = sub;
      updateVisibility();
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      updateVisibility();
    }});
  </script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\n[OK] Dashboard V2.1 generado con éxito: '{output_html}'")

if __name__ == "__main__":
    generar_dashboard_completo()
