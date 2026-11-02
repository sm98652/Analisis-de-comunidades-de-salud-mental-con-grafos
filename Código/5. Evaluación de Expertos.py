# -*- coding: utf-8 -*-
"""
Evaluación cualitativa por expertos (Mapeo exacto Español <-> Inglés):
1. Evaluación de tópicos (Marco AURA - Utilidad):
   - Cruza las 19 comunidades evaluadas en español en Google Forms con sus tópicos originales en inglés
     en 'Top5_Comentarios_BERTopic_Por_Subreddit.xlsx' (hoja 'Consolidado_Todos').
   - Calcula el promedio de los evaluadores en cada una de las 3 preguntas:
       * Claridad            <- [Coherencia entre las palabras clave]
       * Coherencia          <- [Correspondencia entre comentarios y palabras]
       * Valor analítico     <- [Valor analitico potencial]
   - Identifica qué tópicos NO tomar cuando el promedio en alguna pregunta es < 3.0.
   - Calcula el coeficiente Kappa de Cohen respuesta por respuesta (escala Likert 1-5, con ponderación cuadrática).

2. Evaluación de emociones (Ekman + Neutral + Otra vs. Modelo > 0.5):
   - Cruza los 20 comentarios en español del formulario con los 20 comentarios en inglés.
   - Calcula el match individual, por consenso y por categoría emocional con los psicólogos.
   - Calcula el coeficiente Kappa de Cohen aplanado para las emociones.
"""

from __future__ import annotations

import io
import math
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ==============================================================================
# ARCHIVOS Y PARÁMETROS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

ARCHIVO_RESPUESTAS = BASE_DIR / "Evaluacion de Comentarios por Tema (Responses).xlsx"
ARCHIVO_TOPICOS = BASE_DIR / "Top5_Comentarios_BERTopic_Por_Subreddit.xlsx"
ARCHIVO_EMOCIONES = BASE_DIR / "Comentarios - Evaluación Emociones.xlsx"
ARCHIVO_SALIDA = BASE_DIR / "Resultados_Evaluacion_Expertos.xlsx"
CARPETA_GRAFICAS = BASE_DIR / "graficas"

UMBRAL_TOPICO = 3.0      # No tomar tópico si el promedio en cualquier pregunta es < 3.0
UMBRAL_EMOCION = 0.5     # Tomar solo emociones del modelo con probabilidad > 0.5


# ==============================================================================
# 1. MAPEO EXACTO ESPAÑOL (FORMULARIO) -> INGLÉS (BERTOPIC)
# ==============================================================================
MAPEO_TOPICOS_ES_A_EN: List[Tuple[str, str, str]] = [
    ("pfizer, vacuna", "Anxiety", "C0"),
    ("marketing, trabajo", "Anxiety", "C1"),
    ("electrocardiograma, corazon", "Anxiety", "C2"),
    ("espacio para hablar, terapeuta", "Anxiety", "C5"),
    ("deja, vu, esquizofrenia", "Anxiety", "C6"),
    ("instrumental, lofi", "Anxiety", "C7"),
    ("instagram, desactivar, minecraft", "Anxiety", "C9"),
    ("causas de origen, johann hari", "depression", "C2"),
    ("mirtazapina, bupropion", "depression", "C3"),
    ("autolesiones, cortar", "depression", "C4"),
    ("inutilidad, anhedonia", "depression", "C6"),
    ("musica melancolica, radiohead", "depression", "C7"),
    ("ensayos clinicos, ketamina", "depression", "C8"),
    ("beca, consejero", "SuicideWatch", "C0"),
    ("desolacion, festivos", "SuicideWatch", "C1"),
    ("discord", "SuicideWatch", "C3"),
    ("insomnio, pildoras de dormir", "SuicideWatch", "C9"),
    ("identidad, transfobia", "SuicideWatch", "C7"),
    ("culpa, existencial", "SuicideWatch", "C8"),
]

MAPEO_DIMENSIONES_FORM: Dict[str, str] = {
    "coherencia entre las palabras clave": "Claridad",
    "correspondencia entre comentarios y palabras": "Coherencia",
    "valor analitico potencial": "Valor_Analitico",
}


# ==============================================================================
# 2. MAPEO EXACTO DE LOS 20 COMENTARIOS DE EMOCIONES (ESPAÑOL -> ORDEN 1..20)
# ==============================================================================
MAPEO_COMENTARIOS_EMOCIONES: List[Tuple[int, str]] = [
    (1, "cuando me despierto de una larga siesta"),
    (2, "llevo mucho tiempo deprimido y siempre he pensado que el suicidio era una opcion"),
    (3, "voy a marcharme en paz"),
    (4, "acabo de decidir el metodo de suicidio"),
    (5, "deberia volver a cortarme"),
    (6, "me aterra la idea de renacer"),
    (7, "es mejor avisar a la gente de que te vas a suicidar pronto"),
    (8, "me siento como un capullo"),
    (9, "me acabo de tomar 15 mg de diazepam"),
    (10, "alguien podria hablar conmigo, por favor"),
    (11, "quiero decirle a alguien que le quiero"),
    (12, "tengo mucho miedo y estoy muy nerviosa sin motivo alguno"),
    (13, "es esto disociacion o lo contrario"),
    (14, "estaba conduciendo por un aparcamiento"),
    (15, "creo que voy a mejorar"),
    (16, "creo que he olvidado como es sentir amor"),
    (17, "mi perro acaba de morir"),
    (18, "siento si esto no esta permitido aqui"),
    (19, "he dejado de hablar con la mayoria de mis amigos de la uni"),
    (20, "cada vez que oigo llover, me aterra que el techo se derrumbe"),
]

MAPA_EMOCIONES_ES_EN: Dict[str, str] = {
    "ira": "anger", "enojo": "anger", "rabia": "anger", "enfado": "anger",
    "asco": "disgust", "disgusto": "disgust", "repulsion": "disgust",
    "miedo": "fear", "temor": "fear",
    "alegria": "joy", "felicidad": "joy",
    "tristeza": "sadness",
    "sorpresa": "surprise",
    "neutral": "neutral", "neutra": "neutral", "neutro": "neutral",
    "otra": "other", "otro": "other",
}

COLUMNAS_PROB_MODELO: Dict[str, str] = {
    "anger": "Prob_IA_anger",
    "fear": "Prob_IA_fear",
    "joy": "Prob_IA_joy",
    "sadness": "Prob_IA_sadness",
    "surprise": "Prob_IA_surprise",
    "disgust": "Prob_IA_disgust",
    "neutral": "Prob_IA_neutral",
}


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def normalizar(texto: object) -> str:
    if pd.isna(texto):
        return ""
    s = str(texto).strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def extraer_palabras_clave_es(encabezado: str) -> str:
    m = re.search(r"Palabras clave:\s*(.+?)(?:\n|Comentarios:)", str(encabezado), flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else str(encabezado)[:80].strip()


def resolver_topico_ingles(encabezado: str) -> Tuple[str, str]:
    kw_es_norm = normalizar(extraer_palabras_clave_es(encabezado))
    for prefijo_es, subreddit, topico_id in MAPEO_TOPICOS_ES_A_EN:
        if kw_es_norm.startswith(prefijo_es) or prefijo_es in kw_es_norm:
            return subreddit, topico_id
    return "NO_MAPEADO", "NO_MAPEADO"


def parsear_emociones_celda(celda: object) -> Tuple[Set[str], List[str]]:
    if pd.isna(celda) or str(celda).strip() == "":
        return set(), []
    partes = [p.strip() for p in re.split(r"[,;/]+", str(celda)) if p.strip()]
    resultado: Set[str] = set()
    libres_otra: List[str] = []
    for p in partes:
        p_norm = normalizar(p)
        if p_norm in MAPA_EMOCIONES_ES_EN:
            resultado.add(MAPA_EMOCIONES_ES_EN[p_norm])
        else:
            encontrada = False
            for k_es, v_en in MAPA_EMOCIONES_ES_EN.items():
                if re.search(rf"\b{re.escape(k_es)}\b", p_norm):
                    resultado.add(v_en)
                    encontrada = True
            if not encontrada:
                resultado.add("other")
                libres_otra.append(p)
    return resultado, libres_otra


def abrir_excel_seguro(ruta: Path) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(ruta)
    except PermissionError:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                copia = Path(tmpdir) / ruta.name
                shutil.copy2(ruta, copia)
                contenido = copia.read_bytes()
            return pd.ExcelFile(io.BytesIO(contenido))
        except PermissionError as exc:
            raise PermissionError(
                f"No se pudo leer '{ruta.name}' porque está bloqueado por otro programa."
            ) from exc


# ==============================================================================
# PARTE 1: EVALUACIÓN DE TÓPICOS Y KAPPA (RESPUESTA POR RESPUESTA / LIKERT 1-5)
# ==============================================================================
def cargar_catalogo_bertopic(ruta: Path) -> pd.DataFrame:
    xl = abrir_excel_seguro(ruta)
    hoja = "Consolidado_Todos" if "Consolidado_Todos" in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(hoja)
    return (
        df[[
            "Subreddit", "Topico_ID", "Topico_Numero",
            "Palabras_Clave_BERTopic", "Interpretacion_Psicologica",
            "Total_Comentarios_Topico",
        ]]
        .drop_duplicates(subset=["Subreddit", "Topico_ID"])
        .reset_index(drop=True)
    )


def evaluar_topicos(
    df_resp: pd.DataFrame, df_catalogo: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    bloques: Dict[str, Dict[str, str]] = {}
    orden_claves: List[str] = []

    for col in df_resp.columns:
        col_str = str(col)
        if not normalizar(col_str).startswith("palabras clave:"):
            continue
        corchetes = re.findall(r"\[([^\]]+)\]", col_str)
        if not corchetes:
            continue
        dim_aura = MAPEO_DIMENSIONES_FORM.get(normalizar(corchetes[-1]))
        if dim_aura is None:
            continue
        base_topico = re.sub(r"\s*\[[^\]]+\]\s*$", "", col_str).strip()
        if base_topico not in bloques:
            bloques[base_topico] = {}
            orden_claves.append(base_topico)
        bloques[base_topico][dim_aura] = col_str

    idx_catalogo = {
        (str(r["Subreddit"]).strip().lower(), str(r["Topico_ID"]).strip().upper()): r
        for _, r in df_catalogo.iterrows()
    }
    usados_catalogo: Set[Tuple[str, str]] = set()

    filas = []
    evaluador_1_ratings = []
    evaluador_2_ratings = []
    tiene_dos_evaluadores = len(df_resp) >= 2

    for num_orden, base_texto in enumerate(orden_claves, start=1):
        cols_dim = bloques[base_texto]
        kw_es = extraer_palabras_clave_es(base_texto)
        subreddit, topico_id = resolver_topico_ingles(base_texto)
        clave_cat = (subreddit.lower(), topico_id.upper())

        kw_en, interp_psic, total_coms = "", "", np.nan
        if clave_cat in idx_catalogo:
            usados_catalogo.add(clave_cat)
            info_cat = idx_catalogo[clave_cat]
            subreddit = str(info_cat["Subreddit"])
            topico_id = str(info_cat["Topico_ID"])
            kw_en = str(info_cat["Palabras_Clave_BERTopic"])
            interp_psic = str(info_cat["Interpretacion_Psicologica"])
            total_coms = int(info_cat["Total_Comentarios_Topico"])

        s_clar = pd.to_numeric(df_resp[cols_dim["Claridad"]], errors="coerce")
        s_cohe = pd.to_numeric(df_resp[cols_dim["Coherencia"]], errors="coerce")
        s_valo = pd.to_numeric(df_resp[cols_dim["Valor_Analitico"]], errors="coerce")

        if tiene_dos_evaluadores:
            for s_dim in [s_clar, s_cohe, s_valo]:
                if pd.notna(s_dim.iloc[0]) and pd.notna(s_dim.iloc[1]):
                    evaluador_1_ratings.append(int(s_dim.iloc[0]))
                    evaluador_2_ratings.append(int(s_dim.iloc[1]))

        p_clar = float(s_clar.mean())
        p_cohe = float(s_cohe.mean())
        p_valo = float(s_valo.mean())
        p_glob = float(np.nanmean([p_clar, p_cohe, p_valo]))

        preguntas_bajas = []
        if p_clar < UMBRAL_TOPICO: preguntas_bajas.append(f"Claridad ({p_clar:.2f})")
        if p_cohe < UMBRAL_TOPICO: preguntas_bajas.append(f"Coherencia ({p_cohe:.2f})")
        if p_valo < UMBRAL_TOPICO: preguntas_bajas.append(f"Valor analítico ({p_valo:.2f})")

        filas.append({
            "Orden_Form": num_orden,
            "Subreddit": subreddit,
            "Topico_ID": topico_id,
            "Palabras_Clave_Ingles_BERTopic": kw_en,
            "Palabras_Clave_Espanol_Form": kw_es,
            "Interpretacion_Psicologica": interp_psic,
            "Total_Comentarios_Corpus": total_coms,
            "Prom_Claridad": round(p_clar, 2),
            "Prom_Coherencia": round(p_cohe, 2),
            "Prom_Valor_Analitico": round(p_valo, 2),
            "Prom_Global": round(p_glob, 2),
            "Decision": "NO TOMAR" if preguntas_bajas else "TOMAR",
            "Preguntas_Menor_a_3": "; ".join(preguntas_bajas) if preguntas_bajas else "Ninguna",
        })

    df_evaluados = pd.DataFrame(filas)
    df_no_incluidos = df_catalogo[
        [
            (str(r["Subreddit"]).strip().lower(), str(r["Topico_ID"]).strip().upper()) not in usados_catalogo
            for _, r in df_catalogo.iterrows()
        ]
    ].reset_index(drop=True)

    # Cálculo de Kappa de Cohen ponderado para respuestas Likert (1 a 5) en Tópicos
    kappa_topicos_linear, kappa_topicos_quad = float('nan'), float('nan')
    if tiene_dos_evaluadores and len(evaluador_1_ratings) > 0:
        try:
            kappa_topicos_linear = float(cohen_kappa_score(evaluador_1_ratings, evaluador_2_ratings, weights='linear'))
            kappa_topicos_quad = float(cohen_kappa_score(evaluador_1_ratings, evaluador_2_ratings, weights='quadratic'))
        except Exception:
            pass

    return df_evaluados, df_no_incluidos, kappa_topicos_linear, kappa_topicos_quad


# ==============================================================================
# PARTE 2: EVALUACIÓN DE EMOCIONES Y KAPPA APLANADO
# ==============================================================================
def encontrar_columna_comentario_es(df_resp: pd.DataFrame, fragmento_es: str) -> Optional[str]:
    frag_norm = normalizar(fragmento_es)
    for col in df_resp.columns:
        col_norm = normalizar(col)
        if not col_norm.startswith("palabras clave:") and frag_norm in col_norm:
            return str(col)
    return None


def evaluar_emociones(
    df_resp: pd.DataFrame, df_emo_modelo: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    df_mod = df_emo_modelo.sort_values("Orden").reset_index(drop=True)
    n_expertos = len(df_resp)
    umbral_consenso = max(2, math.ceil(n_expertos / 2)) if n_expertos >= 2 else 1

    todas_categorias = list(COLUMNAS_PROB_MODELO.keys()) + ["other"]
    conteo_cat = {e: {"modelo_gt_05": 0, "consenso_exp": 0, "union_exp": 0, "match_consenso": 0, "match_union": 0, "votos_totales": 0} for e in todas_categorias}

    filas_detalle = []
    pares_individuales = []
    
    emociones_lista = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral", "other"]
    mat_r1, mat_r2 = [], []
    tiene_dos_evaluadores = n_expertos >= 2

    for orden_num, frag_es in MAPEO_COMENTARIOS_EMOCIONES:
        col_form = encontrar_columna_comentario_es(df_resp, frag_es)
        if col_form is None:
            raise KeyError(f"No se encontró en el formulario el comentario #{orden_num}: '{frag_es}'")

        fila_m = df_mod.loc[df_mod["Orden"] == orden_num].iloc[0]
        probs = {emo: float(fila_m[col_p]) for emo, col_p in COLUMNAS_PROB_MODELO.items()}
        pred_gt_05: Set[str] = {emo for emo, p in probs.items() if p > UMBRAL_EMOCION}
        pred_ordenadas = sorted(pred_gt_05, key=lambda e: probs[e], reverse=True)
        top1_emo, top1_prob = max(probs.items(), key=lambda kv: kv[1])

        respuestas_exp: List[Set[str]] = []
        respuestas_raw: List[str] = []
        textos_otra: List[str] = []
        votos: Dict[str, int] = {e: 0 for e in todas_categorias}

        celdas_comentario = list(df_resp[col_form])
        for celda in celdas_comentario:
            respuestas_raw.append(str(celda) if not pd.isna(celda) else "")
            conj_e, libres = parsear_emociones_celda(celda)
            if conj_e:
                respuestas_exp.append(conj_e)
                textos_otra.extend(libres)
                for e in conj_e:
                    votos[e] += 1
                    conteo_cat[e]["votos_totales"] += 1

        if tiene_dos_evaluadores and len(celdas_comentario) >= 2:
            conj_r1, _ = parsear_emociones_celda(celdas_comentario[0])
            conj_r2, _ = parsear_emociones_celda(celdas_comentario[1])
            mat_r1.append([1 if emo in conj_r1 else 0 for emo in emociones_lista])
            mat_r2.append([1 if emo in conj_r2 else 0 for emo in emociones_lista])

        n_validos = len(respuestas_exp)
        consenso_mayoria: Set[str] = {e for e, v in votos.items() if v >= umbral_consenso}
        union_expertos: Set[str] = set().union(*respuestas_exp) if respuestas_exp else set()

        m_indiv = [metricas_match(pred_gt_05, r_e) for r_e in respuestas_exp]
        pares_individuales.extend(m_indiv)

        n_psic_coinciden = sum(int(m["al_menos_una"]) for m in m_indiv)
        pct_psic_coinciden = (n_psic_coinciden / n_validos * 100.0) if n_validos > 0 else 0.0
        m_consenso = metricas_match(pred_gt_05, consenso_mayoria)
        match_algun_experto = len(pred_gt_05 & union_expertos) > 0

        for e in todas_categorias:
            en_m = e in pred_gt_05
            en_c = e in consenso_mayoria
            en_u = e in union_expertos
            if en_m: conteo_cat[e]["modelo_gt_05"] += 1
            if en_c: conteo_cat[e]["consenso_exp"] += 1
            if en_u: conteo_cat[e]["union_exp"] += 1
            if en_m and en_c: conteo_cat[e]["match_consenso"] += 1
            if en_m and en_u: conteo_cat[e]["match_union"] += 1

        votos_txt = ", ".join(f"{e} ({v}/{n_validos})" for e, v in sorted(votos.items(), key=lambda kv: kv[1], reverse=True) if v > 0)
        modelo_gt_05_txt = ", ".join(f"{e} ({probs[e]:.4f})" for e in pred_ordenadas) if pred_ordenadas else f"Ninguna > 0.5 [Top1: {top1_emo}]"

        registro = {
            "Orden": orden_num,
            "Subreddit": str(fila_m["Subreddit"]),
            "Emociones_Modelo_Mayor_0.5": modelo_gt_05_txt,
            "Votos_Psicologos_EN": votos_txt,
            "Consenso_Mayoria_EN": ", ".join(sorted(consenso_mayoria)) if consenso_mayoria else "Sin consenso",
            "Psicologos_Que_Hacen_Match": f"{n_psic_coinciden}/{n_validos} ({pct_psic_coinciden:.1f}%)",
            "Match_Con_Consenso_Mayoria": "SÍ" if m_consenso["al_menos_una"] == 1.0 else "NO",
            "Jaccard_Promedio_Individual": round(float(np.mean([m["jaccard"] for m in m_indiv])) if m_indiv else 0.0, 3),
        }
        for i_exp, r_txt in enumerate(respuestas_raw, start=1):
            registro[f"Resp_Psicologo_{i_exp}_ES"] = r_txt
        filas_detalle.append(registro)

    df_detalle = pd.DataFrame(filas_detalle)

    kappa_emociones = float('nan')
    if tiene_dos_evaluadores and len(mat_r1) > 0:
        try:
            arr_r1 = np.array(mat_r1).flatten()
            arr_r2 = np.array(mat_r2).flatten()
            kappa_emociones = float(cohen_kappa_score(arr_r1, arr_r2, labels=[0, 1]))
        except Exception:
            kappa_emociones = 0.0

    resumen = [
        {"Metrica": "Total de comentarios evaluados", "Resultado": len(df_detalle), "Interpretacion": "20 comentarios cruzados"},
        {"Metrica": "Coeficiente Kappa de Cohen (Emociones - Aplanado)", "Resultado": round(kappa_emociones, 4), "Interpretacion": "Acuerdo inter-evaluador multietiqueta en los 20 comentarios"},
        {"Metrica": "Match con Consenso de mayoría", "Resultado": f"{(df_detalle['Match_Con_Consenso_Mayoria'] == 'SÍ').sum()}/{len(df_detalle)}", "Interpretacion": "Comentarios con match en consenso"},
    ]
    df_resumen = pd.DataFrame(resumen)

    filas_cat = []
    for emo, d in conteo_cat.items():
        m_cnt = d["modelo_gt_05"]
        c_cnt = d["consenso_exp"]
        mc_cnt = d["match_consenso"]
        filas_cat.append({
            "Emocion": emo,
            "Predicha_Modelo_Mayor_0.5": m_cnt,
            "En_Consenso_Psicologos": c_cnt,
            "Match_Modelo_y_Consenso": mc_cnt,
        })
    df_por_emocion = pd.DataFrame(filas_cat)

    return df_detalle, df_resumen, df_por_emocion, kappa_emociones


def metricas_match(pred_modelo: Set[str], resp_experto: Set[str]) -> Dict[str, float]:
    inter = pred_modelo & resp_experto
    union = pred_modelo | resp_experto
    return {
        "exacto": 1.0 if pred_modelo == resp_experto else 0.0,
        "al_menos_una": 1.0 if len(inter) > 0 else 0.0,
        "jaccard": len(inter) / len(union) if len(union) > 0 else 1.0,
        "f1": (2 * len(inter) / (len(pred_modelo) + len(resp_experto))) if (len(pred_modelo) + len(resp_experto)) > 0 else 0.0,
    }


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
def main() -> None:
    xl_resp = abrir_excel_seguro(ARCHIVO_RESPUESTAS)
    df_resp = xl_resp.parse(xl_resp.sheet_names[0]).dropna(how="all").reset_index(drop=True)
    df_catalogo = cargar_catalogo_bertopic(ARCHIVO_TOPICOS)
    xl_emo = abrir_excel_seguro(ARCHIVO_EMOCIONES)
    df_emo_modelo = xl_emo.parse(xl_emo.sheet_names[0]).dropna(how="all").reset_index(drop=True)

    print(f"Evaluadores encontrados en Responses.xlsx: {len(df_resp)}\n")

    # 1. Evaluación de Tópicos & Kappa respuesta por respuesta (Likert 1-5)
    df_topicos, df_no_incluidos, k_lin, k_quad = evaluar_topicos(df_resp, df_catalogo)
    print("=" * 95)
    print(f"1. EVALUACIÓN DE TÓPICOS & KAPPA (Respuesta por respuesta / Likert 1-5)")
    print("=" * 95)
    print(f"Kappa Ponderado Lineal (Tópicos): {k_lin:.4f}")
    print(f"Kappa Ponderado Cuadrático (Tópicos): {k_quad:.4f}\n")

    # 2. Evaluación de Emociones & Kappa aplanado
    df_emo_detalle, df_emo_resumen, df_emo_cat, kappa_emociones = evaluar_emociones(df_resp, df_emo_modelo)
    print("=" * 95)
    print(f"2. EVALUACIÓN DE EMOCIONES & KAPPA APLANADO")
    print("=" * 95)
    print(f"Coeficiente Kappa de Cohen (Emociones): {kappa_emociones:.4f}\n")
    print(df_emo_resumen.to_string(index=False))

    # Guardar en Excel
    with pd.ExcelWriter(ARCHIVO_SALIDA, engine="openpyxl") as writer:
        df_topicos.to_excel(writer, sheet_name="Topicos_Evaluados", index=False)
        df_emo_resumen.to_excel(writer, sheet_name="Emociones_Resumen_Kappa", index=False)
        df_emo_detalle.to_excel(writer, sheet_name="Emociones_Detalle", index=False)

    print(f"\nResultados guardados en: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    main()