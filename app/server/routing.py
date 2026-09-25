"""Decisión de la banda: a qué ruta va cada huevo según la predicción.

El diferencial (dano_v1) convierte la decisión binaria sano/rajado en tres salidas útiles:
un huevo con una grieta pequeña no se pierde, se aprovecha en reproceso industrial.
"""

LEVE_MAX = 0.15  # mismo umbral "leve" de MODELO_IO.md

ROUTES = {
    "empaque": {
        "name": "Empaque",
        "rule": "Todos los huevos detectados son Intact",
        "desc": "Huevo sano: sigue a empaque para venta en fresco.",
    },
    "industria": {
        "name": "Industria",
        "rule": f"Crack con gravedad < {int(LEVE_MAX * 100)} % (leve)",
        "desc": "Grieta pequeña: reproceso industrial (huevo líquido pasteurizado).",
    },
    "descarte": {
        "name": "Descarte",
        "rule": f"Crack con gravedad ≥ {int(LEVE_MAX * 100)} % (media o grave)",
        "desc": "Daño extenso: se retira de la línea.",
    },
    "revision": {
        "name": "Revisión manual",
        "rule": "No se detectó ningún huevo",
        "desc": "El detector no encontró huevo con confianza suficiente.",
    },
}


def decide_route(summary: dict) -> str:
    if summary["n_eggs"] == 0:
        return "revision"
    if summary["n_crack"] == 0:
        return "empaque"
    sev = summary["max_severity"] or 0.0
    return "industria" if sev < LEVE_MAX else "descarte"
