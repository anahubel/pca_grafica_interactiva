import numpy as np
import pandas as pd

def apply_recodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica recodificaciones y ordena categorías (equivalente a tu RStudio).
    Devuelve una copia para no mutar el df original.
    """
    out = df.copy()

    # -------------------------
    # segmentacion_tamano_2_grp
    # -------------------------
    if "segmentacion_tamano_2" in out.columns:
        out["segmentacion_tamano_2_grp"] = np.select(
            [
                out["segmentacion_tamano_2"] == "Menos de 1M€",
                out["segmentacion_tamano_2"] == "De 1 - 5M€",
                out["segmentacion_tamano_2"].isin(["De 5 - 20M€", "Más de 20M€"]),
            ],
            ["Menos de 1M€", "De 1 - 5M€", "Más de 5M€"],
            default=None,
        )
        out["segmentacion_tamano_2_grp"] = pd.Categorical(
            out["segmentacion_tamano_2_grp"],
            categories=["Menos de 1M€", "De 1 - 5M€", "Más de 5M€"],
            ordered=True,
        )

    # -------------------------
    # segmentacion_tamano_1_grp
    # -------------------------
    if "segmentacion_tamano_1" in out.columns:
        out["segmentacion_tamano_1_grp"] = np.select(
            [
                out["segmentacion_tamano_1"] == "Menos de 1M€",
                out["segmentacion_tamano_1"] == "De 1 a 5M€",
                out["segmentacion_tamano_1"] == "De 5 a 10M€",
                out["segmentacion_tamano_1"].isin(["De 10 - 50M€", "De 50 - 100M€", "Más de 100M€"]),
            ],
            ["Menos de 1M€", "De 1 a 5M€", "De 5 a 10M€", "Más de 10M€"],
            default=None,
        )
        out["segmentacion_tamano_1_grp"] = pd.Categorical(
            out["segmentacion_tamano_1_grp"],
            categories=["Menos de 1M€", "De 1 a 5M€", "De 5 a 10M€", "Más de 10M€"],
            ordered=True,
        )

    # -------------------------
    # localidad_grp
    # -------------------------
    if "localidad" in out.columns:
        localidad_map = {
            "AGRES": "El comtat",
            "ALCOCER DE PLANES": "El comtat",
            "BENASAU": "El comtat",
            "COCENTAINA": "El comtat",
            "MURO DE ALCOY": "El comtat",
            "GAIANES": "El comtat",
            "ALBAIDA": "La Vall d'Albaida",
            "AGULLENT": "La Vall d'Albaida",
            "ONTINYENT": "La Vall d'Albaida",
            "BENIGANIM": "La Vall d'Albaida",
            "BOCAIRENT": "La Vall d'Albaida",
            "ALCOY/ALCOI": "L'Alcoià",
            "BANYERES DE MARIOLA": "L'Alcoià",
            "CASTALLA": "L'Alcoià",
            "ONIL": "L'Alcoià",
        }
        out["localidad_grp"] = out["localidad"].map(localidad_map)

    # -------------------------
    # sector_grp
    # -------------------------
    if "sector" in out.columns:
        sector_map = {
            "Alimentación": "Alimentación",
            "Juguete": "Juguete",
            "Madera": "Madera",
            "Metal": "Metal",
            "Papel y cartón": "Papel y cartón",
            "Papel y Cartón": "Papel y cartón",
            "Plástico": "Plástico",
            "Textil": "Textil",
        }
        out["sector_grp"] = out["sector"].map(sector_map).fillna("Otros")

    # -------------------------
    # crece (ordinal)
    # -------------------------
    if "crece" in out.columns:
        out["crece"] = pd.Categorical(
            out["crece"],
            categories=["Decrece", "Se mantiene", "Crece"],
            ordered=True,
        )

    # -------------------------
    # segmentacion_de_antiguedad (ordinal)
    # -------------------------
    if "segmentacion_de_antiguedad" in out.columns:
        out["segmentacion_de_antiguedad"] = pd.Categorical(
            out["segmentacion_de_antiguedad"],
            categories=[
                "Menor de 10 años",
                "Entre 10 y 25 años",
                "Entre 25 y 50 años",
                "Mayor de 50 años",
            ],
            ordered=True,
        )

    return out