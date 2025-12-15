"""
Ejemplo de uso:
--------------
    python get_jsonld.py --url url-de-la-noticia

El script:
- Descarga la página HTML
- Extrae bloques <script type="application/ld+json">
- Busca el que tenga @type "NewsArticle" o "Article"
- Imprime el JSON-LD formateado

@author Rafael Valencia-García <valencia@um.es>
@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
"""

import argparse
import requests
from bs4 import BeautifulSoup
import json


def extract_jsonld (url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JSON-LD-scraper/1.0)"
    }

    resp = requests.get (url, headers=headers)
    resp.raise_for_status ()

    soup = BeautifulSoup (resp.text, "html.parser")

    ld_blocks = soup.find_all ("script", type = "application/ld+json")
    newsarticle_data = None

    for script in ld_blocks:
        try:
            data = json.loads (script.string)
        except json.JSONDecodeError:
            continue

        # Normalizamos a lista para buscar cómodamente
        items = data if isinstance (data, list) else [data]

        for item in items:
            if isinstance(item, dict) and item.get ("@type") in ("NewsArticle", "Article"):
                newsarticle_data = item
                break

        if newsarticle_data is not None:
            break

    return newsarticle_data


def main ():
    parser = argparse.ArgumentParser (
        description="Extrae el JSON-LD (tipo NewsArticle) de una noticia web."
    )
    parser.add_argument ("--url", 
        type = str, 
        required = False,
        help = "URL de la noticia a procesar",
        default = "https://www.eldiario.es/politica/aval-amnistia-abogado-general-tribunal-justicia-ue-allana-pp-camino-junts_129_12769531.html"
    )

    args = parser.parse_args()

    data = extract_jsonld (args.url)

    if data is None:
        print ("No se encontró ningún bloque JSON-LD de tipo NewsArticle/Article")
    else:
        print (json.dumps(data, indent = 4, ensure_ascii = False))


if __name__ == "__main__":
    main()
