"""
Ejemplo para acceder a la API de REDDIT con PRAW

PRAW es un wrapper que facilita la interacción con la plataforma.
Para usar la API, es necesario crear una aplicación en https://www.reddit.com/prefs/apps
Se debe seleccionar la opción "script" y proporcionar un nombre 
identificativo (por ejemplo: dl-pln-2025-tu-correo-umu). También hay que 
especificar un redirect uri, para lo cual podéis usar: http://www.pln.inf.um.es.

La creación de la aplicación os proporcionará:
- un client_id,
- un client_secret,
- el usuario de Reddit asociado a la aplicación.

que tenéis que poner en el script

@author Rafael Valencia-García <valencia@um.es>
@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
"""

import praw
import json


# @var reddit Configure data to retrieve
reddit = praw.Reddit (
    client_id = "",
    client_secret = "",
    user_agent = "",
)

# Función para obtener publicaciones por flair en un subreddit
def get_posts_by_flair (subreddit_name, flair, limit = 10, max_comments = 10):
    
    # Cargamos el hilo
    subreddit = reddit.subreddit (subreddit_name)
    
    
    # Buscar publicaciones con el flair especificado
    if flair:
        posts = subreddit.search (f'flair:"{flair}"', limit=limit)
    
    else:
        posts = subreddit.new (limit = limit)
    
    
    # Almacenamos los resultados
    results = []
    for post in posts:
        
        # Cargamos los comentarios
        post.comments.replace_more (limit = 0)
        comments = post.comments.list()[:max_comments]
        
        
        # Extraer información de cada comentario
        comment_list = []
        for comment in comments:
            comment_list.append ({
                'author': comment.author.name if comment.author else "Deleted",
                'body': comment.body,
                'score': comment.score,
                'created_utc': comment.created_utc
            })
        
        results.append ({
            'title': post.title,
            'author': post.author.name,
            'url': post.url,
            'score': post.score,
            'flair': post.link_flair_text,
            'created_utc': post.created_utc,
            'description': post.selftext,
            'comments': comment_list
        })
    
    return results


# Ejemplo de uso
subreddit_name = "ArgentinaCocina"
flair = "" 
limit = 10


# Llamar a la función para obtener publicaciones
posts = get_posts_by_flair (subreddit_name, flair, limit)


# Mostrar los resultados
for idx, post in enumerate (posts, start=1):
    print (json.dumps (post, sort_keys = True, indent = 4, ensure_ascii = False).replace ('\\n','\n'))
