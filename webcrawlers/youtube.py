#!/usr/bin/env python3
"""
Ejemplo de uso
--------------
1) Obtener el channelId de un canal  (desde la URL o vía API):
https://developers.google.com/youtube/v3/quickstart/python?hl = es-419

2) Ejemplo de ejecición
    python youtube.py \
        --api-key TU_API_KEY \
        --channel @DotCSV \
        --max-videos 5 \
        --max-comments 20

El script:
- Lista los últimos N vídeos de un canal de YouTube
- Para cada vídeo, recupera hasta M comentarios
- Imprime la información en formato JSON  (uno por línea: video + comentarios)

@author Rafael Valencia-García <valencia@um.es>
@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
"""

"""
Ejemplo de uso
--------------
"""

import argparse
import json
import requests


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


def resolve_handle_to_channel_id (api_key, handle):
    """
    Convierte un handle de YouTube  (@usuario) a un channelId usando la API de búsqueda.
    """
    handle_query = handle.lstrip ("@")

    url = f"{YOUTUBE_API_BASE}/search"
    params = {
        "part": "snippet",
        "q": handle_query,
        "type": "channel",
        "maxResults": 1,
        "key": api_key,
    }

    resp = requests.get (url, params = params)
    resp.raise_for_status ()
    data = resp.json ()

    items = data.get ("items", [])
    if not items:
        raise ValueError (f"No se encontró ningún canal con handle '{handle}'")

    return items[0]["snippet"]["channelId"]


def get_channel_uploads_playlist_id (api_key, channel_id):
    url = f"{YOUTUBE_API_BASE}/channels"
    params = {"part": "contentDetails", "id": channel_id, "key": api_key}
    resp = requests.get (url, params = params)
    resp.raise_for_status ()
    data = resp.json ()

    items = data.get ("items", [])
    if not items:
        raise ValueError (f"No se encontró el canal con id = {channel_id}")

    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_videos_from_playlist (api_key, playlist_id, max_videos):
    url = f"{YOUTUBE_API_BASE}/playlistItems"
    videos = []
    params = {
        "part": "snippet",
        "playlistId": playlist_id,
        "maxResults": 50,
        "key": api_key,
    }

    while True:
        resp = requests.get (url, params = params)
        resp.raise_for_status ()
        data = resp.json ()

        for item in data.get ("items", []):
            sn = item["snippet"]
            video_id = sn["resourceId"]["videoId"]
            videos.append ({
                "video_id": video_id,
                "title": sn.get ("title", "")
            })
            if len (videos) >=  max_videos:
                return videos

        next_page = data.get ("nextPageToken")
        if not next_page:
            break
        params["pageToken"] = next_page

    return videos


def get_video_comments (api_key, video_id, max_comments):
    url = f"{YOUTUBE_API_BASE}/commentThreads"
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": api_key,
        "order": "relevance",
    }

    while True:
        resp = requests.get (url, params = params)
        if resp.status_code !=  200:
            break

        data = resp.json ()
        for item in data.get ("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            comments.append ({
                "author": sn.get ("authorDisplayName", ""),
                "text": sn.get ("textDisplay", ""),
                "published_at": sn.get ("publishedAt", ""),
                "like_count": sn.get ("likeCount", 0)
            })
            if len (comments) >=  max_comments:
                return comments

        next_page = data.get ("nextPageToken")
        if not next_page:
            break
        params["pageToken"] = next_page

    return comments


def main ():
    parser = argparse.ArgumentParser (
        description = "Recupera vídeos y comentarios de un canal de YouTube."
    )
    
    parser.add_argument ("--api-key", 
        required = True, help = "API key de YouTube Data API v3"
    )
    
    parser.add_argument ("--channel", 
        required = True,
        help = "ID del canal o handle (@usuario), ej: @DotCSV"
    )
    
    parser.add_argument ("--max-videos", 
        type = int, 
        default = 5
    )
    parser.add_argument ("--max-comments", 
        type = int, 
        default = 50
    )

    args = parser.parse_args ()
    
    
    # Detectar si es handle
    if args.channel.startswith ("@"):
        channel_id = resolve_handle_to_channel_id (args.api_key, args.channel)
    else:
        channel_id = args.channel

    uploads_playlist = get_channel_uploads_playlist_id (args.api_key, channel_id)
    videos = get_videos_from_playlist (args.api_key, uploads_playlist, args.max_videos)

    for v in videos:
        comments = get_video_comments (args.api_key, v["video_id"], args.max_comments)
        out = {
            "video_id": v["video_id"],
            "title": v["title"],
            "comments": comments,
        }
        print (json.dumps (out, sort_keys = True, indent = 4, ensure_ascii = False)
            .replace ('\\n','\n'))


if __name__  == "__main__":
    main ()
