import yt_dlp

# for video link downloads
def download_link(url): 
    settings = {
        "format": "best",
        "outtmpl": "video.mp4",
        "quiet": True
    }

    yt_dlp.YoutubeDL(settings).download([url])