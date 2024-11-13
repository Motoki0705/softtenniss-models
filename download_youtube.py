import yt_dlp

# ダウンロードしたいYouTube動画のURL
url = "https://www.youtube.com/watch?v=HNUmiiliD8M"

# オプション設定（MP4フォーマットを指定）
ydl_opts = {
    'format': 'mp4',
    'outtmpl': '保存先フォルダのパス/%(title)s.%(ext)s',  # 保存先フォルダとファイル名の指定
}

# 動画ダウンロード
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("ダウンロードが完了しました")
