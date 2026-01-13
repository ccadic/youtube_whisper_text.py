#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import subprocess
import shlex
import queue
import tkinter as tk
from tkinter import filedialog, ttk
from datetime import datetime
from shutil import which

# ==============================
# CONFIG
# ==============================
WHISPER_PROJECT_DIR = os.path.expanduser("~/whisper-gpu")

FFMPEG_BIN = "ffmpeg"
YTDLP_BIN = "yt-dlp"

DEFAULT_MODEL = "medium"
DEFAULT_LANG = "fr"   # fr / en / es / auto


def which_ok(cmd: str) -> bool:
    return which(cmd) is not None


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


class Logger:
    """Thread-safe logger to Tk Text via queue + after(). Also writes to a file if set."""
    def __init__(self, text_widget: tk.Text):
        self.text = text_widget
        self.q = queue.Queue()
        self.logfile_path = None
        self.text.configure(state="disabled")
        self._pump()

    def set_logfile(self, path: str):
        self.logfile_path = path

    def _write_file(self, msg: str):
        if not self.logfile_path:
            return
        try:
            with open(self.logfile_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def _pump(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.text.configure(state="normal")
                self.text.insert("end", msg + "\n")
                self.text.see("end")
                self.text.configure(state="disabled")
        except queue.Empty:
            pass
        self.text.after(60, self._pump)

    def log(self, msg: str):
        self._write_file(msg)
        self.q.put(msg)


def _pretty_cmd(cmd):
    if isinstance(cmd, list):
        return " ".join(shlex.quote(str(x)) for x in cmd)
    return str(cmd)


def run_cmd_stream(logger: Logger, cmd, cwd=None, env=None):
    """Run command and stream stdout+stderr to logger."""
    logger.log(f"$ {_pretty_cmd(cmd)}")

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    out_lines = []
    for line in p.stdout:
        line = line.rstrip("\n")
        out_lines.append(line)
        logger.log(line)

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Commande échouée (code {rc}).")
    return "\n".join(out_lines)


def ensure_whisper_available(python_exe: str, logger: Logger):
    cmd = [python_exe, "-c", "import whisper; print('OK: whisper importable')"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run_cmd_stream(logger, cmd, cwd=WHISPER_PROJECT_DIR, env=env)


def download_youtube(url: str, video_dir: str, base_with_date: str, logger: Logger) -> str:
    """
    Télécharge la vidéo dans video_dir avec un nom basé sur base_with_date.
    Retourne le chemin réel du fichier téléchargé.
    """
    safe_makedirs(video_dir)

    # On laisse yt-dlp choisir le conteneur optimal et on force un nom propre.
    # --restrict-filenames évite les caractères exotiques
    # --no-playlist pour ne télécharger qu'une vidéo
    out_tmpl = os.path.join(video_dir, base_with_date + ".%(ext)s")

    cmd = [
        YTDLP_BIN,
        "--no-playlist",
        "--restrict-filenames",
        "-f", "bv*+ba/best",
        "-o", out_tmpl,
        "--print", "after_move:filepath",
        url
    ]

    out = run_cmd_stream(logger, cmd, cwd=video_dir, env=os.environ.copy())

    # La dernière ligne non vide de --print after_move:filepath est normalement le fichier final
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    downloaded = lines[-1] if lines else ""

    # Sécurité si yt-dlp ne renvoie pas comme attendu
    if downloaded and os.path.isfile(downloaded):
        return downloaded

    # fallback : chercher un fichier qui commence par base_with_date
    for fn in os.listdir(video_dir):
        if fn.startswith(base_with_date + "."):
            cand = os.path.join(video_dir, fn)
            if os.path.isfile(cand):
                return cand

    raise RuntimeError("Téléchargement terminé, mais fichier vidéo introuvable dans /video.")


def to_mp4(input_video: str, mp4_dir: str, mp4_path: str, logger: Logger):
    safe_makedirs(mp4_dir)

    # Tentative 1 : remux + audio AAC (rapide si compatible)
    cmd1 = [
        FFMPEG_BIN, "-y",
        "-i", input_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        mp4_path
    ]
    try:
        run_cmd_stream(logger, cmd1, cwd=mp4_dir, env=os.environ.copy())
        return
    except Exception:
        logger.log("⚠️ Remux/copy échoué. Fallback: ré-encodage H.264…")

    # Tentative 2 : ré-encodage vidéo H.264 + audio AAC (plus lent mais robuste)
    cmd2 = [
        FFMPEG_BIN, "-y",
        "-i", input_video,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        mp4_path
    ]
    run_cmd_stream(logger, cmd2, cwd=mp4_dir, env=os.environ.copy())


def mp4_to_wav(mp4_path: str, wav_path: str, logger: Logger):
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", mp4_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        wav_path
    ]
    run_cmd_stream(logger, cmd, cwd=os.path.dirname(mp4_path), env=os.environ.copy())


def explain_whisper_command(python_exe: str, wav_path: str, out_dir: str, model: str, lang: str):
    cmd = [
        python_exe, "-m", "whisper",
        wav_path,
        "--model", model,
        "--device", "cuda",
        "--task", "transcribe",
        "--output_dir", out_dir,
        "--output_format", "txt",
        "--verbose", "True",
    ]
    if lang != "auto":
        cmd += ["--language", lang]
    return cmd


def run_whisper(python_exe: str, wav_path: str, out_dir: str, model: str, lang: str, logger: Logger):
    cmd = explain_whisper_command(python_exe, wav_path, out_dir, model, lang)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run_cmd_stream(logger, cmd, cwd=WHISPER_PROJECT_DIR, env=env)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube → MP4 → Whisper → TXT (GPU)")
        self.geometry("1050x700")

        self.python_exe = sys.executable

        self.url_var = tk.StringVar(value="")
        self.workdir_var = tk.StringVar(value=os.getcwd())

        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.lang_var = tk.StringVar(value=DEFAULT_LANG)

        self.output_txt_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Prêt")

        self._build_ui()
        self.logger = Logger(self.log)

        self.logger.log("=== App démarrée ===")
        self.logger.log(f"Python utilisé: {self.python_exe}")
        self.logger.log(f"WHISPER_PROJECT_DIR: {WHISPER_PROJECT_DIR}")
        self.logger.log(f"Dossier de travail: {self.workdir_var.get()}")

        if not os.path.isdir(WHISPER_PROJECT_DIR):
            self.logger.log("⚠️ Dossier whisper introuvable. Modifie WHISPER_PROJECT_DIR.")
        if not which_ok(FFMPEG_BIN):
            self.logger.log("⚠️ ffmpeg introuvable. Installe: sudo apt install ffmpeg")
        if not which_ok(YTDLP_BIN):
            self.logger.log("⚠️ yt-dlp introuvable. Installe: pip install -U yt-dlp")

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # URL row
        row0 = ttk.Frame(frm)
        row0.pack(fill="x", pady=(0, 10))
        ttk.Label(row0, text="URL YouTube :").pack(side="left")
        ttk.Entry(row0, textvariable=self.url_var).pack(side="left", fill="x", expand=True, padx=8)

        # workdir row
        rowW = ttk.Frame(frm)
        rowW.pack(fill="x", pady=(0, 10))
        ttk.Label(rowW, text="Dossier de sortie :").pack(side="left")
        ttk.Entry(rowW, textvariable=self.workdir_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(rowW, text="Choisir…", command=self.pick_workdir).pack(side="left")

        # options row
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", pady=(0, 10))

        ttk.Label(row2, text="Langue :").pack(side="left")
        ttk.Combobox(row2, textvariable=self.lang_var, width=10, state="readonly",
                     values=["fr", "en", "es", "auto"]).pack(side="left", padx=(6, 14))

        ttk.Label(row2, text="Modèle :").pack(side="left")
        ttk.Combobox(row2, textvariable=self.model_var, width=12, state="readonly",
                     values=["tiny", "base", "small", "medium", "large"]).pack(side="left", padx=(6, 14))

        self.btn_run = ttk.Button(row2, text="Lancer (YouTube → TXT)", command=self.start_pipeline)
        self.btn_run.pack(side="left")

        self.btn_copy = ttk.Button(row2, text="Copier chemin TXT", command=self.copy_txt, state="disabled")
        self.btn_copy.pack(side="left", padx=(10, 0))

        row3 = ttk.Frame(frm)
        row3.pack(fill="x", pady=(0, 10))
        ttk.Label(row3, text="Sortie TXT :").pack(side="left")
        ttk.Entry(row3, textvariable=self.output_txt_var).pack(side="left", fill="x", expand=True, padx=8)

        ttk.Label(frm, text="Terminal / Étapes :").pack(anchor="w")
        self.log = tk.Text(frm, height=26, wrap="word")
        self.log.pack(fill="both", expand=True)

        status = ttk.Frame(frm)
        status.pack(fill="x", pady=(8, 0))
        ttk.Label(status, text="Statut :").pack(side="left")
        ttk.Label(status, textvariable=self.status_var).pack(side="left", padx=8)

    def pick_workdir(self):
        path = filedialog.askdirectory(title="Choisir le dossier de sortie")
        if path:
            self.workdir_var.set(path)
            self.logger.log(f"Dossier de sortie: {path}")

    def copy_txt(self):
        txt = self.output_txt_var.get().strip()
        if txt:
            self.clipboard_clear()
            self.clipboard_append(txt)
            self.status_var.set("Chemin TXT copié")
            self.logger.log("✅ Chemin TXT copié dans le presse-papiers.")

    def start_pipeline(self):
        url = self.url_var.get().strip()
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            self.status_var.set("Erreur: URL invalide")
            self.logger.log("❌ Erreur: colle une URL YouTube valide.")
            return

        if not which_ok(YTDLP_BIN):
            self.status_var.set("Erreur: yt-dlp introuvable")
            self.logger.log("❌ Erreur: yt-dlp introuvable (pip install -U yt-dlp).")
            return

        if not which_ok(FFMPEG_BIN):
            self.status_var.set("Erreur: ffmpeg introuvable")
            self.logger.log("❌ Erreur: ffmpeg introuvable.")
            return

        workdir = self.workdir_var.get().strip() or os.getcwd()
        if not os.path.isdir(workdir):
            self.status_var.set("Erreur: dossier de sortie invalide")
            self.logger.log("❌ Erreur: dossier de sortie invalide.")
            return

        # Logs dans le workdir (pas dans /text, tu ne veux pas de sous-dossier texte)
        log_path = os.path.join(workdir, "whisper_youtube_log.txt")
        self.logger.set_logfile(log_path)

        self.btn_run.configure(state="disabled")
        self.btn_copy.configure(state="disabled")
        self.output_txt_var.set("")
        self.status_var.set("En cours…")
        self.logger.log("=== Début du process ===")

        t = threading.Thread(target=self._pipeline_thread, daemon=True)
        t.start()

    def _pipeline_thread(self):
        try:
            url = self.url_var.get().strip()
            model = self.model_var.get().strip() or DEFAULT_MODEL
            lang = self.lang_var.get().strip() or DEFAULT_LANG
            workdir = self.workdir_var.get().strip() or os.getcwd()

            video_dir = os.path.join(workdir, "video")
            mp4_dir = os.path.join(workdir, "mp4")
            txt_dir = os.path.join(workdir, "txt")

            safe_makedirs(video_dir)
            safe_makedirs(mp4_dir)
            safe_makedirs(txt_dir)

            # DATE
            date_tag = datetime.now().strftime("%Y%m%d")

            # Base de nom : on va faire simple et robuste:
            # On télécharge en nommant: yt_{DATE}__{id}__{title}...
            # MAIS on veut "les 3 fichiers portent le même nom", donc base unique.
            # On laisse yt-dlp remplir title/id mais on force une base fixe avec placeholders.
            # Pour obtenir une base commune, on télécharge d’abord avec template
            # puis on déduit base_with_date du fichier réel.

            self.logger.log(f"Langue={lang} | Modèle={model}")
            self.logger.log(f"Workdir: {workdir}")
            self.logger.log(f"video/: {video_dir}")
            self.logger.log(f"mp4/:   {mp4_dir}")

            self.logger.log("[0/4] Vérification whisper…")
            ensure_whisper_available(self.python_exe, self.logger)

            # Template yt-dlp: title + id + _DATE (DATE injecté côté python)
            # On limite title pour éviter les chemins trop longs
            base_template = f"%(title).80s_%(id)s_{date_tag}"
            self.logger.log("[1/4] Téléchargement YouTube…")
            downloaded_path = download_youtube(url, video_dir, base_template, self.logger)

            downloaded_name = os.path.basename(downloaded_path)
            base = os.path.splitext(downloaded_name)[0]  # <-- contient déjà _DATE
            self.logger.log(f"✅ Vidéo: {downloaded_path}")
            self.logger.log(f"Base commune: {base}")

            mp4_path = os.path.join(mp4_dir, base + ".mp4")
            wav_path = os.path.join(mp4_dir, base + ".wav")  # temp
            txt_path = os.path.join(txt_dir, base + ".txt")

            self.logger.log("[2/4] Conversion/remux en MP4…")
            to_mp4(downloaded_path, mp4_dir, mp4_path, self.logger)
            if not os.path.isfile(mp4_path):
                raise RuntimeError("Conversion MP4 terminée, mais fichier MP4 introuvable.")

            self.logger.log("[3/4] Extraction WAV 16k mono…")
            mp4_to_wav(mp4_path, wav_path, self.logger)
            if not os.path.isfile(wav_path):
                raise RuntimeError("Extraction WAV terminée, mais WAV introuvable.")

            self.logger.log("[4/4] Transcription Whisper…")
            # Whisper va écrire base.txt dans workdir (output_dir=workdir)
            run_whisper(self.python_exe, wav_path, txt_dir, model, lang, self.logger)

            # Whisper nomme selon le fichier audio: base.wav => base.txt
            if not os.path.isfile(txt_path):
                # fallback: chercher un .txt récent
                cands = [f for f in os.listdir(workdir) if f.lower().endswith(".txt") and f.startswith(base)]
                if cands:
                    txt_path = os.path.join(workdir, sorted(cands)[0])

            # Nettoyage wav (optionnel)
            try:
                os.remove(wav_path)
                self.logger.log("🧹 WAV temporaire supprimé.")
            except Exception:
                pass

            if os.path.isfile(txt_path):
                self.output_txt_var.set(txt_path)
                self.btn_copy.configure(state="normal")
                self.status_var.set("Terminé ✅")
                self.logger.log("=== Terminé ✅ ===")
                self.logger.log(f"VIDEO: {downloaded_path}")
                self.logger.log(f"MP4:   {mp4_path}")
                self.logger.log(f"TXT:   {txt_path}")
            else:
                raise RuntimeError("Transcription terminée, mais TXT introuvable.")

        except Exception as e:
            self.status_var.set("Erreur ❌ (voir log)")
            self.logger.log("=== Erreur ❌ ===")
            self.logger.log(str(e))
        finally:
            self.btn_run.configure(state="normal")


if __name__ == "__main__":
    App().mainloop()
