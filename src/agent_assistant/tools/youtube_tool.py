import os
import re
import tempfile
from typing import Optional

from pydantic import BaseModel, HttpUrl, Field, PrivateAttr
from crewai.tools import BaseTool
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

from agent_assistant.config import KNOWLEDGE_DIR
from dotenv import load_dotenv

class YouTubeToolInput(BaseModel):
    """Input schema for YouTubeTool."""
    url: HttpUrl = Field(..., description="YouTube video URL to process.")
    action: str = Field(
        "transcript",
        description="Action to perform: 'transcript', 'summary', or 'ask'",
        pattern="^(transcript|summary|ask)$"
    )
    question: Optional[str] = Field(None, description="Question to ask about the video (required if action is 'ask')")

def _extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError(f"Invalid YouTube URL: {url}")
    return m.group(1)

class YouTubeTool(BaseTool):
    model_config = {"extra": "allow"}
    _client: any = PrivateAttr()
    name: str = "youtube_tool"
    description: str = (
        "Fetch transcript or summary of a YouTube video, or answer questions about it. "
        "Usage: action='transcript'|'summary'|'ask'; for 'ask', provide a question."
    )
    args_schema = YouTubeToolInput

    def __init__(self, **data):
        super().__init__(**data)
        print("YouTubeTool __init__ called")  # Debug: confirm __init__ is called
        load_dotenv(override=False)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for YouTubeTool.")
        print("OpenAI class:", OpenAI)
        print("OpenAI module:", getattr(OpenAI, '__module__', 'N/A'))
        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Exception during OpenAI client initialization: {e}")
            raise

    def _run(self, url: str, action: str = "transcript", question: Optional[str] = None) -> str:
        # Debug: check if self._client exists
        if not hasattr(self, '_client'):
            raise RuntimeError("YouTubeTool is missing the '_client' attribute. This usually means __init__ was not called. Please check how YouTubeTool is instantiated.")
        # Prepare storage directory
        yt_dir = os.path.join(KNOWLEDGE_DIR, "youtube")
        os.makedirs(yt_dir, exist_ok=True)
        # Extract video ID and set file paths
        video_id = _extract_video_id(url)
        transcript_path = os.path.join(yt_dir, f"{video_id}.txt")
        summary_path = os.path.join(yt_dir, f"{video_id}_summary.txt")

        # Load or fetch transcript
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        else:
            transcript = self._fetch_transcript(video_id, url)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        # Perform requested action
        # Action: transcript: download and index for Q&A
        if action == "transcript":
            # build vector index for Q&A
            try:
                self._ensure_index(video_id, transcript)
                return "Tekstitys tallennettu."
            except Exception as e:
                return f"Error indexing transcript: {e}"

        if action == "summary":
            # Reuse existing summary if available
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    return f.read()
            summary = self._summarize(transcript)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            return summary

        if action == "ask":
            if not question:
                return "Error: question is required when action is 'ask'."
            # ensure index exists
            try:
                self._ensure_index(video_id, transcript)
            except Exception:
                pass
            # retrieve and answer
            try:
                return self._ask(video_id, question)
            except Exception as e:
                # fallback to basic full-text QA
                return self._basic_ask(transcript, question)

        return f"Unknown action '{action}'. Valid actions: transcript, summary, ask."

    def _fetch_transcript(self, video_id: str, url: str) -> str:
        """Try to get existing captions, else download audio and transcribe with Whisper."""
        # First attempt: fetch auto-generated or uploaded captions
        try:
            entries = YouTubeTranscriptApi.list_transcripts(video_id)
            # Prefer manually created, else fallback
            transcript_list = None
            try:
                transcript_list = entries.find_manually_created_transcript(["en"])
            except Exception:
                transcript_list = entries.find_transcript(entries._languages)
            data = transcript_list.fetch()
            # Combine text segments
            combined = "\n".join(seg["text"] for seg in data)
            return combined
        except Exception:
            # Fallback: download audio and use Whisper
            with tempfile.TemporaryDirectory() as tmpdir:
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": os.path.join(tmpdir, f"%(id)s.%(ext)s"),
                    "quiet": True,
                    "no_warnings": True,
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }],
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    audio_file = os.path.join(tmpdir, f"{info['id']}.mp3")
                # Transcribe with Whisper
                with open(audio_file, "rb") as af:
                    resp = self._client.audio.transcriptions.create(
                        model="whisper-1",
                        file=af,
                        language="en"
                    )
                return resp.text

    def _summarize(self, transcript: str) -> str:
        """Summarize transcript via GPT."""
        prompt = (
            "Please provide a concise summary of the following YouTube video transcript:\n\n"
            f"{transcript}"
        )
        resp = self._client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return resp.choices[0].message.content

    def _basic_ask(self, transcript: str, question: str) -> str:
        """Fallback: answer question by providing full transcript context."""
        prompt = (
            "You have the transcript of a YouTube video. "
            "Answer the following question based on the transcript:\n\n"
            f"{question}\n\nTranscript:\n{transcript}"
        )
        resp = self._client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return resp.choices[0].message.content

    def _ensure_index(self, video_id: str, transcript: str) -> str:
        """Ensure embeddings index exists for given transcript, else build it."""
        import json
        import numpy as np
        idx_path = os.path.join(KNOWLEDGE_DIR, "youtube", f"{video_id}_embeddings.json")
        if os.path.exists(idx_path):
            return idx_path
        # chunk transcript
        chunks = self._chunk_text(transcript, max_chars=1000)
        embeddings = []
        for chunk in chunks:
            # embed each chunk
            resp = self._client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            vec = resp.data[0].embedding
            embeddings.append({"chunk": chunk, "embedding": vec})
        # save index
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        return idx_path

    def _load_index(self, video_id: str):
        """Load embeddings index for video_id."""
        import json
        idx_path = os.path.join(KNOWLEDGE_DIR, "youtube", f"{video_id}_embeddings.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _chunk_text(self, text: str, max_chars: int = 1000):
        """Split text into chunks of at most max_chars, respecting line boundaries."""
        paragraphs = text.split("\n")
        chunks = []
        current = []
        count = 0
        for para in paragraphs:
            if not para:
                continue
            length = len(para)
            if count + length + 1 > max_chars and current:
                chunks.append("\n".join(current))
                current = [para]
                count = length
            else:
                current.append(para)
                count += length + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _ask(self, video_id: str, question: str) -> str:
        """Answer a question using vector retrieval over indexed transcript."""
        import numpy as np
        # load index
        entries = self._load_index(video_id)
        # embed question
        qresp = self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        qvec = qresp.data[0].embedding
        # compute similarities
        sims = []
        for entry in entries:
            vec = entry["embedding"]
            # cosine similarity
            sim = np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec))
            sims.append((sim, entry["chunk"]))
        # take top 3
        sims.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in sims[:3]]
        context = "\n---\n".join(top_chunks)
        prompt = (
            "Use the following relevant excerpts from a YouTube transcript to answer the question:\n\n"
            f"{context}\n\nQuestion: {question}"
        )
        resp = self._client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return resp.choices[0].message.content