# -*- coding: utf-8 -*-
"""LLM-based Japanese lyric generation with accurate mora counting via SudachiPy."""

import re
from dataclasses import dataclass

import jaconv


@dataclass
class LyricPhrase:
    hiragana: str   # mora-by-mora reading (used for note assignment)
    kanji: str      # natural kanji+kana form for display
    english: str    # English translation

# ── Mora counting ─────────────────────────────────────────────────────────────

_sudachi_tokenizer = None
_sudachi_mode = None
_sudachi_available = None

# Small kana that attach to the preceding mora (not their own beat).
# Note: っ/ッ (sokuon) is intentionally excluded — it IS its own mora in Japanese prosody.
_SMALL_KANA = set('ャュョァィゥェォゃゅょぁぃぅぇぉ')


def _init_sudachi():
    global _sudachi_tokenizer, _sudachi_mode, _sudachi_available
    if _sudachi_available is not None:
        return _sudachi_available
    try:
        import sudachipy
        from sudachipy import tokenizer as tkn
        _sudachi_tokenizer = sudachipy.Dictionary().create()
        _sudachi_mode = tkn.Tokenizer.SplitMode.C
        _sudachi_available = True
    except Exception:
        _sudachi_available = False
    return _sudachi_available


def get_reading(text: str) -> str:
    """Return the hiragana reading of a mixed kanji/kana string using SudachiPy."""
    if _init_sudachi():
        tokens = _sudachi_tokenizer.tokenize(text, _sudachi_mode)
        reading = ''.join(token.reading_form() for token in tokens)
        return jaconv.kata2hira(reading)
    # Fallback: convert any katakana to hiragana, leave kana as-is
    return jaconv.kata2hira(text)


def count_morae(text: str) -> int:
    """Count Japanese morae in text. Uses SudachiPy when available, falls back to character count."""
    if _init_sudachi():
        tokens = _sudachi_tokenizer.tokenize(text, _sudachi_mode)
        count = 0
        for token in tokens:
            reading = token.reading_form()  # always katakana
            count += sum(1 for c in reading if c not in _SMALL_KANA)
        return count
    # Fallback: convert to hiragana and count non-small-kana characters
    hira = jaconv.kata2hira(text)
    return sum(1 for c in hira if c not in _SMALL_KANA)


# ── Lyric generator ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    'あなたは日本語の歌詞を書く専門家です。'
    '指定されたモーラ数にぴったり合う短い歌詞フレーズを書いてください。'
    '余計な説明やラベルは不要です。'
    '日本語フレーズのみ書いてから、改行して英語訳のみ書いてください。'
)

_IS_JAPANESE = re.compile(r'[぀-鿿ｦ-ﾟ]')
# Matches common label prefixes the model might prepend (e.g. "日本語：", "Japanese: ")
_LABEL_PREFIX = re.compile(r'^[\w\s]*[：:]\s*')


class QwenLyricGenerator:
    """Generates Japanese lyric phrases via an LLM through Ollama.

    Falls back to the Markov chain generator if Ollama is unavailable or the
    requested model is not installed.
    """

    def __init__(self, model='qwen2.5:7b', theme='青春', max_retries=3):
        self.model = model
        self.theme = theme
        self.max_retries = max_retries
        self._ollama_ok = self._check_ollama()
        if self._ollama_ok:
            print(f'LyricGenerator: using Ollama model "{model}"')
        else:
            print('LyricGenerator: Ollama unavailable, falling back to Markov chain')

    def _check_ollama(self):
        try:
            import ollama
            available = [m.model for m in ollama.list().models]
            # Accept either exact match or prefix match (e.g. "llama3.1:8b" matches "llama3.1:8b")
            if not any(self.model == m or m.startswith(self.model.split(':')[0]) for m in available):
                print(f'LyricGenerator: model "{self.model}" not found. Available: {available}')
                return False
            return True
        except Exception:
            return False

    def generate_phrase(self, mora_count: int) -> LyricPhrase:
        """Return a LyricPhrase(hiragana, kanji, english) with ~mora_count morae."""
        if mora_count <= 0:
            return LyricPhrase('', '', '')
        if not self._ollama_ok:
            return self._markov_fallback(mora_count)
        return self._generate_with_llm(mora_count)

    def _generate_with_llm(self, mora_count: int) -> LyricPhrase:
        import ollama
        best: LyricPhrase | None = None
        best_diff = 999
        for _ in range(self.max_retries):
            try:
                response = ollama.chat(model=self.model, messages=[
                    {'role': 'system', 'content': _SYSTEM_PROMPT},
                    {'role': 'user', 'content':
                        f'テーマ：{self.theme}\nモーラ数：{mora_count}\n'}
                ])
                raw = response.message.content.strip()
                lines = [l.strip() for l in raw.splitlines() if l.strip()]

                # First line with Japanese characters = phrase; first all-ASCII line after = English
                kanji = ''
                english = ''
                for line in lines:
                    # Strip any label prefix the model might add (e.g. "Japanese: ", "日本語：")
                    clean = _LABEL_PREFIX.sub('', line).strip()
                    if not kanji and _IS_JAPANESE.search(clean):
                        kanji = clean
                    elif kanji and clean and not _IS_JAPANESE.search(clean):
                        english = clean
                        break
                if not kanji:
                    continue

                hiragana = get_reading(kanji)
                diff = abs(count_morae(kanji) - mora_count)
                if diff < best_diff:
                    best_diff = diff
                    best = LyricPhrase(hiragana=hiragana, kanji=kanji, english=english)
                if diff == 0:
                    break
            except Exception:
                break
        if best is not None:
            return best
        return self._markov_fallback(mora_count)

    def _markov_fallback(self, mora_count: int) -> LyricPhrase:
        """Use the existing Markov chain generator as a fallback."""
        try:
            from support.parsingHelper import generateLyric
            import pykakasi
            kks = pykakasi.kakasi()
            res, _ = generateLyric(mora_count, min(3, mora_count), None, kks)
            hiragana = jaconv.kata2hira(res)
            return LyricPhrase(hiragana=hiragana, kanji=res, english='')
        except Exception:
            _fillers = 'あいうえおかきくけこさしすせそなにぬねのはひふへほ'
            hira = ''.join(_fillers[i % len(_fillers)] for i in range(mora_count))
            return LyricPhrase(hiragana=hira, kanji=hira, english='')
