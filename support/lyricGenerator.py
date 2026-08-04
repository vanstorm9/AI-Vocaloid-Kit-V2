# -*- coding: utf-8 -*-
"""LLM-based Japanese lyric generation with accurate mora counting via SudachiPy."""

import jaconv

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
    '指定されたモーラ数にぴったり合う短い歌詞フレーズを1つだけ書いてください。'
    'ひらがなのみで返答してください。句読点・記号・改行は不要です。'
)


class QwenLyricGenerator:
    """Generates Japanese lyric phrases via Qwen2.5 through Ollama.

    Falls back to the Markov chain generator if Ollama is unavailable.
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
            ollama.list()
            return True
        except Exception:
            return False

    def generate_phrase(self, mora_count: int) -> str:
        """Return a Japanese phrase with approximately mora_count morae."""
        if mora_count <= 0:
            return ''
        if not self._ollama_ok:
            return self._markov_fallback(mora_count)
        return self._generate_with_llm(mora_count)

    def _generate_with_llm(self, mora_count: int) -> str:
        import ollama
        best = None
        best_diff = 999
        for _ in range(self.max_retries):
            try:
                response = ollama.chat(model=self.model, messages=[
                    {'role': 'system', 'content': _SYSTEM_PROMPT},
                    {'role': 'user', 'content':
                        f'テーマ：{self.theme}\nモーラ数：{mora_count}\n歌詞：'}
                ])
                phrase = response['message']['content'].strip().split('\n')[0]
                # Strip punctuation/symbols that slip through
                phrase = ''.join(c for c in phrase if '぀' <= c <= 'ヿ' or c == 'ー')
                if not phrase:
                    continue
                diff = abs(count_morae(phrase) - mora_count)
                if diff < best_diff:
                    best_diff = diff
                    best = phrase
                if diff == 0:
                    break
            except Exception:
                break
        if best is not None:
            return best
        return self._markov_fallback(mora_count)

    def _markov_fallback(self, mora_count: int) -> str:
        """Use the existing Markov chain generator as a fallback."""
        try:
            from support.parsingHelper import generateLyric
            import pykakasi
            kks = pykakasi.kakasi()
            res, _ = generateLyric(mora_count, min(3, mora_count), None, kks)
            return res
        except Exception:
            # Last resort: hiragana filler
            _fillers = 'あいうえおかきくけこさしすせそなにぬねのはひふへほ'
            return ''.join(_fillers[i % len(_fillers)] for i in range(mora_count))
