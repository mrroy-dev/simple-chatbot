import re
import unicodedata
import html
from typing import List


class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r"https?://[^\s]+")
        self.html_pattern = re.compile(r"<[^>]+>")
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        self.whitespace_pattern = re.compile(r"\s+")
        self.multiple_punct = re.compile(r"([!?.]){2,}")

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def remove_urls(self, text: str) -> str:
        return self.url_pattern.sub("", text).strip()

    def remove_html(self, text: str) -> str:
        return self.html_pattern.sub("", text).strip()

    def normalize_emojis(self, text: str) -> str:
        return self.emoji_pattern.sub(" ", text).strip()

    def normalize_whitespace(self, text: str) -> str:
        return self.whitespace_pattern.sub(" ", text).strip()

    def normalize_punctuation(self, text: str) -> str:
        return self.multiple_punct.sub(r"\1", text)

    def clean(self, text: str, remove_urls: bool = True, remove_html: bool = True,
              normalize_unicode: bool = True, normalize_emojis: bool = True,
              normalize_whitespace: bool = True, normalize_punct: bool = True) -> str:
        text = text.strip()
        if not text:
            return ""
        if normalize_unicode:
            text = self.normalize_unicode(text)
        if remove_urls:
            text = self.remove_urls(text)
        if remove_html:
            text = html.unescape(text)
            text = self.remove_html(text)
        if normalize_emojis:
            text = self.normalize_emojis(text)
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
        if normalize_punct:
            text = self.normalize_punctuation(text)
        return text.strip()

    def clean_batch(self, texts: List[str], **kwargs) -> List[str]:
        return [self.clean(t, **kwargs) for t in texts]
