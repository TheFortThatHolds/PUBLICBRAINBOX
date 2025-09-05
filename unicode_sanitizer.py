#!/usr/bin/env python3

import re
import unicodedata

def sanitize_for_windows_terminal(text):
    """Remove problematic Unicode characters for Windows terminal output"""
    if not text:
        return text
    
    # Remove emoji and special Unicode symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    
    # Remove emojis
    text = emoji_pattern.sub('', text)
    
    # Remove other problematic Unicode characters
    # Keep only ASCII + common extended characters
    sanitized = ""
    for char in text:
        if ord(char) < 127:  # Basic ASCII
            sanitized += char
        elif ord(char) in [8211, 8212, 8216, 8217, 8220, 8221, 8594, 8592, 8593, 8595, 8596]:  # Common punctuation + arrows
            # Replace with ASCII equivalents
            replacements = {
                8211: '-',   # en dash
                8212: '--',  # em dash
                8216: "'",   # left single quote
                8217: "'",   # right single quote
                8220: '"',   # left double quote
                8221: '"',   # right double quote
                8594: '->',  # right arrow →
                8592: '<-',  # left arrow ←
                8593: '^',   # up arrow ↑
                8595: 'v',   # down arrow ↓
                8596: '<->', # left right arrow ↔
            }
            sanitized += replacements.get(ord(char), char)
        else:
            # Skip other Unicode characters
            continue
    
    return sanitized

def test_sanitizer():
    """Test the sanitizer with common problematic strings"""
    test_strings = [
        "🎵 This has emoji 🎭",
        "Smart quotes: \"hello\" and 'world'",
        "Em dash — and en dash –",
        "Normal text with no issues",
        "**Voice**: Some content here",
        "*italic text* and **bold text**"
    ]
    
    print("=== UNICODE SANITIZER TEST ===")
    for test in test_strings:
        sanitized = sanitize_for_windows_terminal(test)
        # Sanitize the original for printing too
        safe_original = sanitize_for_windows_terminal(test)
        print(f"Input: {repr(safe_original)}")
        print(f"Output: {repr(sanitized)}")
        print(f"Same: {safe_original == sanitized}")
        print()

if __name__ == "__main__":
    test_sanitizer()