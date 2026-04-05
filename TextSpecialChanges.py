import random
import string
import unicodedata
import pandas as pd

class ObscuredText:
    def __init__(self, seed: int = None, intensity: float = 0.5):
        if seed is not None:
            random.seed(seed)
        self.intensity = intensity

    def deep_word_bug(self, text):
        special_symbols = '~!@#$%^&*'
        result_text = []
        for symbol in text:
            result_text.append(symbol)
            if random.random() < self.intensity:  # < для intensity
                result_text.append(random.choice(special_symbols))
        return ''.join(result_text)

    def diacritics(self, text):
        diacritics_map = {
            'a': 'áàäâãåāăą',
            'e': 'éèëêēĕėę',
            'i': 'íìïîīĭį',
            'o': 'óòöôōŏő',
            'u': 'úùüûūŭů',
            'c': 'çćĉċč',
            'n': 'ñńņň',
            's': 'śŝşš',
            'z': 'źżž',
            'y': 'ýÿ',
        }
        result_text = []
        for symbol in text:

            lower_symbol = symbol.lower()
            if lower_symbol in diacritics_map and random.random() < self.intensity:
                replacement = random.choice(diacritics_map[lower_symbol])

                if symbol.isupper():
                    replacement = replacement.upper()
                result_text.append(replacement)
            else:
                result_text.append(symbol)
        return ''.join(result_text)

    def underline_accent_marks(self, text):

        accents = ['\u0332', '\u0333', '\u0331', '\u0330']

        result_text = []
        for symbol in text:
            result_text.append(symbol)
            if symbol.isalpha() and random.random() < self.intensity:
                result_text.append(random.choice(accents))
        return ''.join(result_text)

    def upside_down_text(self, text):

        upside_down_map = {
            'a': 'ɐ', 'b': 'q', 'c': 'ɔ', 'd': 'p', 'e': 'ǝ',
            'f': 'ɟ', 'g': 'ɓ', 'h': 'ɥ', 'i': 'ı', 'j': 'ɾ',
            'k': 'ʞ', 'l': 'l', 'm': 'ɯ', 'n': 'u', 'o': 'o',
            'p': 'd', 'q': 'b', 'r': 'ɹ', 's': 's', 't': 'ʇ',
            'u': 'n', 'v': 'ʌ', 'w': 'ʍ', 'x': 'x', 'y': 'ʎ',
            'z': 'z', 'A': '∀', 'B': '𐐒', 'C': 'Ↄ', 'D': '◖',
            'E': 'Ǝ', 'F': 'Ⅎ', 'G': '⅁', 'H': 'H', 'I': 'I',
            'J': 'ſ', 'K': 'ʞ', 'L': '⅂', 'M': 'W', 'N': 'N',
            'O': 'O', 'P': 'Ԁ', 'Q': 'Ò', 'R': 'ᴚ', 'S': 'S',
            'T': '⊥', 'U': '∩', 'V': 'Λ', 'W': 'M', 'X': 'X',
            'Y': '⅄', 'Z': 'Z', '?': '¿', '!': '¡', '.': '˙',
            ',': "'", "'": ',', '"': ',,', '`': ',', '(': ')',
            ')': '(', '[': ']', ']': '[', '{': '}', '}': '{',
            '<': '>', '>': '<', '&': '⅋', '_': '‾',
        }

        reversed_text = text[::-1]
        result_text = []
        for symbol in reversed_text:
            result_text.append(upside_down_map.get(symbol, symbol))
        return ''.join(result_text)

    def bidirectional_text(self, text):
        rlo = '\u202E'
        lro = '\u202D'
        pdf = '\u202C'

        result_text = []
        for i, symbol in enumerate(text):
            if random.random() < self.intensity:
                result_text.append(random.choice([rlo, lro]))
            result_text.append(symbol)
            if random.random() < self.intensity * 0.5:  # <, а не >
                result_text.append(pdf)
        return ''.join(result_text)

    def full_width_text(self, text):
        full_width_map = {}

        for i in range(0x21, 0x7F):
            full_width_map[chr(i)] = chr(i + 0xFEE0)

        full_width_map[' '] = '\u3000'

        result_text = []
        for symbol in text:
            result_text.append(full_width_map.get(symbol, symbol))
        return ''.join(result_text)

    def emoji_smuggling(self, text):
        emojis = ['😀', '😂', '😅', '😊', '😍', '🥺', '😎', '🤔', '🤯',
                  '🔥', '✨', '⭐', '💯', '💀', '👽', '🎉', '🚀', '❤️']

        result_text = []
        words = text.split()
        for i, word in enumerate(words):
            result_text.append(word)
            if i < len(words) - 1 and random.random() < self.intensity:
                result_text.append(random.choice(emojis))
        return ' '.join(result_text)

    def spaces_text(self, text):
        result_text = []
        for char in text:
            result_text.append(char)
            if char == ' ' and random.random() < self.intensity:
                result_text.append(' ' * random.randint(1, 3))
        return ''.join(result_text)

    def homoglyphs(self, text):
        homoglyphs_map = {
            'a': ['а', '@', 'α', '⍺'],
            'b': ['Ь', 'в', 'β', '6'],
            'c': ['с', 'ϲ', '©'],
            'e': ['е', 'ё', '€', 'ǝ'],
            'h': ['н', 'ħ', 'ℎ'],
            'i': ['і', '1', '!', '|'],
            'k': ['к', 'κ', 'ќ'],
            'm': ['м', 'ⅿ', '♏'],
            'o': ['о', '0', '○', '◯'],
            'p': ['р', 'ρ', 'þ'],
            't': ['т', 'τ', '†'],
            'x': ['х', '×', 'ⅹ'],
            'y': ['у', 'γ', '¥'],
            'l': ['l', '1', '|', 'ℓ'],
        }

        result_text = []
        for symbol in text:
            lower_symbol = symbol.lower()
            if lower_symbol in homoglyphs_map and random.random() < self.intensity:
                replacement = random.choice(homoglyphs_map[lower_symbol])

                if symbol.isupper() and replacement.isalpha():
                    replacement = replacement.upper()
                result_text.append(replacement)
            else:
                result_text.append(symbol)
        return ''.join(result_text)

    def unicode_tags_smuggling(self, text):
        tag_start = '\uE0000'
        result_text = [tag_start]

        for symbol in text:
            if symbol.isalpha():
                if symbol.islower():
                    tag_char = chr(ord(symbol) - ord('a') + 0xE0041)
                else:
                    tag_char = chr(ord(symbol) - ord('A') + 0xE0041)
                result_text.append(tag_char)
            elif symbol.isdigit():
                tag_char = chr(ord(symbol) - ord('0') + 0xE0030)
                result_text.append(tag_char)
            else:
                result_text.append(symbol)
        return ''.join(result_text)

    def numbers_text(self, text):
        leet_map = {
            'a': '4', 'e': '3', 'i': '1', 'o': '0',
            's': '5', 't': '7', 'z': '2', 'g': '9',
            'b': '8', 'l': '1',
        }

        result_text = []
        for symbol in text:
            lower_symbol = symbol.lower()
            if lower_symbol in leet_map and random.random() < self.intensity:
                result_text.append(leet_map[lower_symbol])
            else:
                result_text.append(symbol)
        return ''.join(result_text)


def apply_obscured_transformation(prompts_column:pd.Series,
                                  transformation_function_list:list= None,
                                  number_transformed_prompts_by_category:int=10,
                                  seed:int=17,
                                  intensity:float=0.5):
    if transformation_function_list is None:
        transformation_function_list = ['deep_word_bug',
                                        'diacritics',
                                        'underline_accent_marks',
                                        'upside_down_text',
                                        'bidirectional_text',
                                        'full_width_text',
                                        'emoji_smuggling',
                                        'spaces_text',
                                        'homoglyphs',
                                        'unicode_tags_smuggling',
                                         'numbers_text']

    number_total_transformed_prompts=number_transformed_prompts_by_category*len(transformation_function_list)
    obfuscator = ObscuredText(seed=seed, intensity=intensity)
    result_column=prompts_column.copy()
    index_to_transform=random.sample(range(len(prompts_column)), number_total_transformed_prompts)
    idx_pos=0
    for transformation_function in transformation_function_list:
        for i in range(number_transformed_prompts_by_category):
            idx=index_to_transform[idx_pos]
            original_prompt=prompts_column.iloc[idx]
            transform_method=getattr(obfuscator, transformation_function)
            transformed_prompt=transform_method(original_prompt)
            result_column.iloc[idx] = transformed_prompt
            idx_pos=idx_pos+1
    return result_column




if __name__ == "__main__":

    obfuscator = ObscuredText(seed=42, intensity=0.7)

    test_text = "Hello World! Test 123"

    print(f"Оригинал: {test_text}")
    print(f"Deep Word Bug: {obfuscator.deep_word_bug(test_text)}")
    print(f"Diacritics: {obfuscator.diacritics(test_text)}")
    print(f"Underline: {obfuscator.underline_accent_marks(test_text)}")
    print(f"Upside Down: {obfuscator.upside_down_text(test_text)}")
    print(f"Bidirectional: {obfuscator.bidirectional_text(test_text)}")
    print(f"Full Width: {obfuscator.full_width_text(test_text)}")
    print(f"Emoji: {obfuscator.emoji_smuggling(test_text)}")
    print(f"Spaces: {obfuscator.spaces_text(test_text)}")
    print(f"Homoglyphs: {obfuscator.homoglyphs(test_text)}")
    print(f"Unicode Tags: {obfuscator.unicode_tags_smuggling(test_text)}")
    print(f"Numbers: {obfuscator.numbers_text(test_text)}")