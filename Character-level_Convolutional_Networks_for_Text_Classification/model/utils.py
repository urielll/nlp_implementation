import os, sys
import re
import string

# reference
# https://github.com/neotune/python-korean-handler/blob/master/korean_handler.py

class JamoTokenizer:
    def __init__(self):
        # 유니코드 한글 시작 : 44032, 끝 : 55199
        self.__base_code = 44032
        self.__chosung = 588
        self.__jungsung = 28
        # 초성 리스트. 00 ~ 18
        self.__chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                               'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                               'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        # 중성 리스트. 00 ~ 20
        self.__jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                                'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                                'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        # 종성 리스트. 00 ~ 27 + 1(1개 없음)
        self.__jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ',
                                'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
                                'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                                'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        self.__token_dic = sorted(list(set(self.__chosung_list + self.__jungsung_list + self.__jongsung_list)))
        self.__token_dic.extend(['<eng>', '<num>', '<unk>'])
        self.__token_dic = {token : idx for idx, token in enumerate(self.__token_dic)}

    def tokenize(self):
        pass

    def token2idx(self):
        pass


jamo = JamoTokenizer()
jamo._JamoTokenizer__token_dic

def tokenize_jamo(string:str) -> list:
    # 유니코드 한글 시작 : 44032, 끝 : 55199
    base_code, chosung, jungsung = 44032, 588, 28

    # 초성 리스트. 00 ~ 18
    chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                    'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                    'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                     'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                     'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ',
                     'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
                     'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                     'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    split_string = list(string)

    result = []
    for char in split_string:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', char) is not None:
            char_code = ord(char) - base_code
            alphabet1 = int(char_code / chosung)
            result.append(chosung_list[alphabet1])
            alphabet2 = int((char_code - (chosung * alphabet1)) / jungsung)
            result.append(jungsung_list[alphabet2])
            alphabet3 = int((char_code - (chosung * alphabet1) - (jungsung * alphabet2)))
            if alphabet3 != 0:
                result.append(jongsung_list[alphabet3])
        else:
            result.append(char)

    return result


for tok


chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                    'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                    'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                     'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                     'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ',
                 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
                 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                  'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

unified_list = []

unified_list.extend(chosung_list)
unified_list.extend(jungsung_list)
unified_list.extend(jongsung_list)
unified_list = set(unified_list)
unified_list = list(unified_list)
unified_list= sorted(unified_list)
unified_list

chosung_list + jungsung_list + jongsung_list

tokenize_jamo('김보서')

str.isalnum('1')
str.isnumeric('123')
str.isalpha('ㅑ')

# tokenize 구현시
import re
re.match('[A-z]', 'a') is not None
re.match('[0-9]', 'a') is not None