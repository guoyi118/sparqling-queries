import pandas as pd 
import pickle
import deepl
import os 
import requests
import random
import json
from hashlib import md5
from ast import literal_eval
import time
from nltk.tokenize import word_tokenize

# modify rule 1 : 打乱顺序
# modify rule 2 ： 更改args
# modify rule 3: 随意删除
# modify rule 4 ： 更改ref
# modify rule 5: 更改step type
# modify rule 6， 翻译raw question


class BaiduTranslate:
    def __init__(self, appid, appkey, from_lang='en', to_lang='zh'):
    # Set your own appid/appkey.
        self.appid = appid
        self.appkey = appkey
        # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
        self.from_lang = from_lang
        self.to_lang =  to_lang
        self.endpoint = 'http://api.fanyi.baidu.com'
        self.path = '/api/trans/vip/translate'
        self.url = self.endpoint + self.path

    def make_md5(self, s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    def translate(self, query):
        # query = 'Hello World!'

        # Generate salt and sign

        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.appid + query + str(salt) + self.appkey)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': query, 'from': self.from_lang, 'to': self.to_lang, 'salt': salt, 'sign': sign}

    # Send request
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()

        result_json =  json.dumps(result, indent=4, ensure_ascii=False)
        result_json = literal_eval(result_json)
        return result_json

def translation(input):
    translator_en_zh = BaiduTranslate('20210421000793639','3NddTWkFMRj6O2EHxN4b', from_lang='en', to_lang='zh')
    translate_text = translator_en_zh.translate(input)
    zh_result = translate_text['trans_result'][0]['dst']
    time.sleep(1)

    translator_zh_de = BaiduTranslate('20210421000793639','3NddTWkFMRj6O2EHxN4b', from_lang='zh', to_lang='de')
    translate_text = translator_zh_de.translate(zh_result)
    de_result = translate_text['trans_result'][0]['dst']
    time.sleep(1)

    translator_de_en = BaiduTranslate('20210421000793639','3NddTWkFMRj6O2EHxN4b', from_lang='de', to_lang='en')
    translate_text = translator_de_en.translate(de_result)
    en_result = translate_text['trans_result'][0]['dst']

    return en_result

# pickle_in = open("encode_data.pickle","rb")
# encode_data = pickle.load(pickle_in)

# pickle_in = open("correct_decode_data.pickle","rb")
# correct_decode_data = pickle.load(pickle_in)

pickle_in = open("wrong_decode_data.pickle","rb")
wrong_decode_data = pickle.load(pickle_in)


# pickle_in = open("filtered_orig_data.pickle","rb")
# filtered_orig_data = pickle.load(pickle_in)

# new_encode_data = []
# new_correct_decode_data = []
# new_wrong_decode_data = []
# new_filtered_orig_data = []

# # rule 1:
# for encode, c_decode, w_decode, fil_origin in zip(encode_data, correct_decode_data, wrong_decode_data, filtered_orig_data):
#     new_encode_data.append(encode)
#     new_correct_decode_data.append(c_decode)
#     new_wrong_decode_data.append(w_decode)
#     new_filtered_orig_data.append(fil_origin)


#     new_encode_data.append(encode)
#     new_correct_decode_data.append(c_decode)
#     new_wrong_decode_data.append(w_decode)
#     new_filtered_orig_data.append(fil_origin)

    
print(wrong_decode_data[0])
# print(len(wrong_decode_data[0].tree))
# print(len(wrong_decode_data[0].orig_code))







#~~~~~~~~~~~~~~translation~~~~~~~~~~~~~~~~~~~~
# f = open('data/spider/dev.json')

# dev_spider_json = json.load(f)

# new_dev_spider_json = []

# for item in dev_spider_json:
#     time.sleep(1)
#     try:
#         rephrase_question = translation(item['question'])
#     except:
#         rephrase_question = item['question']

#     if item['question'] == rephrase_question:
#         continue

#     new_item = {
#         'db_id' : item['db_id'],
#         'query' : item['query'],
#         'query_toks' : item['query_toks'],
#         'query_toks_no_value' : item['query_toks_no_value'],
#         'question' : rephrase_question,
#         'question_toks' : word_tokenize(rephrase_question),
#         'sql' : item['sql']
#     }

#     new_dev_spider_json.append(new_item)

# with open('data/spider/new_dev.json', 'w') as fp:
#     json.dump(new_dev_spider_json,fp)
    




