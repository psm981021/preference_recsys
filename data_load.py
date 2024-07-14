import gzip
import pickle as pkl
from collections import defaultdict
import numpy as np
import pandas as pd
import json
import random
from bs4 import BeautifulSoup


def parse(path):
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield json.loads(line)

# html 코드까지 크롤링된 데이터 전처리
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text().strip()

def get_items_meta(meta_path, categories_used='all'):
    item2feature = {}
    item2description = {}    
    item2category = {}
    item2brand = {}

    if categories_used == 'all':
        for l in parse(meta_path):
            asin = l['asin']
            item2brand[asin] = l['brand'] if 'brand' in l else ''
            item2category[asin] = []
            item2feature[asin] = []
            item2description[asin]=[]

            for cs in l['category']:
                item2category[asin].append(cs)
            for ft in l['feature']:
                item2feature[asin].append(clean_html(ft))
            for dc in l['description']:
                item2description[asin].append(clean_html(dc))
            
    else:
        for l in parse(meta_path):
            asin = l['asin']
            item2description[asin] = clean_html(l['description'][0]) if l['description'] and clean_html(l['description'][0]) else ''
            item2feature[asin] = clean_html(l['feature'][0]) if l['feature'] and clean_html(l['feature'][0]) else ''
            item2category[asin] = l['category'][0] if l['category'] else ''
            item2brand[asin] = l['brand'] if 'brand' in l else ''

    # Remove empty strings from the lists
    for asin in item2category:
        item2category[asin] = [cat for cat in item2category[asin] if cat]
    for asin in item2feature:
        if isinstance(item2feature[asin], list):
            item2feature[asin] = [feat for feat in item2feature[asin] if feat]
        elif isinstance(item2feature[asin], str):
            item2feature[asin] = item2feature[asin] if item2feature[asin].strip() else ''
    for asin in item2description:
        if isinstance(item2description[asin], list):
            item2description[asin] = [desc for desc in item2description[asin] if desc]
        elif isinstance(item2description[asin], str):
            item2description[asin] = item2description[asin] if item2description[asin].strip() else ''

    items_meta = {
        'item2feature': item2feature,
        'item2description': item2description,
        'item2category': item2category,
        'item2brand': item2brand
    }
    return items_meta

def generate_data(dataset_name, reviews_path, meta_path):
    category_used_list = ['all']
    min_units = ['multi_word']

    for categories_used in category_used_list:
        for min_unit in min_units:
            user2id = {'[PAD]': 0}
            item2id = {'[PAD]': 0}
            items_map = {
                'item2feature': {},
                'item2description': {},
                'item2category': {},
                'item2brand': {}
            }
            user_reviews = defaultdict(list)
            action_times = []
            items_meta = get_items_meta(meta_path, categories_used)

            for l in parse(reviews_path):
                if l['reviewerID'] not in user2id:
                    user2id[l['reviewerID']] = len(user2id)
                action_times.append(l['unixReviewTime'])
                user_reviews[l['reviewerID']].append([l['asin'], l['unixReviewTime']])

            for u in user_reviews:
                user_reviews[u].sort(key=lambda x: x[1])
                for item, time in user_reviews[u]:
                    if item not in item2id:
                        item2id[item] = len(item2id)
            item2_id_list=[] 
            for item in item2id.keys():
                item2_id_list.append(item)  
            
            items_meta_itemid_list=[]
            not_match_item_id=[]
            items_meta_itemid_list=set(list(items_meta['item2brand'].keys())) # meta data itemid
            
            for item in item2_id_list:
                if item not in items_meta_itemid_list:
                      not_match_item_id.append(item) # meta item
            not_match_item_id.remove('[PAD]')  

            remove_unmatch_item_item2id={}
            for itemid in item2id:
                if itemid not in not_match_item_id: 
                    if itemid not in remove_unmatch_item_item2id.keys(): # remove duplicate & first appear
                        remove_unmatch_item_item2id[itemid] = len(remove_unmatch_item_item2id)
            
            item2id=remove_unmatch_item_item2id

            ### user_reviews
            item_purchase_counts = defaultdict(int)

            # user_reviews딕셔너리에서 매핑 안되는 아이템 삭제 후 5개 이상 상호작용한 user만 추출
            for user, interactions_list in user_reviews.items():
                for item, action_time in interactions_list:
                    if item in item2id:  # item2id에 존재하는 아이템만 고려
                        item_purchase_counts[item] += 1

            user_over_5_item_purchases = {}

            for user, interactions_list in user_reviews.items():
                item_count = sum(1 for item, action_time in interactions_list if item in item_purchase_counts)
                if item_count >= 5:
                    user_over_5_item_purchases[user] = interactions_list
            user_reviews=defaultdict(list)
            user_reviews=user_over_5_item_purchases

            ### 
            #items_map : review에 매칭되는 항목 저장
            for u in user_reviews:
                user_reviews[u].sort(key=lambda x: x[1])
                for item, time in user_reviews[u]:
                    for s in ['item2feature','item2description','item2category', 'item2brand']:  
                        if item in items_meta[s] and item not in not_match_item_id:
                            items_map[s][item] = items_meta[s][item]  # item_map=41280
            ### brand 매핑
            brand2id = {'[PAD]': 0} 
            item2brand_id = {}
            for k in items_map['item2brand'].keys():
                if items_map['item2brand'][k] in brand2id:
                    item2brand_id[k] = brand2id[items_map['item2brand'][k]]
                else:
                    brand2id[items_map['item2brand'][k]] = len(brand2id)  # 존재하지 않으면 새로운 ID를 생성하여 brand2id에 추가
                    item2brand_id[k] = brand2id[items_map['item2brand'][k]]  # item2brand_id에 해당 아이템의 브랜드 ID를 저장
            ### category 매핑
            category2id = {'[PAD]': 0}
            item2category_id = defaultdict(list)
            categories_n_max = 0  # 각 아이템이 가질 수 있는 최대 카테고리 수
            if min_unit == 'single_word': # 카테고리를 단어로 분할하여 처리 ex) "Men's Clothing"->"Men's":ID 1, "Clothing":ID 2
                for k in items_map['item2category'].keys():
                    for category in items_map['item2category'][k]:
                        for w in category.split(" "):
                            if w not in category2id:
                                category2id[w] = len(category2id)
                            if category2id[w] not in item2category_id[k]:
                                item2category_id[k].append(category2id[w])
                    categories_n_max = len(item2category_id[k]) if len(
                        item2category_id[k]) > categories_n_max else categories_n_max
            else:  # 카테고리 전체를 하나의 단위로 처리 ex) "Men's Clothing"->"Men's Clothing":ID 1
                for k in items_map['item2category'].keys(): 
                    for category in items_map['item2category'][k]: 
                        if category not in category2id:
                            category2id[category] = len(category2id)
                        if category2id[category] not in item2category_id[k]:
                            item2category_id[k].append(category2id[category])
                    categories_n_max = len(item2category_id[k]) if len(
                        item2category_id[k]) > categories_n_max else categories_n_max
            ### description 매핑
            description2id = {'[PAD]': 0}
            item2description_id = defaultdict(list)
            descriptions_n_max = 0  
            if min_unit == 'single_word': 
                for k in items_map['item2description'].keys():
                    for description in items_map['item2description'][k]:
                        for w in description.split(" "):
                            if w not in description2id:
                                description2id[w] = len(description2id)
                            if description2id[w] not in item2description_id[k]:
                                item2description_id[k].append(description2id[w])
                    descriptions_n_max = len(item2description_id[k]) if len(
                        item2description_id[k]) > descriptions_n_max else descriptions_n_max
            else:  
                for k in items_map['item2description'].keys():
                    for description in items_map['item2description'][k]:
                        if description not in description2id:
                            description2id[description] = len(description2id)
                        if description2id[description] not in item2description_id[k]:
                            item2description_id[k].append(description2id[description])
                    descriptions_n_max = len(item2description_id[k]) if len(
                        item2description_id[k]) > descriptions_n_max else descriptions_n_max
            
            ### feature 매핑 
            feature2id = {'[PAD]': 0}
            item2feature_id = defaultdict(list)
            features_n_max = 0  
            if min_unit == 'single_word': 
                for k in items_map['item2feature'].keys():
                    for feature in items_map['item2feature'][k]:
                        for w in feature.split(" "):
                            if w not in feature2id:
                                feature2id[w] = len(feature2id)
                            if feature2id[w] not in item2feature_id[k]:
                                item2feature_id[k].append(feature2id[w])
                    features_n_max = len(item2feature_id[k]) if len(
                        item2feature_id[k]) > features_n_max else features_n_max
            else:  
                for k in items_map['item2feature'].keys():
                    for feature in items_map['item2feature'][k]:
                        if feature not in feature2id:
                            feature2id[feature] = len(feature2id)
                        if feature2id[feature] not in item2feature_id[k]:
                            item2feature_id[k].append(feature2id[feature])
                    features_n_max = len(item2feature_id[k]) if len(
                        item2feature_id[k]) > features_n_max else features_n_max
            
            data = {
                'user_seq': user_reviews,
                'items_map': items_map,  # review에 있는 실제 item 값에 매칭되는 특징들 
                'user2id': user2id,  # userid factorizing
                'item2id': item2id,  # itemid factorizing
                'category2id': category2id,
                'brand2id': brand2id,
                'max_categories_n': categories_n_max,
                'max_features_n':features_n_max,
                'max_descriptions_n':descriptions_n_max
            }
        return data

if __name__ == '__main__':
    for dataset_name in ['All_Beauty']:
        example_data = generate_data(dataset_name, f'data/{dataset_name}.json.gz',
                      f'data/meta_{dataset_name}.json.gz')
