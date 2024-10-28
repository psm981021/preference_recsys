import csv
import gzip
from collections import defaultdict
import json
import pandas as pd
import tqdm
import csv 
from pathlib import Path
import re
import random

def parse_gzip(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        for line in file:
            line = line.replace('\'','\"')
            yield json.loads(line)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse_gzip(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')
#df = getDF('reviews_Video_Games.json.gz')


def write_mapping(User, dataset_name,k):
    if k==5:
        f = open(f'5core/{dataset_name}.txt','w')
    elif k ==10:
        f = open(f'10core/{dataset_name}.txt','w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' %(user, i[1]))
    f.close()

def write_mapping_sample(User, dataset_name):
    f = open(f'{dataset_name}_sample.txt','w')
    random_user_id = [random.randint(1,len(User)) for _ in range(int(len(User)*0.1))]
    
    for random_user in random_user_id:  
        for i in User[random_user]:
            f.write('%d %d\n' %(random_user, i[1]))
    f.close()


def write_seq(User, dataset_name,k):
    if k==5:
        f = open(f'5core/{dataset_name}.txt','w')
    elif k ==10:
        f = open(f'10core/{dataset_name}.txt','w')
    for user in User.keys():
        f.write('%d '%(user))
        for i in User[user]:
           
            try:
                f.write('%d ' %(i[1]))
            except:
                f.write('%s ' %(i[1]))
        f.write('\n')

def write_seq_sample(User, dataset_name):
    f = open(f'{dataset_name}_seq_sample.txt','w')
    random_user_id = [random.randint(1,len(User)) for _ in range(int(len(User)*0.1))]
    for random_user in random_user_id:
        f.write('%d '%(random_user))
        for i in User[random_user]:
            f.write('%d '%(i[1]))
        f.write('\n')
    



def Avg_actions_per_user(user_dict):
    total_actions= 0
    for key in user_dict:
        total_actions += len(user_dict[key])
    
    return total_actions, total_actions/len(user_dict)

def Avg_actions_per_item(item_dict):
    total_actions = 0
    for key in item_dict:
        total_actions += len(item_dict[key])
    
    return total_actions, total_actions/len(item_dict)

def sparsity(user_dict,itemnum):
    sum_interacted = 0
    for key in user_dict:
        sum_interacted += len(user_dict[key])

    return 1 - sum_interacted/(len(user_dict)*itemnum)

def get_stats(usernum, itemnum, User, Item,dataset_name):
    interactions, avg_action_user = Avg_actions_per_user(User)
    interactions_, avg_action_item = Avg_actions_per_item(Item)

    print("-"*20)
    print(f"Dataset name: {dataset_name}")
    print(f"#of users: {usernum}")
    print(f"#of items: {itemnum}")
    print(f"#of actions: {interactions}")
    print(f"#Avg.actions/User: {avg_action_user:.4f}")
    print(f"#Avg.actions/Item: {avg_action_item:.4f}")
    print(f"Sparsity: {sparsity(User,itemnum):.4f}")
    print("-"*20)

def steam_generate_data(file_path):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)

    usermap = dict()
    usernum = 0
    itemmap = dict() 
    itemnum = 0
    User = dict()
    Item = dict()

    line = 0
    interaction = 0
    #dictionary purpose, to count user_id and item_id
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for l in gz_file:
            
            l = l.replace('\'','\"')
            #import IPython; IPython.embed(colors='Linux');exit(1);
            steam_id = re.findall(r'"user_id": "(.*?)",', l)[0]
            game_id= re.findall(r'"item_id": "(.*?)",', l)

            for item in game_id:
                interaction+=1
                countP[item] +=1
                countU[steam_id] +=1
    count = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for l in gz_file:
            l = l.replace('\'','\"')
            line +=1
            steam_id = re.findall(r'"user_id": "(.*?)",', l)[0]
            game_id= re.findall(r'"item_id": "(.*?)",', l)
            
            for item in game_id:
                if countU[steam_id] < 3 or countP[item] < 3:
                    count+=1
                    continue
                

                if steam_id in usermap:
                    userid = usermap[steam_id]
                else:
                    usernum += 1
                    userid = usernum
                    usermap[steam_id] = userid
                    User[userid] = []
                if item in itemmap:
                    itemid = itemmap[item]
                else:
                    itemnum += 1
                    itemid = itemnum+1 #item id가 2부터 시작하게 한다
                    itemmap[item] = itemid 
                    Item[itemid] =[]
                
                Item[itemid].append(userid)
                User[userid].append(itemid)

    get_stats(usernum,itemnum,User,Item,'Steam')
    print(line)
    print(count)
    print(interaction)

def steam_large(file_path):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)

    usermap = dict()
    usernum = 0
    itemmap = dict() 
    itemnum = 0
    User = dict()
    Item = dict()   

    line = 0
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for l in gz_file:
            l = l.replace("u'", "'").replace("'", '"')

            try :
                username = re.findall(r'"username": "(.*?)",', l)[0]
            except :
                username = re.findall(r'"username": u"(.*?)"', l)[0]
                
            product_id = re.findall(r'"product_id": "(.*?)",', l)
            line +=1
            for item in product_id:
                countP[item] +=1
                countU[username] +=1

    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        for l in gz_file:
            l = l.replace('\'','\"')
            line +=1
            username = re.findall(r'"user_id": "(.*?)",', l)[0]
            product_id= re.findall(r'"item_id": "(.*?)",', l)
            
            for item in product_id:
                if countU[username] < 5 or countP[item] < 5:
                    count+=1
                    continue
                

                if username in usermap:
                    userid = usermap[username]
                else:
                    usernum += 1
                    userid = usernum
                    usermap[username] = userid
                    User[userid] = []
                if item in itemmap:
                    itemid = itemmap[item]
                else:
                    itemnum += 1
                    itemid = itemnum+1 #item id가 2부터 시작하게 한다
                    itemmap[item] = itemid 
                    Item[itemid] =[]
                
                Item[itemid].append(userid)
                User[userid].append(itemid)

    get_stats(usernum,itemnum,User,Item)

def write_inter_format(User, itemmap, dataset_name):
    with open(f'{dataset_name}.inter', 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        
        
        for user_id, interactions in User.items():
            # import IPython; IPython.embed(colors='Linux');exit(1);
            for interaction in interactions:
                time_stamp, item_id = interaction
                f.write(f'{user_id}\t{item_id}\t{time_stamp}\n')

def Amazon_generate_data(dataset_name, k_core):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)

    usermap = dict()
    usernum = 0
    itemmap = dict() 
    itemnum = 0
    User = dict()
    Item = dict()
    
    line = 0
    for l in parse('reviews_'+ dataset_name + '_5.json.gz'):
        line += 1
        asin = l['asin'] #item id
        rev = l['reviewerID'] #user id
        time = l['unixReviewTime'] #timestamp
        countU[rev] += 1
        countP[asin] += 1


    line =0
    for l in parse('reviews_'+ dataset_name + '_5.json.gz'):
        line += 1
    
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        if countU[rev] < k_core or countP[asin] < k_core:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum+1 #item id가 2부터 시작하게 한다
            itemmap[asin] = itemid 
            Item[itemid] =[]
        
        Item[itemid].append(userid)
        User[userid].append([time, itemid])

    # sort reviews in User according to time

    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
    

    get_stats(usernum,itemnum,User,Item,dataset_name)
    # write_mapping_sample(User, dataset_name)
    # write_mapping(User,dataset_name,k_core)
    write_seq(User,dataset_name,k_core)
    # write_seq_sample(User,dataset_name)


def ml_generate_data(path, k_core):

    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)

    for row in csv.reader(open(path,'r',encoding='utf-8')):
        
        # user_id, movie_id, rating, timestamp = row[0].split('::')
        user_id, movie_id, rating, timestamp = row
        countU[user_id] += 1
        countP[movie_id] += 1

    line = 0
    usermap = dict()
    usernum = 0
    itemmap = dict() 
    itemnum = 0
    User = dict()
    Item = dict()

    for row in csv.reader(open(path,'r',encoding='utf-8')):
       
        line +=1
        # user_id, movie_id, rating, timestamp = row[0].split('::')
        user_id, movie_id, rating, timestamp = row
        if countU[user_id] < k_core or countP[movie_id] < k_core:
            continue
        if user_id in usermap:
            userid = usermap[user_id]
        else:
            usernum += 1
            userid = usernum
            usermap[user_id] = userid
            User[userid] = []
        
        if movie_id in itemmap:
            itemid = itemmap[movie_id]
        else:
            itemnum += 1
            itemid = itemnum+1 #item id가 2부터 시작하게 한다
            itemmap[movie_id] = itemid 
            Item[itemid] =[]

        Item[itemid].append(userid)
        User[userid].append([timestamp, movie_id])

    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    get_stats(usernum,itemnum,User,Item,'Ml-20m')

    write_seq(User, 'Ml-20m', k_core)
    write_inter_format(User, itemmap, 'Ml-20m')

def Amazon_generate_data_rec(dataset_name, k_core):
    countU = defaultdict(lambda: 0)  # 사용자 리뷰 수 카운트
    countP = defaultdict(lambda: 0)  # 아이템 리뷰 수 카운트

    usermap = dict()
    usernum = 0
    itemmap = dict() 
    itemnum = 0
    User = dict()
    Item = dict()
    
    # 리뷰 데이터 파일 경로
    data_path = 'reviews_' + dataset_name + '_5.json.gz'

    # 1차로 각 사용자와 아이템의 리뷰 수를 계산
    for l in parse(data_path):
        asin = l['asin']  # item ID
        rev = l['reviewerID']  # user ID
        countU[rev] += 1
        countP[asin] += 1

    # 2차로 k-core 조건에 맞는 사용자와 아이템만 처리
    for l in parse(data_path):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        # k-core 기준에 맞지 않으면 건너뜀
        if countU[rev] < k_core or countP[asin] < k_core:
            continue

        # 유저 ID 매핑
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []

        # 아이템 ID 매핑 (아이템 ID는 2부터 시작)
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum + 1  # item ID는 2부터 시작
            itemmap[asin] = itemid
            Item[itemid] = []

        # 아이템과 유저 상호작용 저장
        Item[itemid].append(userid)
        User[userid].append([time, itemid])


    user_ids = list(User.keys())

    # 각 유저의 리뷰를 시간 순으로 정렬
    for userid in user_ids:
        # 시간 순으로 정렬
        User[userid].sort(key=lambda x: x[0])

        # 유저의 상호작용 길이가 최소 요구되는 길이 (min_interactions) 이상인지 확인
        if len(User[userid]) < 5:
            # 상호작용이 최소 요구 길이 미만이면 해당 유저를 제거
            del User[userid]

    # 데이터 통계 출력
    get_stats(usernum, itemnum, User, Item, dataset_name)

    # 시간 순으로 정렬된 유저 데이터를 파일로 저장
    write_seq(User, dataset_name, k_core)
    write_inter_format(User, itemmap, dataset_name)



# amazon = ['Baby','Clothing_Shoes_and_Jewelry','Beauty',
#           'Electronics','Grocery_and_Gourmet_Food',
#            'Home_and_Kitchen','Movies_and_TV','Toys_and_Games', 'Video_Games','Sports_and_Outdoors',
#            'Tools_and_Home_Improvement']

# amazon = ['Sports_and_Outdoors','Clothing_Shoes_and_Jewelry','Video_Games','Tools_and_Home_Improvement']
amazon = ['Office_Products']

k_core = 10

for i in amazon:
    # Amazon_generate_data(i,k_core)
    Amazon_generate_data_rec(i,k_core)


#steam_generate_data('/Users/sb/Desktop/project/preference_rec/data/australian_user_reviews.json.gz') #using version 1 review data
# ml_generate_data('/home/sb/data/ml-20m/ratings.dat',10)  #m1-1m
    
# import IPython; IPython.embed(colors='Linux');exit(1);


