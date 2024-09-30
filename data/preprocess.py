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


def write_mapping(User, dataset_name):
    f = open(f'{dataset_name}.txt','w')
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


def write_seq(User, dataset_name):
    f = open(f'{dataset_name}_seq.txt','w')
    for user in User.keys():
        f.write('%d '%(user))
        for i in User[user]:
            f.write('%d ' %(i[1]))
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
                import IPython; IPython.embed(colors='Linux');exit(1);
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


def Amazon_generate_data(dataset_name):
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

        if countU[rev] < 5 or countP[asin] < 5:
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
    write_mapping_sample(User, dataset_name)
    write_mapping(User,dataset_name)
    write_seq(User,dataset_name)
    write_seq_sample(User,dataset_name)


def ml_generate_data(path):

    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)

    for row in csv.reader(open(path,'r',encoding='utf-8')):
        user_id, movie_id, rating, timestamp = row[0].split('::')
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
        user_id, movie_id, rating, timestamp = row[0].split('::')
        
        if countU[user_id] < 5 or countP[movie_id] < 5:
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

    get_stats(usernum,itemnum,User,Item,'Ml-1m')
    write_mapping_sample(User,'Ml-1m')
    write_mapping(User,'Ml-1m')
    write_seq(User,'Ml-1m')
    write_seq_sample(User,'Ml-1m')

amazon = ['Beauty', 'Toys_and_Games', 'Video_Games','Sports_and_Outdoors']
# for i in amazon:
#     Amazon_generate_data(i)


#steam_generate_data('/Users/sb/Desktop/project/preference_rec/data/australian_user_reviews.json.gz') #using version 1 review data
ml_generate_data('/home/sb/data/ml-20m.inter')  #m1-1m
    
# import IPython; IPython.embed(colors='Linux');exit(1);


