# from pymongo import MongoClient
import pymongo
from bson.json_util import dumps


def init():
    url = "mongodb+srv://enmingg:gem010131@cluster0.s6idkhp.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(url)
    return client.Alzhemier
def update_authority(Alzhemier,data):
    username = data.get('username')
    authority = data.get('authority')
    Alzhemier.user.update_many(
        {"username": username},
        {"$set": {"authority": authority}}
    )
    return True;
def get_authority(Alzhemier,username):
    result = Alzhemier.user.find_one({"username": username})

    # 提取"authority"字段值并将其作为字符串返回
    authority = result.get("authority") if result else ""
    authority_string = str(authority)
    print(authority_string)
    return authority_string

def is_username_available(Alzhemier,username):
    user_collection = Alzhemier.user
    existing_user = user_collection.find_one({"username": username})
    if existing_user:
        print("Username already exists.")
        return False
    else:
        print("Username is available.")
        return True

def insert_mydata(Alzhemier,data):
    print(data)
    username = data.get('username')
    if(is_username_available(Alzhemier,username)==False):
        return False
    result = Alzhemier.user.insert_one(data);
    if result.acknowledged:
        print("Insertion successful!")
        print("Inserted document ID:", result.inserted_id)
        return True
    else:
        print("Insertion failed.")
        return False
def get_all_user(Alzhemier):
    User_collection = Alzhemier.user
    matching_items = User_collection.find({
        "username": {"$exists": True}
    }).sort("username", -1)
    return matching_items
def update_data(Alzhemier,data):
    username = data.get('username')
    original = data.get('original')
    if(username != original and is_username_available(Alzhemier,username)==False):
        return False

    delete_result = Alzhemier.user.delete_many({'username': original})
    new_data={
        "username": data['username'],
        "password": data['password'],
        "email": data['email'],
        "gender": data['gender'],
        "birthday": data['birthday'],
        'interests': data['interests'],
        "authority": data['authority']
    }
    result = Alzhemier.user.insert_one(new_data);
    if(username != original):
        Alzhemier.case.update_many(
            {"username": original},
            {"$set": {"username": username}}
        )
    if result.acknowledged:
        print("Insertion successful!")
        print("Inserted document ID:", result.inserted_id)
        return True
    else:
        print("Insertion failed.")
        return False

def find_username(Alzhemier,username):
    user_collection = Alzhemier.user
    existing_user = user_collection.find_one({"username": username})
    if existing_user:
        return existing_user.get('password')
    else:
        print("Username doesn't exist.")
        return False
def get_username(Alzhemier,username):
    user_collection = Alzhemier.user
    existing_user = user_collection.find_one({"username": username})
    if existing_user:
        return existing_user
    else:
        print("Username doesn't exist.")
        return False

def check_Login(Alzhemier,data):
    username = data.get('username')
    password = data.get('password')
    result = find_username(Alzhemier,username)
    if(result==False):
        return 0
    else:
        get_password = result
        if(get_password == password):
            return 1
        else:
            return 2
def insert_Training(Alzhemier,data):
    print(data)
    result = Alzhemier.Training.insert_one(data);
    if result.acknowledged:
        print("Insertion successful!")
        print("Inserted document ID:", result.inserted_id)
        return True
    else:
        print("Insertion failed.")
        return False
def insert_case(Alzhemier,data):
    print(data)
    result = Alzhemier.case.insert_one(data);
    if result.acknowledged:
        print("Insertion successfully!")
        print("Inserted document ID:",result.inserted_id)
        return True
    else:
        print("Insertion failed.")
        return False
def insert_message(Alzhemier,data):
    result = Alzhemier.message.insert_one(data);
    if result.acknowledged:
        print("Insertion successfully!")
        print("Inserted")
        return True
    else:
        print("Insertion failed.")
        return False
def update_Training(Alzhemier,data):
    request_Id= data.get('request_Id')
    Training_collection = Alzhemier.Training;
    existing_user = Training_collection.find_one({"request_Id": request_Id})
    if existing_user:
        existing_user['train_accuracy'] = data.get('train_accuracy')
        existing_user['test_accuracy'] = data.get('test_accuracy')
        Training_collection.update_one({"request_Id": request_Id}, {"$set": existing_user})
        return True
    else:
        print("Username doesn't exist.")
        return False

def get_all_Training(Alzhemier,data):
    username = data.get('username')
    Training_collection = Alzhemier.Training
    matching_items = Training_collection.find({
        "username": username,
        "test_accuracy": {"$exists": True},
        "train_accuracy": {"$exists": True},
        "selected_model": {"$ne": "", "$exists": True}
    }).sort("datetime", -1)
    return matching_items

def get_all_testcase(Alzhemier, data):
    username = data.get('username')
    Case_collection = Alzhemier.case
    if get_authority(Alzhemier,username) == 'advanced':
        matching_items = Case_collection.find({
            "datetime": {"$exists": True},
            'Normal Case Probability': {"$exists": True}
        }).sort('datetime', -1)
        return matching_items
    else:
        matching_items = Case_collection.find({
            "username": username,
            "datetime": {"$exists": True},
            'Normal Case Probability': {"$exists": True}
        }).sort('datetime', -1)
        return matching_items
