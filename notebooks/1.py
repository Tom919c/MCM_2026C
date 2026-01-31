import pickle
path = r"D:/Code/VSCode/MCM_2026C/data/processed\datas.pkl"
with open(path,'rb') as f:
    datas = pickle.load(f)
#print(datas.keys())
tf = datas['train_data']
#print(tf.keys())
print(tf["X_pro_names"])

# train_data['X_celeb_names']  # list[str]
# train_data['X_pro_names']    # list[str]
# train_data['X_obs_names']    # list[str]
print()

print(tf['X_obs_names'])

print()

print(tf['X_celeb_names'])