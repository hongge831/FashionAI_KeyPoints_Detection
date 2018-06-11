# -- coding: utf-8 --
import pandas as pd

# train1 = pd.read_csv('/home/tanghm/Documents/YFF/project/data/train/Annotations/train_all.csv')
# evaluation1 = pd.read_csv('/home/tanghm/Documents/YFF/project/data/evaluation/fashionAI_key_points_test_b_answer_20180426.csv')
# train_all = pd.concat([train1,evaluation1])
# train_all.to_csv('../data/train/Annotations/train.csv',index=None)

data_all = pd.read_csv('../data/test/test.csv')
src = '../data/test/'

blouse = data_all[data_all['image_category']=='blouse']
skirt = data_all[data_all['image_category']=='skirt']
outwear = data_all[data_all['image_category']=='outwear']
dress = data_all[data_all['image_category']=='dress']
trousers = data_all[data_all['image_category']=='trousers']

blouse.to_csv(src+'blouse.csv',index=None)
skirt.to_csv(src+'skirt.csv',index=None)
outwear.to_csv(src+'outwear.csv',index=None)
dress.to_csv(src+'dress.csv',index=None)
trousers.to_csv(src+'trousers.csv',index=None)

#
# data_all = pd.read_csv('../data/train/Annotations/train.csv')
# src = '../data/train/Annotations/'
#
# blouse = data_all[data_all['image_category']=='blouse']
# skirt = data_all[data_all['image_category']=='skirt']
# outwear = data_all[data_all['image_category']=='outwear']
# dress = data_all[data_all['image_category']=='dress']
# trousers = data_all[data_all['image_category']=='trousers']
#
# blouse.to_csv(src+'blouse.csv',index=None)
# skirt.to_csv(src+'skirt.csv',index=None)
# outwear.to_csv(src+'outwear.csv',index=None)
# dress.to_csv(src+'dress.csv',index=None)
# trousers.to_csv(src+'trousers.csv',index=None)

#
# data_all = pd.read_csv('/home/tanghm/Documents/YFF/project/data/evaluation/test.csv')
# src = '/home/tanghm/Documents/YFF/project/data/evaluation/'
#
# blouse = data_all[data_all['image_category']=='blouse']
# skirt = data_all[data_all['image_category']=='skirt']
# outwear = data_all[data_all['image_category']=='outwear']
# dress = data_all[data_all['image_category']=='dress']
# trousers = data_all[data_all['image_category']=='trousers']
#
# blouse.to_csv(src+'blouse.csv',index=None)
# skirt.to_csv(src+'skirt.csv',index=None)
# outwear.to_csv(src+'outwear.csv',index=None)
# dress.to_csv(src+'dress.csv',index=None)
# trousers.to_csv(src+'trousers.csv',index=None)

print('data processing done！')



#这里可以做一些图像截取的工作
