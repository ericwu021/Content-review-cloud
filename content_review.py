#from imp import reload
import docx2txt
import pandas as pd
import numpy as np

import zipfile
from PIL import Image
import io
import numpy as np

from keras.preprocessing import image
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import jieba.posseg as pseg
#jieba.enable_paddle()
import pickle

def keywords_check_func(content_input,keywords_table,legal_table,brand_name,angle):

  angle_list = []

  angle_list.append(angle)
  angle_list.append('General')

  df_keywords_filtered = keywords_table[keywords_table.Brand.isin(brand_name)]
  df_keywords_filtered = df_keywords_filtered[df_keywords_filtered.Key_Message_Category != 'Product_Picture']
  df_keywords_filtered = df_keywords_filtered[df_keywords_filtered.Angle.isin(angle_list)]
  df_keywords_filtered.reset_index(drop=True,inplace=True)

  for row_ in range(df_keywords_filtered.shape[0]):
    keywords_list = df_keywords_filtered.loc[row_,'Key_Word_Segment'].split(' ')

    while '' in keywords_list: #remove null in the list
      keywords_list.remove('')

    check_status = 0
    for key_ in keywords_list:

      if key_ in content_input:
        check_status = check_status + 1
      else:
        check_status = check_status + 0

    if check_status >= round(len(keywords_list) * 0.6) and check_status!=0:

      df_keywords_filtered.loc[row_,'Check_Status'] = 1

  final_recommendation = content_recommendation_func(df_keywords_filtered)

  legal_results = legal_check_func(content_input,legal_table)
  
  #language_check_recommendations = language_check_func(content_input)

  final_recommendation = final_recommendation + '\n'  + legal_results + '\n'

  return df_keywords_filtered,final_recommendation


def content_recommendation_func(df): #used for content generation

  check_fields = set(list(df['Key_Message_Category']))

  final_results = 1
  recommendation_str = 'Oops,以下内容需要调整:'
  overall_results_str = '以下内容撰写得不错，棒棒哒!\n'

  for keyword_category_ in check_fields:

    df_keywords_filtered = df[df.Key_Message_Category == keyword_category_]
    amount_check_fields = set(list(df_keywords_filtered['Amount']))

    overall_check = 1
    for amount_ in amount_check_fields:

      df_keywords_filtered_2nd = df_keywords_filtered[(df_keywords_filtered.Key_Message_Category == keyword_category_) & (df_keywords_filtered.Amount == amount_)]

      if amount_ == 'All':
        cat_check = df_keywords_filtered_2nd['Check_Status'].prod()

        if cat_check == 0:
          recommendation_str = recommendation_str + '\n请将以下【' + keyword_category_ + '】信息补充完整：'
          for word in df_keywords_filtered_2nd[df_keywords_filtered_2nd.Check_Status == 0]['Key_Word'].values:
            recommendation_str = recommendation_str + '\n- ' + str(word)


      else:
        cat_check = df_keywords_filtered_2nd['Check_Status'].sum()

        if cat_check == 0:
          recommendation_str = recommendation_str + '\n请将以下【' + keyword_category_ + '】信息中选择任意一项补充：'
          for word in df_keywords_filtered_2nd[df_keywords_filtered_2nd.Check_Status == 0]['Key_Word'].values:
            recommendation_str = recommendation_str + ' \n- ' + str(word)

      if cat_check >= 1:
        cat_check = 1

      overall_check = overall_check * cat_check

    final_results = final_results * overall_check

    if (overall_check >= 1):
        overall_results_temp = keyword_category_ + ': /:rose\n'

        overall_results_str = overall_results_str + overall_results_temp

    else:
      pass

  if final_results >=1:
    overall_results_str = '审核结果: 通过\n' + overall_results_str
  else:
    overall_results_str = '审核结果: 未通过\n' + overall_results_str
    
  if len(recommendation_str) <= 20:
    recommendation_str = ''
  else:
    recommendation_str = '\n' + recommendation_str
  
  return overall_results_str + recommendation_str

def content_abstraction_func(path):

  sub_content = docx2txt.process(path)
  sub_content_process = sub_content.split('\n')
  content_final_ = ''

  for n,line_ in enumerate(sub_content_process):
    if line_ != '':

      if '平台：' in line_:

        platform_ = line_.split('平台：')[1]

      elif '标题：' in line_:

        title_ = line_.split('标题：')[1]
        
      elif '内容方向：' in line_:

        angle_ = line_.split('内容方向：')[1]

      elif '品牌：' in line_:

        brand_ = line_.split('品牌：')[1]

        if '+' in brand_:
          brand_list = brand_.split('+')
          for brand_ in range(len(brand_list)):
            brand_temp = brand_list[brand_].replace(' ','')                  
            brand_list[brand_] = brand_temp
            
          if brand_list == ['爱乐维男士','爱乐维']:
            brand_list = ['爱乐维','爱乐维男士']

        else:

          brand_temp = brand_.replace(' ','')                
          brand_list = [brand_temp]

      elif '正文' in line_:

        content_start = n
        break

      else:
        content_start = 0

  for line_ in range(content_start+1,len(sub_content_process),1):
    if (sub_content_process[line_] != '') and not ('RESTRICTED' in sub_content_process[line_]):

      content_final_ = content_final_ + str(sub_content_process[line_])

  content_final_.replace(' ','')

  return title_,platform_,brand_list,angle_,content_final_

def legal_check_func(content_input,df_keywords):

  df_keywords['Check_Status'] = 0
  recommendation_str = ''
  for row_ in range(df_keywords.shape[0]):
    if df_keywords.loc[row_,'Key_Word'] in content_input:
      df_keywords.loc[row_,'Check_Status'] = 1

  df_temp = df_keywords[df_keywords.Check_Status==1].copy()
  df_temp = df_temp.reset_index(drop=True)

  if df_temp.shape[0] != 0:
    for row_ in range(df_temp.shape[0]):
      recommendation_str = recommendation_str + df_temp.loc[row_,'Key_Word'] + ' + '

    recommendation_str = '\n请将以下【不合规内容】删除：' + recommendation_str

  else:
    recommendation_str = '\n【不合规内容】未检出 + '

  return recommendation_str[:-2]


def picture_check_func(docx_path,cv_model_path):

  cv_model = tf.keras.models.load_model(cv_model_path)
  z = zipfile.ZipFile(docx_path)
  count_ = 0
  for file_ in z.namelist():

    if ('.png' in file_) or ('.jpeg' in file_) or ('.jpg' in file_):

      data = z.read(file_)
      dataEnc = io.BytesIO(data)
      img = Image.open(dataEnc)

      suffix = file_.split('/')[-1].split('.')[-1]
      temp_file_path = './/outputs//temp//' #temp dictionary for picture storage
      img.save(temp_file_path + 'temp.{}'.format(suffix))

      fn = 'temp.' + suffix
      path=temp_file_path + fn
      img=image.load_img(path, target_size=(150, 150))

      x=image.img_to_array(img)
      x=np.expand_dims(x, axis=0)
      images = np.vstack([x])

      if count_ == 0:
          sv_sub_result_last = cv_model.predict(images, batch_size=10)[0]
          sv_sub_result_last = [int(integer) for integer in sv_sub_result_last]

      else:
          sv_sub_result_current = cv_model.predict(images, batch_size=10)[0]
          sv_sub_result_current = [int(integer) for integer in sv_sub_result_current]
          sv_sub_result_last = [a + b for a, b in zip(sv_sub_result_last, sv_sub_result_current)]

      count_ = count_ + 1
        
  if (len(z.namelist()) == 0) or (count_ == 0):
    sv_sub_result_last = [0,0,0,-1] #should be updated according to the model

  return sv_sub_result_last
  
def word_segment_func(text_input):
  text_input = text_input.replace(',',' ')
  text_input = text_input.replace('，',' ')

  seg_list = jieba.cut(text_input, cut_all=False)
  text_segment = (" ".join(seg_list))
  return text_segment

def language_check_func(text_input):

  model_path = './/model//'
  NLP_model = tf.keras.models.load_model(model_path + 'NLP_model.h5')
  tokenizer = pickle.load(open(model_path + 'tokenizer.pickle','rb'))
  text_segment = word_segment_func(text_input)
  text_sequences = tokenizer.texts_to_sequences([text_segment])  
  text_padded = pad_sequences(text_sequences,padding='pre',maxlen=500)

  text_pred = NLP_model.predict(text_padded)[0][0]
  
  if text_pred > 0.5:
    text_style = '【语言风格】文采飞扬，可圈可点'
  else:
    text_style = '【语言风格】中规中矩，继续努力'  

  return text_style

#CORE
def content_review_func(file):

    docx_path = './/uploads//'
    model_folder_path = './/model//'

    df_keywords = pd.read_excel(model_folder_path + 'Keywords_DB.xlsx',sheet_name = 'Brands')
    df_legal = pd.read_excel(model_folder_path + 'Keywords_DB.xlsx',sheet_name = 'Self-check')

    df_keywords['Check_Status'] = 0

    title,platform, brand, angle, content = content_abstraction_func(docx_path + '{}'.format(file))
    df_outputs,recommendations = keywords_check_func(content,df_keywords,df_legal,brand,angle)

    cv_model_path = './/model//cv_Jul.h5'
    pictures_recommendations = picture_check_func(docx_path + 'temp.docx', cv_model_path)


    if (pictures_recommendations[-1] != 0) and (pictures_recommendations[-1] != -1):
        pictures_recommendations = '【照片检查】未通过： {} 张图片不符合要求'.format(pictures_recommendations[-1])

    elif pictures_recommendations[-1] == -1:
        pictures_recommendations = '【照片检查】未检测到照片'

    else:
        pictures_recommendations = '【照片检查】通过'


    brand_str = ''
    for b_ in brand:
        brand_str = brand_str + b_ + ' + '

    return ('文章标题：' + title + '\n内容角度:' + brand_str[:-2] + '\n发布平台:' + platform + '\n' + \
              recommendations + '\n' + pictures_recommendations)
