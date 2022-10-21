import PyPDF2
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import en_core_web_sm
from spacy.matcher import PhraseMatcher
from matplotlib.gridspec import GridSpec

from flask import Flask
app = Flask(__name__)

nlp = en_core_web_sm.load()

#Function to read resumes from the folder one by one
mypath='Resume/' 
onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    
def pdfextract(file):
    fileReader = PyPDF2.PdfFileReader(open(file,'rb'), strict=False)
    countpage = fileReader.getNumPages()
    count = 0
    text = []
    while count < countpage:    
        pageObj = fileReader.getPage(count)
        count +=1
        t = pageObj.extractText()
        print ("+++++++++++++********",t)
        text.append(t)
    return text

#function to read resume ends

#function that does phrase matching and builds a candidate profile
def create_profile(file):
    text = pdfextract(file) 
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    #below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('template_new.csv')
    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
    R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]
    marketing_words = [nlp(text) for text in keyword_dict['Marketing'].dropna(axis = 0)]
    java_words = [nlp(text) for text in keyword_dict['Java'].dropna(axis = 0)]
    unknown_words = [nlp(text) for text in keyword_dict['Unknown'].dropna(axis = 0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('Machine Learning', None, *ML_words)
    matcher.add('Deep Learning', None, *DL_words)
    matcher.add('R', None, *R_words)
    matcher.add('Python', None, *python_words)
    matcher.add('Data Engineering', None, *Data_Engineering_words)
    matcher.add('Marketing', None, *marketing_words)
    matcher.add('Java', None, *java_words)
    matcher.add('Unknown', None, *unknown_words)
    doc = nlp(text)
    
    d = []  
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        d.append((rule_id, span.text))      
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
    
    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

    return(dataf)

@app.route('/')
def index():        
#   execute/call the above functions

    final_database=pd.DataFrame()
    i = 0 
    while i < len(onlyfiles):
        file = onlyfiles[i]
        dat = create_profile(file)
        final_database = final_database.append(dat)
        i +=1
        print(final_database)

    #  count words under each category and visulaize it through Matplotlib
    final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
    final_database2.reset_index(inplace = True)
    final_database2.fillna(0,inplace=True)
    new_data = final_database2.iloc[:,1:]
    new_data.index = final_database2['Candidate Name']
    sample2=new_data.to_csv('sample.csv')

    resumeDataSet = pd.read_csv('sample.csv' ,encoding='utf-8')
    print(resumeDataSet)
    print("--------------------------++++++------------------------------")
    resumeDataSet['UnKnown'] = ''
    print(resumeDataSet)

    labels = []
    for j in new_data.columns:
        for i in new_data.index:
            label = str(j)+": " + str(new_data.loc[i][j])
            labels.append(label)
    print("label", labels)
    print("TUPLE", tuple(labels))
    tup = tuple(labels)
 
    arpit = [i.split(': ') for i in tup]
    print("arpit", arpit)

    aryan = []
    for x in arpit:
        aryan.append({x[0]:x[1]})
    d= defaultdict(list)
    for i in aryan:
        for key, value in i.items():
            d[key].append(value)

    for key, value in d.items():
        # for val in value:
        #     float(val)    
        d.update({key:[float(val) for val in value]})

    for key, value in d.items(): 
        d.update({key:sum(value)})

    nik = dict(d)
    print("NIK", nik)
    targetCounts = nik.values()
    targetLabels  = nik.keys()
    # Make square figures and axes
    plt.figure(1, figsize=(200,22))
    plt.get_cmap('coolwarm')

    plt.pie(targetCounts,radius = 0.8, labels=targetLabels, autopct='%1.1f%%')
    plt.show()
   
    return ("Graph plotted successfully..")
    

    
    # print('aryan', aryan)
    # print("Type ", type(aryan))
    # result = {}
    # data_list = []
    # years_dict = dict()

    # for line in aryan:
    #     if line[0] in years_dict:
    #         # append the new number to the existing array at this slot
    #         years_dict[line[0]].append(line[1])
    #     else:
    #         # create a new array in this slot
    #         years_dict[line[0]] = [line[1]]
    # for d in aryan:
    #     for k, v in d.items():
    #         result[k] = result.get(k, 0) + v
    # print("result", result)


    # for i in aryan:

    #    for (key, value) in i.items():
    #         # data_list.append(value)
    #         print(key, value)
    #         # result[key] = i.keys()
    #         # result[value] = i.values()

    #         #  data_list.append(value)
    # print("result", result)


# data = [24, 6, 16, 36, 28]
    # label = ['Natural Gas', 'Hydro', 'Nuclear', 'Oil', 'Coal']
    
    # plt.pie(data, labels=label, autopct='%1.1f%%', explode=[0,0,0,0.1,0], shadow=True, startangle=90)
    # plt.title('World Energy Consumption')
    # plt.axis('equal')
    # plt.show()

    # plt.figure(figsize=(20,5))
    # plt.xticks(rotation=90)
    # ax=sns.countplot(x="Category", data=resumeDataSet)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    # plt.grid()

    # graph work
    # plt.rcParams.update({'font.size': 10})

    # ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)


    # res_dct = {tuple(sub[:2]): tuple(sub[2:]) for sub in arpit}                                                                       #map(lambda i: (arpit[i], arpit[i+1]), range(len(arpit)-1)[::2])
    # print("res_dct", dict(res_dct))

    # patches = ax.patches
    # print("patches", patches)
    # print("final_database2------>>>", final_database2)
    # print("ax", ax)

    # for label, rect in zip(labels, patches):
    #     width = rect.get_width()
    #     if width > 0:
    #         x = rect.get_x()
    #         y = rect.get_y()
    #         height = rect.get_height()
    #         ax.text(x + width/2., y + height/2., label, ha='center', va='center')    
    #         plt.pie(labels, labels=rect, autopct='%1.1f%%', explode=[0,0,0,0.1,0], shadow=True)
    # plt.show()

    # data = [24, 6, 6, 36, 28]
    # label = ['Natural Gas', 'Hydro', 'Nuclear', 'Oil', 'Coal']
    # plt.pie(labels, labels=rect, autopct='%1.1f%%', explode=[0,0,0,0.1,0], shadow=True, startangle=90)
    # plt.title('World Energy Consumption')
    # plt.axis('equal')
    # plt.show()
    # print("-----------------------------------------------------------")
    # plt.figure(figsize=(20,5))
    # plt.xticks(rotation=90)
    # ax=sns.countplot(x="Subject", data=final_database2)
    # for p in ax.patches:
    #     ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    # plt.grid()

    # targetCounts = final_database2['Subject'].value_counts()
    # targetLabels  = final_database2['Candidate Name'].unique()
    # # Make square figures and axes
    # plt.figure(1, figsize=(22,22))
    # the_grid = GridSpec(2, 2)


    # cmap = plt.get_cmap('coolwarm')
    # plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

    # source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)
    # plt.show()

    # print("-----------------------------------------------------------")

    # plt.pie(final_database['Subject'], labels=final_database.index, explode = (0.1,0,0,0,0,0), autopct='%1.0f%%',shadow=True,startangle=90)
    # pie = plt.figure(figsize=(10,10))
    # plt.pie(final_database['Keyword'], labels=final_database.index, explode = (0.1,0,0,0,0,0), autopct='%1.0f%%',shadow=True,startangle=90)
    # plt.title('Resume keywords by category')
    # plt.axis('equal')
    # plt.show()



    # arr = np.array([x, y, height, width])
    # # ax_list = float(ax)
    # # print("ax_list",ax_list)s
    # op_col = []
    # for i in new_data['Keyword']:
    #     op_col.append(i)
    # print(op_col)
    # plt.pie(new_data) 

    # return ("Graph plotted successfully..")


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)






    # [['DE', '0.0'], ['DE', '4.0'], ['DE', '4.0'], ['DE', '2.0'], 
    # ['DE', '0.0'], ['DE', '0.0'], ['DE', '0.0'], ['DE', '4.0'], 
    # ['MKT', '2.0'], ['MKT', '1.0'], ['MKT', '1.0'], ['MKT', '1.0'], 
    # ['MKT', '1.0'], ['MKT', '1.0'], ['MKT', '0.0'], ['MKT', '1.0'], 
    # ['ML', '0.0'], ['ML', '0.0'], ['ML', '0.0'], ['ML', '0.0'], 
    # ['ML', '0.0'], ['ML', '0.0'], ['ML', '1.0'], ['ML', '0.0'], 
    # ['NLP', '0.0'], ['NLP', '0.0'], ['NLP', '0.0'], ['NLP', '0.0'], 
    # ['NLP', '0.0'], ['NLP', '0.0'], ['NLP', '1.0'], ['NLP', '0.0'], 
    # ['Python', '1.0'], ['Python', '0.0'], ['Python', '0.0'], 
    # ['Python', '0.0'], ['Python', '1.0'], ['Python', '1.0'], 
    # ['Python', '0.0'], ['Python', '2.0']]

    # [['DE', '0.0'], ['DE', '4.0'], ['DE', '4.0'], ['DE', '2.0'], 
    # ['DE', '0.0'], ['DE', '0.0'], ['DE', '0.0'], ['DE', '4.0'],
    # ['MKT', '2.0'], ['MKT', '1.0'], ['MKT', '1.0'], ['MKT', '1.0'], 
    # ['MKT', '1.0'], ['MKT', '1.0'], ['MKT', '0.0'], ['MKT', '1.0'], 
    # ['ML', '0.0'], ['ML', '0.0'], ['ML', '0.0'], ['ML', '0.0'], 
    # ['ML', '0.0'], ['ML', '0.0'], ['ML', '1.0'], ['ML', '0.0'], 
    # ['NLP', '0.0'], ['NLP', '0.0'], ['NLP', '0.0'], ['NLP', '0.0'], 
    # ['NLP', '0.0'], ['NLP', '0.0'], ['NLP', '1.0'], ['NLP', '0.0'], 
    # ['Python', '1.0'], ['Python', '0.0'], ['Python', '0.0'], 
    # ['Python', '0.0'], ['Python', '1.0'], ['Python', '1.0'], 
    # ['Python', '0.0'], ['Python', '2.0']]