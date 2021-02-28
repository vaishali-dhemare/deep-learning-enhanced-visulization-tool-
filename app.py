import json
from json import JSONEncoder
import numpy as np
import pandas as pd
from flask import Flask, request
from sklearn.model_selection import train_test_split
from extensions import cache
from dnn import dnn_api
from decisiontree import decision_tree_api
app = Flask(__name__)
app.register_blueprint(dnn_api)
app.register_blueprint(decision_tree_api)

#cache = {}

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers[
        'Access-Control-Allow-Headers'] = "Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With"
    response.headers['Access-Control-Allow-Methods'] = "POST, GET, PUT, DELETE, OPTIONS"
    return response


@app.route("/UploadCSVFile", methods=["POST"])
def UploadCSVFile():
    if request.method == 'POST':
       files = request.files['uploadedFile']
    df = pd.read_csv(files)
    cache['uploadedFile'] = df
    df.describe()
    X = df.drop('Target', axis=1)  # axis=1 means column and axis 0 is row
    Y = df['Target']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    from sklearn.tree import DecisionTreeClassifier
    tree_model = DecisionTreeClassifier()
    tree_model = tree_model.fit(x_train, y_train)
    y_predict = tree_model.predict(x_test)
    pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
    Label = y_predict
    x_test["Source (one)"] = x_test["Source (one)"].astype(str)
    x_test["Source (one)"]
    x_test["Source (two)"] = x_test["Source (two)"].astype(str)
    x_test["Source (one)"]
    x_test["Source (three)"] = x_test["Source (three)"].astype(str)
    x_test["Source (three)"]
    x_test["Source (four)"] = x_test["Source (four)"].astype(str)
    x_test["Source (four)"]
    x_test["Source IP"] = x_test["Source (one)"] + '.' + x_test["Source (two)"] + '.' + x_test["Source (three)"] + '.' + \
                          x_test["Source (four)"]
    x_test["Destination (one)"] = x_test["Destination (one)"].astype(str)
    x_test["Destination (two)"] = x_test["Destination (two)"].astype(str)
    x_test["Destination (three)"] = x_test["Destination (three)"].astype(str)
    x_test["Destination (four)"] = x_test["Destination (four)"].astype(str)
    x_test["Destination IP"] = x_test["Destination (one)"] + '.' + x_test["Destination (two)"] + '.' + x_test[
        "Destination (three)"] + '.' + x_test["Destination (four)"]
    x_test = x_test.drop(['Source (one)', 'Source (two)', 'Source (three)', 'Source (four)'], axis=1)
    x_test = x_test.drop(['Destination (one)', 'Destination (two)', 'Destination (three)', 'Destination (four)'],
                         axis=1)
    x_test = x_test.drop(x_test.columns[0], axis=1)
    x_test = x_test.drop(x_test.columns[0], axis=1)
    x_test = x_test.drop(x_test.columns[1], axis=1)
    Label = pd.DataFrame(Label)
    Label.columns = ['Label']
    Label = Label.astype(str)
    new_test = x_test
    new_test['Label'] = Label
    Darpa_data = new_test
    df = Darpa_data
    df.info()
    df['Dest_IP_count'] = df.groupby('Destination IP')['Destination IP'].transform('count')
    df["Label"] = df["Label"].astype(str)
    df["Protocol"] = df["Protocol"].astype(str)
    result = df.groupby('Destination IP').agg({'Label': lambda x: x.iloc[0]})
    result['count'] = df['Destination IP'].value_counts()
    result.reset_index(inplace=True)
    result.head(95)
    df['1s_Dest_Count'] = df.groupby(["Destination IP", "Label"])["Destination IP"].transform("count")
    df.tail()
    df['Dest_id'] = df['Destination IP'].str.replace(r'\D', '')
    df['Source_id'] = df['Source IP'].str.replace(r'\D', '')
    df['Label'] = df['Label'].replace({'0': np.nan, 0: np.nan})
    df.tail(15)
    df['1s_Dest_Count'] = np.where(df['Label'].isnull(), df['Label'], df['1s_Dest_Count'])
    df = df.fillna(0)
    df['Score'] = list(map(lambda x, y: x / y, df['1s_Dest_Count'], df['Dest_IP_count']))
    df['weight'] = df.groupby(["Destination IP", "Source IP"])["Destination IP"].transform("count")
    df['weight'] = df['weight'] / 100
    my_list = []

    for ind in df.index:
        rawNodeDetailsType = RawNodeDetailsType(df['Dest_id'][ind], df['Destination IP'][ind], 0)
        vsData = RawVisualizationData(rawNodeDetailsType, 'nodes', True, True)
        my_list.append(vsData)
        rawNodeDetailsTypeSource = RawNodeDetailsType(df['Source_id'][ind], df['Source IP'][ind], 0)
        vsDataSource = RawVisualizationData(rawNodeDetailsTypeSource, 'nodes', True, True)
        my_list.append(vsData)
        my_list.append(vsDataSource)
        edge = Edge(df['Source_id'][ind], df['Dest_id'][ind], 0, df['Source_id'][ind])
        edgeData = EdgeData(edge)
        my_list.append(edgeData)

    visualizationJSONData = json.dumps(my_list, indent=4, cls=VisualizationDataEncoder)

    return visualizationJSONData


@app.route("/svm", methods=["GET"])
def svm():
    df = cache['uploadedFile']
    df.describe()
    print(df)
    X = df.drop('Target', axis=1)  # axis=1 means column and axis 0 is row
    Y = df['Target']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    from sklearn.svm import SVC
    svm_model = SVC()
    svm_model = svm_model.fit(x_train, y_train)
    y_predict = svm_model.predict(x_test)
    print(y_predict)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_predict))
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_predict))
    pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
    Label = y_predict
    x_test["Source (one)"] = x_test["Source (one)"].astype(str)
    x_test["Source (one)"]
    x_test["Source (two)"] = x_test["Source (two)"].astype(str)
    x_test["Source (one)"]
    x_test["Source (three)"] = x_test["Source (three)"].astype(str)
    x_test["Source (three)"]
    x_test["Source (four)"] = x_test["Source (four)"].astype(str)
    x_test["Source (four)"]
    x_test["Source IP"] = x_test["Source (one)"] + '.' + x_test["Source (two)"] + '.' + x_test["Source (three)"] + '.' + \
                          x_test["Source (four)"]
    x_test["Destination (one)"] = x_test["Destination (one)"].astype(str)
    x_test["Destination (two)"] = x_test["Destination (two)"].astype(str)
    x_test["Destination (three)"] = x_test["Destination (three)"].astype(str)
    x_test["Destination (four)"] = x_test["Destination (four)"].astype(str)
    x_test["Destination IP"] = x_test["Destination (one)"] + '.' + x_test["Destination (two)"] + '.' + x_test[
        "Destination (three)"] + '.' + x_test["Destination (four)"]
    x_test = x_test.drop(['Source (one)', 'Source (two)', 'Source (three)', 'Source (four)'], axis=1)
    x_test = x_test.drop(['Destination (one)', 'Destination (two)', 'Destination (three)', 'Destination (four)'],
                         axis=1)
    x_test = x_test.drop(x_test.columns[0], axis=1)
    x_test = x_test.drop(x_test.columns[0], axis=1)
    x_test = x_test.drop(x_test.columns[1], axis=1)
    Label = pd.DataFrame(Label)
    Label.columns = ['Label']
    Label = Label.astype(str)
    new_test = x_test
    new_test['Label'] = Label
    Darpa_data = new_test
    df = Darpa_data
    df.info()
    df['Dest_IP_count'] = df.groupby('Destination IP')['Destination IP'].transform('count')
    df["Label"] = df["Label"].astype(str)
    df["Protocol"] = df["Protocol"].astype(str)
    result = df.groupby('Destination IP').agg({'Label': lambda x: x.iloc[0]})
    result['count'] = df['Destination IP'].value_counts()
    result.reset_index(inplace=True)
    result.head(95)
    df['1s_Dest_Count'] = df.groupby(["Destination IP", "Label"])["Destination IP"].transform("count")
    df.tail()
    df['Dest_id'] = df['Destination IP'].str.replace(r'\D', '')
    df['Source_id'] = df['Source IP'].str.replace(r'\D', '')
    df['Label'] = df['Label'].replace({'0': np.nan, 0: np.nan})
    df.tail(15)
    df['1s_Dest_Count'] = np.where(df['Label'].isnull(), df['Label'], df['1s_Dest_Count'])
    df = df.fillna(0)
    df['Score'] = list(map(lambda x, y: x / y, df['1s_Dest_Count'], df['Dest_IP_count']))
    df['weight'] = df.groupby(["Destination IP", "Source IP"])["Destination IP"].transform("count")
    df['weight'] = df['weight'] / 10
    my_list = []

    for ind in df.index:
        analysedNodeDetailsType = AnalysedNodeDetailsType(df['Dest_id'][ind],df['Destination IP'][ind],df['Score'][ind])
        vsData = AnalysedVisualizationData(analysedNodeDetailsType,'nodes',True,True)
        my_list.append(vsData)
        AnalysedNodeDetailsTypeSource = AnalysedNodeDetailsType(df['Source_id'][ind], df['Source IP'][ind], df['Score'][ind])
        vsDataSource = AnalysedVisualizationData(AnalysedNodeDetailsTypeSource, 'nodes', True, True)
        my_list.append(vsData)
        my_list.append(vsDataSource)
        edge = Edge(df['Source_id'][ind],df['Dest_id'][ind],df['Score'][ind],df['Source_id'][ind])
        edgeData = EdgeData(edge)
        my_list.append(edgeData)

    visualizationJSONData = json.dumps(my_list, indent=4, cls=VisualizationDataEncoder)

    return visualizationJSONData

class AnalysedNodeDetailsType:
    def __init__(self, id, name, score):
        self.id = id
        self.name = name
        self.score = score
        self.vulnerable = True if score >= 0.5 else False
        self.clean = True if score < 0.5 else False
        self.mildVulnerable = False

class AnalysedVisualizationData:
    def __init__(self, AnalysedNodeDetailsType, group, selectable,grabbable):
        self.data = AnalysedNodeDetailsType
        self.group = group
        self.selectable = selectable
        self.grabbable = grabbable

class RawNodeDetailsType:
    def __init__(self, id, name, score):
        self.id = id
        self.name = name

class RawVisualizationData:
    def __init__(self, RawNodeDetailsType, group, selectable,grabbable):
        self.data = RawNodeDetailsType
        self.group = group
        self.selectable = selectable
        self.grabbable = grabbable

class EdgeData:
    def __init__(self,edge):
        self.data = edge

class Edge:
    def __init__(self, source, target, score, id):
        self.source = source
        self.target = target
        self.weight = score
        self.group = 'clean' if score <= 0.5 else 'my'
        self.id = id+'edge'

class VisualizationDataEncoder(JSONEncoder): #json encoder to convert collection to JSON
        def default(self, o):
            return o.__dict__

@app.route("/")
def hello():
    return "Hello "

if __name__ == "__main__":
    app.run()
