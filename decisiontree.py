import json
from json import JSONEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Blueprint
from extensions import cache

decision_tree_api = Blueprint('decision_tree_api', __name__, template_folder='templates')
@decision_tree_api.route('/decisiontree', methods=["GET"])
def GetDataFromCSVAndConvertAndProcessToJSON():
    df = cache['uploadedFile']
    df.describe()
    print(df)
    X = df.drop('Target', axis=1)  # axis=1 means column and axis 0 is row
    Y = df['Target']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    from sklearn.tree import DecisionTreeClassifier
    tree_model = DecisionTreeClassifier()
    tree_model = tree_model.fit(x_train, y_train)
    y_predict = tree_model.predict(x_test)
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
    x_test["Source IP"] = x_test["Source (one)"] + '.' + x_test["Source (two)"] + '.' + x_test["Source (three)"] + '.' + x_test["Source (four)"]
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
        nodeDetailsType = NodeDetailsType(df['Dest_id'][ind],df['Destination IP'][ind],df['Score'][ind])
        vsData = VisualizationData(nodeDetailsType,'nodes',True,True)
        nodeDetailsTypeSource = NodeDetailsType(df['Source_id'][ind], df['Source IP'][ind], df['Score'][ind])
        vsDataSource = VisualizationData(nodeDetailsTypeSource, 'nodes', True, True)
        my_list.append(vsData)
        my_list.append(vsDataSource)
        edge = Edge(df['Source_id'][ind], df['Dest_id'][ind], df['weight'][ind])
        edgeData = EdgeData(edge)
        my_list.append(edgeData)


    visualizationJSONData = json.dumps(my_list, indent=4, cls=VisualizationDataEncoder)
    return visualizationJSONData


class NodeDetailsType:
    def __init__(self, id, name, score):
        self.id = id
        self.name = name
        self.score = score
        self.vulnerable = True if score >= 0.5 else False
        self.clean = True if score < 0.3 else False
        self.mildVulnerable = True if score >= 0.3 and score < 0.5 else False

class VisualizationData:
    def __init__(self, NodeDetailsType, group, selectable,grabbable):
        self.data = NodeDetailsType
        self.group = group
        self.selectable = selectable
        self.grabbable = grabbable

class EdgeData:
    def __init__(self,edge):
        self.data = edge

class Edge:
    def __init__(self, source, target, weight):
        self.source = source
        self.target = target
        self.weight = weight
        self.group = 'clean' if weight <= 0.3 else 'mild-vulnerable' if weight > 0.3 and weight <= 0.7 else 'vulnerable'
        self.id = source+'edge'

class VisualizationDataEncoder(JSONEncoder): #json encoder to convert collection to JSON
        def default(self, o):
            return o.__dict__



