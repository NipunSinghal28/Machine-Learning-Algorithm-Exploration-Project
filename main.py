import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def SimpleLinearReg():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    data = pd.read_csv('homeprices.csv')
    # print(data)
    plt.xlabel('area(sqr ft)')
    plt.ylabel('price(US$)')
    plt.scatter(data.Area, data.Price, color='red', marker='*')
    # plt.show()

    reg = linear_model.LinearRegression()
    x = data[['Area']]
    y = data['Price']
    reg.fit(x, y)  # training the model using available points
    z = reg.predict([[3300]])
    U = reg.predict([[5000]])
    '''
    print('----------')
    print('prediction for 3300 using ML model')
    print(z)
    print('----------')
    print('----------')
    print('prediction for 5000 using ML model')
    print(U)
    print('----------')
    print('-\-\-\-\-')
    '''

    print('slope/gradient/coefficient=', reg.coef_)
    print('intercept=', reg.intercept_)

    # y=m*x+b
    cal1 = 135.78767123 * 3300 + 180616.43835616432
    # print(cal1)
    cal2 = 135.78767123 * 5000 + 180616.43835616432
    # print(cal2)

    d = pd.read_csv('Areas.csv')
    print(d.head(3))
    t = reg.predict(d)
    d['prices'] = t
    print(d)
#    d.to_csv('prediction.csv', index=False)

    plt.xlabel('area(sqr ft)')
    plt.ylabel('price(US$)')
    plt.scatter(data.Area, data.Price, color='red', marker='*')
    plt.plot(data.Area, reg.predict(data[['Area']]), color='blue')
    plt.show()

def PolyLinearReg():
    from sklearn import linear_model
    import matplotlib.pyplot as plt

    data = pd.read_csv('Height_Weight_Dataset.csv')
    print(data.head())

    # store data in dependent and independent variables separately
    x = data.iloc[:, 0:1].values  # age
    y = data.iloc[:, 1].values  # height

    # split into training and test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # fitting simple linear regression
    from sklearn.linear_model import LinearRegression
    Linreg = LinearRegression()
    Linreg.fit(x_train, y_train)
    '''
    # visualize linear regression
    plt.scatter(x_train,y_train,color='green')
    plt.plot(x_train, Linreg.predict(x_train), color='blue')
    plt.title('Linear Regression')
    plt.xlabel('Age')
    plt.ylabel('Height')
    #plt.show()
    '''
    # add polynomial to eqn
    from sklearn.preprocessing import PolynomialFeatures
    polyom = PolynomialFeatures(degree=2)
    x_polyom = polyom.fit_transform(x_train)
    print(x_polyom)

    # fit polynomial regression model
    Polyreg = LinearRegression()
    Polyreg.fit(x_polyom, y_train)

    # visualize the polynomial regression
    plt.scatter(x_train, y_train, color='green')
    plt.plot(x_train, Polyreg.predict(polyom.fit_transform(x_train)), color='blue')
    plt.title('Polynomial regression')
    plt.xlabel('Age')
    plt.ylabel('Height')
    plt.show()

def SupportVectorMachine():
    from sklearn.datasets import load_iris
    iris = load_iris()
#    print(iris.feature_names)
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
#   print(data[data.target==1].head())
    data['flower_name'] = data.target.apply(lambda x: iris.target_names[x])
#    print(data.head())

    data0 = data[data.target == 0]
    data1 = data[data.target == 1]
    data2 = data[data.target == 2]

#    print('hello\n', data0, '\nhello')
#    print(data1)
#    print(data2)

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.scatter(data0['petal length (cm)'], data0['petal width (cm)'], color='green', marker='*')
    plt.scatter(data1['petal length (cm)'], data1['petal width (cm)'], color='blue', marker='.')
    plt.show()
    X=data.drop(['target','flower_name'], axis = 'columns')
#    print(X.head())
def DecisionTree():
    data = pd.read_csv('salaries.csv')
#    print(data.head())
    inputs = data.drop('salary_more_then_100k', axis='columns')
#    print(inputs)
    target = data['salary_more_then_100k']
#    print(target)
    from sklearn.preprocessing import LabelEncoder
    le_company = LabelEncoder()
    le_job = LabelEncoder()
    le_degree = LabelEncoder()

    inputs['company_n'] = le_company.fit_transform(inputs['company'])
    inputs['job_n']=le_job.fit_transform(inputs['job'])
    inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
#    print(inputs)
    inputs_n = inputs.drop(['company','job','degree'], axis= 'columns')

#    print(inputs_n)
    from sklearn import tree
    model = tree.DecisionTreeClassifier()

    model.fit(inputs_n,target)
    print(model.score(inputs_n,target))
    prediction1 = model.predict([[2,1,0]]) # google, bachelor degree, salary above 100k
    print('prediction that a GOOGLE employee with BACHELORS degree will have SALARY ABOVE 100K:', prediction1,'(meaning salary is not above 100k)')
    prediction2 = model.predict([[2, 1, 1]])  # google, masters degree, salary above 100k
    print('prediction that a GOOGLE employee with MASTERS degree will have SALARY ABOVE 100K:',prediction2,'(meaning salary is above 100k)')

def RandomForest():
    from sklearn.datasets import load_digits
    digits = load_digits()
#    print(digits.data[0])
    plt.gray()
    for i in range(4):
        plt.matshow(digits.images[i])
#        plt.show()

#    print((digits.target[:4]))
    data= pd.DataFrame(digits.data)
#    print(data.head())

    data['target']= digits.target
#    print(data.head())

    x= data.drop('target', axis='columns')
#    print(x)
    y=data.target
#    print(y)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)
#    print(len(x_train))
#    print(len(y_test))
#    print(len(x))

    from sklearn.ensemble import  RandomForestClassifier
    model= RandomForestClassifier()
    model.fit(x_train,y_train)

    predict= model.score(x_test, y_test)
    print('accuracy %age = ',predict*100)

    y_predicted= model.predict(x_test)
#    print(y_predicted)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,y_predicted)
#    print(cm)

    import seaborn as sb
    plt.figure(figsize=(10,7))
    sb.heatmap(cm, annot= True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()




def main():
#    SimpleLinearReg()
#    PolyLinearReg()
#    SupportVectorMachine()
#    DecisionTree()
#    RandomForest()

main()