from flask import Flask,render_template,request,redirect,url_for,session
import mysql.connector
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
db=mysql.connector.connect(host='localhost', user='root',password='',port=3306,database='obiseity')
cur=db.cursor()
app=Flask(__name__)
app.config['uploadfiles']=r"uploads"


@app.route("/")
def base():
    return render_template("index.html")


@app.route('/Register',methods=['POST','GET'])
def Register():
    if request.method=="POST":
        pass
        Name=request.form['Name']
        Email=request.form['Email']
        Contact=request.form['Contact']
        Age=request.form['Age']
        Address=request.form['Address']
        Password=request.form['Password']
        confirmpassword=request.form['confirmpassword']
        if Password==confirmpassword:
            sql="select * from obiseityusers where (Name='%s' and Password='%s') or (Email='%s' and Password='%s')"%(Name,Password,Email,Password)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            if data==[]:
                sql="insert into obiseityusers(Name,Email,Contact,Age,Address,Password)values(%s,%s,%s,%s,%s,%s)"
                val=(Name,Email,Contact,Age,Address,Password)
                cur.execute(sql,val)
                db.commit()
                msg="Data Registered successfuly ......"
                return render_template("Register.html",msg=msg)
            else:
                msg = "Details already registerd !!!!!!!"
                return render_template("Register.html", msg=msg)

    return render_template("Register.html")


@app.route('/login',methods=["POST","GET"])
def login():
    if request.method=="POST":
        Email=request.form['Email']
        password=request.form['password']
        sql="select * from obiseityusers where Email=%s and Password=%s"
        val=(Email,password)
        cur.execute(sql,val)
        data=cur.fetchall()
        db.commit()
        print(data)
        if data!=[]:
            return render_template("userhome.html")
        else:
            msg="details are not valid"
            return render_template("login.html",msg=msg)

    return render_template("login.html")


@app.route('/uploadfile',methods=["POST","GET"])
def uploadfile():
    if request.method=="POST":
        file=request.files['files']
        print(file)
        print(file.filename)
        file_name=file.filename
        store=os.path.join(app.config['uploadfiles'],file_name)
        file.save(store)
        msg="file uploaded successfuly"
        return render_template('uploadfile.html',msg=msg)
    return render_template('uploadfile.html')

@app.route("/viewdata")
def viewdata():
    file=os.listdir(app.config['uploadfiles'])
    path = os.path.join(app.config['uploadfiles'], file[0])
    df = pd.read_csv(path)
    return render_template("viewdata.html",cols=df.columns.values,rows=df.values.tolist())





@app.route('/preprocessing')
def preprocessing():
    global fixed_data
    labelencoder = LabelEncoder()
    file = os.listdir(app.config['uploadfiles'])
    path = os.path.join(app.config['uploadfiles'], file[0])
    df = pd.read_csv(path)
    print("======")
    Q1 = df['Age'].quantile(0.25)
    Q3 = df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR)))]
    print("//////////")
    original_data = df[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']]
    labeldata = df[['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']]
    print(original_data)
    print(labeldata)
    labeldata["Gender"] = labeldata["Gender"].apply(lambda x: 1 if x == "Female" else 0)
    labeldata["family_history_with_overweight"] = labeldata["family_history_with_overweight"].apply(lambda x: 1 if x == "yes" else 0)
    labeldata["FAVC"] = labeldata["FAVC"].apply(lambda x: 1 if x == "yes" else 0)
    labeldata["SMOKE"] = labeldata["SMOKE"].apply(lambda x: 1 if x == "yes" else 0)
    labeldata["SCC"] = labeldata["SCC"].apply(lambda x: 1 if x == "yes" else 0)
    fixed_data = pd.concat([original_data, labeldata], axis=1)
    fixed_data['CAEC'] = labelencoder.fit_transform(fixed_data['CAEC'])
    fixed_data['CALC'] = labelencoder.fit_transform(fixed_data['CALC'])
    fixed_data['MTRANS'] = labelencoder.fit_transform(fixed_data['MTRANS'])
    fixed_data['NObeyesdad'] = labelencoder.fit_transform(fixed_data['NObeyesdad'])
    return render_template('preprocessing.html',cols=fixed_data.columns.values,rows=fixed_data.values.tolist())

@app.route('/splitdata',methods=['POST','GET'])
def splitdata():
    global X_train,X_test, y_train, y_test
    print("=========")
    X = fixed_data.drop(columns=['NObeyesdad'])
    y = fixed_data[['NObeyesdad']]
    if request.method=='POST':
        testsize=request.form['testsize']
        val=int(testsize)/100
        m=val
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=m, random_state=42)
        return render_template("modelselection.html")
    return render_template('splitdata.html')

@app.route("/modelselection",methods=['POST','GET'])
def modelselection():
    if request.method=='POST':
        modelname=request.form['modelname']
        if modelname=='1':
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            lraccuracy = accuracy_score(y_test, y_pred)
            msg=" Logistic Regression Accuracy {}".format(lraccuracy)
            return render_template("modelselection.html",msg=msg)
        elif modelname=='2':
            DT = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
            DT.fit(X_train, y_train)
            y_pred = DT.predict(X_test)
            DTaccuracy = accuracy_score(y_test, y_pred)
            msg = " Decision Tree Accuracy {}".format(DTaccuracy)
            return render_template("modelselection.html", msg=msg)
        elif modelname=='3':
            RF = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
            RF.fit(X_train, y_train)
            y_pred = RF.predict(X_test)
            RFaccuracy = accuracy_score(y_test, y_pred)
            msg = " Random Forest Accuracy {}".format(RFaccuracy)
            return render_template("modelselection.html", msg=msg)
        elif modelname=='4':
            EF = ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
            EF.fit(X_train, y_train)
            y_pred = EF.predict(X_test)
            EFaccuracy = accuracy_score(y_test, y_pred)
            msg = " Extra tree Accuracy {}".format(EFaccuracy)
            return render_template("modelselection.html", msg=msg)
        elif modelname=='5':
            AB = AdaBoostClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
            AB.fit(X_train, y_train)
            y_pred = AB.predict(X_test)
            ABaccuracy = accuracy_score(y_test, y_pred)
            msg = " Extra tree Accuracy {}".format(ABaccuracy)
            return render_template("modelselection.html", msg=msg)
        else:
            msg="Model is model is not selected properly"
            return render_template("modelselection.html", msg=msg)
    return render_template("modelselection.html")


@app.route("/Prediction",methods=['POST','GET'])
def prediction():
    list1=[]
    if request.method=='POST':
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        fcvc = float(request.form['fcvc'])
        ncp = float(request.form['ncp'])
        ch2o = float(request.form['ch2o'])
        faf = float(request.form['faf'])
        tue = float(request.form['tue'])
        Gender=int(request.form['Gender'])
        overweight=int(request.form['overweight'])
        favc = int(request.form['favc'])
        smoke = int(request.form['smoke'])
        SCC = int(request.form['SCC'])
        CAEC=int(request.form['caec'])
        CALC=int(request.form['CALC'])
        mtrans=int(request.form['mtrans'])
        m=[age,height,weight,fcvc,ncp,ch2o,faf,tue,Gender,overweight,favc,smoke,SCC,CAEC,CALC,mtrans]
        print(m)
        RF = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0)
        RF.fit(X_train, y_train)
        predi = RF.predict([m])
        pred = predi
        print("Final Output:",pred[0])
        if pred[0]=='0': msg="Insufitient Weight"
        elif pred[0]=='1': msg="Normal Weight"
        elif pred[0]=='2': msg="Obesity Type - 1"
        elif pred[0]=='3': msg="Obesity Type - 2"
        elif pred[0]=='4': msg="Obesity Type - 3"
        elif pred[0]=='5': msg='OverWeight Level-1'
        else:msg="OverWeight Level - 2"
        return render_template("prediction.html",msg=msg)
    return render_template("prediction.html")

@app.route("/logout")
def logout():
    return redirect(url_for('base'))
if __name__=="__main__":
    app.run(debug=True)