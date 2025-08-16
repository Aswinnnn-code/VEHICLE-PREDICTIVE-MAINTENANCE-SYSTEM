import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import os

save_path = r"C:/placement/AI AUTO" 

df = pd.read_csv("C:/placement/AI AUTO/vehicle_data.csv")
x = df.drop("needs_maintenance", axis=1)
y = df["needs_maintenance"]



scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
scaler_file = os.path.join(save_path, "scaler.pkl")
joblib.dump(scaler, scaler_file)
print("Scaler saved successfully!")


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size = 0.2,random_state = 42)




model = XGBClassifier(use_label_encoder = False , eval_metric = 'logloss')
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))


model_file = os.path.join(save_path, "vehicle_maintenance_model.pkl")
joblib.dump(model, model_file)
print("Model saved successfully!")