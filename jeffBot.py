import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

arquivo = r"caminho"
df = pd.read_csv(arquivo)

le = LabelEncoder()
df["categoria"] = le.fit_transform(df["categoria"])
x = df[["valor", "categoria", "dia_da_semana", "recorrente"]]
y = df["essencial"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = MLPClassifier(
    hidden_layer_sizes=(32, 16, 8),
    activation='relu',
    solver='adam',
    max_iter=2000,
    early_stopping=True,
    verbose=True,
    random_state=1
)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("\n relatorio de classificaçao:")
print(classification_report(y_test, y_pred))
print("\n matriz de confusao:")
print(confusion_matrix(y_test, y_pred))

novo_gasto = pd.DataFrame(
    [[80.0, le.transform(["Lazer"])[0], 5, 0]],
    columns=["valor", "categoria", "dia_da_semana", "recorrente"]
)
novo_gasto2 = pd.DataFrame(
    [[80.0, le.transform(["Combustível"])[0], 5, 1]],
    columns=["valor", "categoria", "dia_da_semana", "recorrente"]
)

pred = model.predict(novo_gasto)
pred2 = model.predict(novo_gasto2)

print("\n a o jeffbot acha que o gasto é: ", "essencial" if pred[0] == 1 else "não essencial")
print("\n a o jeffbot acha que o gasto é: ", "essencial" if pred2[0] == 1 else "não essencial")

joblib.dump(model, "jeffbot_alpha1.pkl")
