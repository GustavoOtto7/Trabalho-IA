from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from PreProcessamento import X_train, X_test, y_train, y_test

# Converter os rótulos para one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=4)
y_test_encoded = to_categorical(y_test, num_classes=4)

# Criar o modelo de rede neural
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0008), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train_encoded, epochs=200, batch_size=32, validation_data=(X_test, y_test_encoded))

# Avaliar o modelo
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)  # Converter as probabilidades para a classe com maior probabilidade
model.save('modelo_sono.h5')

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Acurácia do modelo: {accuracy:.2f}")