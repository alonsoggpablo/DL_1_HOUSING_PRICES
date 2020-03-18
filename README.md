# DL_1_HOUSING_PRICES

Aportamos 2 scripts: uno usando modelos de regresión estándar de ML (Linear Reg, DecisionTre Reg y Random Forest Reg)
y otro usando Keras/TF y una NN con 2 hidden layers:

model = Sequential()

model.add(Dense(500, activation="relu",input_dim=9))

model.add(Dense(100, activation="relu"))

model.add(Dense(50, activation="relu"))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])

model.fit(X_train, Y_train, epochs=20)
