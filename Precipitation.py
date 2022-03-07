def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(6, input_dim=8,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
                                                           
    regressor.add(Dense(12,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))    
    regressor.add(Dense(24,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    regressor.add(Dense(48,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    regressor.add(Dense(96,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    regressor.add(Dense(96,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    regressor.add(Dense(48,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    regressor.add(Dense(24,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    regressor.add(Dense(12,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))
    
    regressor.add(Dense(6,kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',activation='relu'))             
                     
    
    regressor.add(Dense(1,                                                                    activation='linear'))
    regressor.compile(optimizer=Adam(lr=0.0007),loss='mape',metrics=['mape'])     
    return regressor
