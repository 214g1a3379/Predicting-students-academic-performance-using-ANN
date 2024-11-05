import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Reading data
df = pd.read_csv("Final_Training_Data.csv")
print("Reading Training Data")
print(df.columns)

# Splitting data file into train and test data files
train, test = train_test_split(df, test_size=0.2)

# One-hot encoding on train data
columns_to_encode = ['gender', 'residence', 'location']

# Apply one-hot encoding
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(train[columns_to_encode])

# Convert the encoded data to a DataFrame
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns_to_encode))

# Reset the index of both dataframes before concatenating
train.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the original data with one-hot encoded parameters
df_encoded = pd.concat([train.drop(columns=columns_to_encode), encoded_df], axis=1)
print("\nColumns after encoding on train data")
print(df_encoded.columns)


#separating Feature Labels and Target Labels in train data
X_train = df_encoded.drop('TARGET_PREDICTION_PERCENT',axis=1)
print("\nseparating Feature Labels in train data")
print(X_train.columns)

y_train =df_encoded['TARGET_PREDICTION_PERCENT']
print("\nSeparating Target Labels in train data")
print(y_train.name)

#fit & transform training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("\nfit & transformed train data")
print(X_train_scaled[:2])

#Building model
# Set the input shape
input_shape = (25,)
print(f'Feature shape: {input_shape}')

# Create the model
model = Sequential()
model.add(Dense(25, input_shape=input_shape, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error',tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)

# One-hot encoding on test data
columns_to_encode = ['gender', 'residence', 'location']

# Apply one-hot encoding
encoder = OneHotEncoder()
encoded_test_data = encoder.fit_transform(test[columns_to_encode])

# Convert the encoded data to a DataFrame
encoded_test_df = pd.DataFrame(encoded_test_data.toarray(), columns=encoder.get_feature_names_out(columns_to_encode))

# Reset the index of both dataframes before concatenating
test.reset_index(drop=True, inplace=True)
encoded_test_df.reset_index(drop=True, inplace=True)

# Concatenate the original data with one-hot encoded parameters
test_df_encoded = pd.concat([test.drop(columns=columns_to_encode), encoded_test_df], axis=1)
print("\nColumns after encoding on test data")
print(test_df_encoded.columns)

#separating Feature Labels and Target Labels in test data
X_test = test_df_encoded.drop('TARGET_PREDICTION_PERCENT',axis=1)
y_test =test_df_encoded['TARGET_PREDICTION_PERCENT']


#fit & transform training data
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
print("\nfit & transformed test data")
print(X_test_scaled[:2])


# Prediction on test data
predictions = model.predict(X_test_scaled)

predictions_df = pd.DataFrame(predictions, columns=['Predicted_Target'])

predictions_df.to_csv('test predictions.csv', index=False)




