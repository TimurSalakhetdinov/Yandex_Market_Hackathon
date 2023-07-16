import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def predict(order):

    items = order['items']
    # Count the number of unique SKUs in the order
    num_skus = len(items)

    # Create a mapping from JSON field names to model field names
    field_mapping = {
        'size1': 'a',
        'size2': 'b',
        'size3': 'c',
        'weight': 'goods_wght'
    }

    # Assume these are the columns the model was trained on, in the correct order
    model_columns = ['counts_sku', 'goods_wght', 'volume_sku', 'P', 'cargo_0', 'cargo_1', 'cargo_2', 'cargo_3',
                     'cargo_4', 'cargo_5', 'cargo_6', 'cargo_7', 'cargo_8', 'cargo_9', 'cargo_10']

    df = pd.DataFrame(items)  # Create a DataFrame with a list of dictionaries

    # Convert strings to floats
    df['size1'] = df['size1'].astype(float)
    df['size2'] = df['size2'].astype(float)
    df['size3'] = df['size3'].astype(float)
    df['weight'] = df['weight'].astype(float)

    # Convert 'type' to integer
    df['type'] = df['type'].apply(lambda x: [float(i) for i in x])

    # Apply transformations and feature engineering as done during model training
    df['volume_sku'] = df['size1'] * df['size2'] * df['size3'] * df['count']
    df['P'] = (df['size1'] + df['size2'] + df['size3']) * df['count']

    # Rename columns in the DataFrame
    for json_field, model_field in field_mapping.items():
        df.rename(columns={json_field: model_field}, inplace=True)

    # Create a dataframe that holds all the model features
    model_df = pd.DataFrame(columns=model_columns)
    model_df.loc[0, 'counts_sku'] = num_skus
    model_df.loc[0, 'goods_wght'] = df['goods_wght'].sum()
    model_df.loc[0, 'volume_sku'] = df['volume_sku'].sum()
    model_df.loc[0, 'P'] = df['P'].sum()

    # Initialize all cargo columns with 0
    model_df.iloc[0, 4:] = 0

    # Fill the cargo columns based on the cargotypes in the data
    cargo_types = [item for sublist in df['type'].to_list() for item in sublist]  # Flatten the list
    unique_cargo_types = list(set(cargo_types)) # Get unique values

    for i in range(len(unique_cargo_types)):
        model_df.iloc[0, i + 4] = unique_cargo_types[i]  # Fill the cargo columns

    # Load the model
    with open(f'model{min(num_skus, 3)}.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(model_df)

    # Load LabelEncoder from a file
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()  # convert numpy array to list if it's an array

    y_pred = le.inverse_transform(y_pred)

    # return the first element if it's a single element list, else return the whole list
    return y_pred[0] if len(y_pred) == 1 else y_pred