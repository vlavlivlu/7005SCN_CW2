import pandas as pd
import string

def preprocessing_text(df, text_column):
    # Remove missing values
    df = df.dropna(subset=[text_column])
    # Remove duplicate values
    df = df.drop_duplicates()
    # Convert to lowercase
    df[text_column] =  df[text_column].str.lower()
    # Remove URL
    df[text_column] = df[text_column].str.replace(r"http\S+|www\S+", "", regex=True)
    # Remove HTML tags
    df[text_column] = df[text_column].str.replace(r"<.*?>", "", regex=True)
    # Remove punctuation 
    translator = str.maketrans('', '', string.punctuation)
    df[text_column] = df[text_column].str.translate(translator)
    # Remove extra whitespace 
    df[text_column] = df[text_column].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    print("Data Preprocessing Completed!")
    return df

