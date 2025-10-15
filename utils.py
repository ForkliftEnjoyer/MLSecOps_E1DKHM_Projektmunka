def extract_features(df,column_name):
    # Remove leading/trailing spaces and convert to string
    df[f'{column_name}_clean'] = df[f'{column_name}'].astype(str).str.strip()

    # Extract numeric part of the ticket
    df[f'{column_name}'] = df[f'{column_name}_clean'].apply(lambda x: ''.join([c for c in x if c.isdigit()]) if any(c.isdigit() for c in x) else '0')
    df[f'{column_name}'] = df[f'{column_name}'].astype(int)

    # Extract ticket length
    df[f'{column_name}_length'] = df[f'{column_name}_clean'].apply(lambda x: len(x))

    # Drop the temporary cleaned column
    df.drop(columns=[f'{column_name}_clean'], inplace=True)

    return df