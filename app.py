import streamlit as st
st.title("Expected CTC Prediction App")
st.write("Upload your dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.head())

    st.subheader("Data Info")
    st.write(data.describe())
    st.write("Missing values:")
    st.write(data.isnull().sum())

    data = data.dropna()
    cat_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    if 'Expected_CTC' in data_encoded.columns:
        X = data_encoded.drop('Expected_CTC', axis=1)
        y = data_encoded['Expected_CTC']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        st.subheader("Model Performance")
        st.write(f"R² Score: {model.score(X_test, y_test):.2f}")
    else:
        st.warning("Column 'Expected_CTC' not found in your data.")
