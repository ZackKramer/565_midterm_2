# App to predict the prices of diamonds using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt 

# Set up the app title and image
st.title(':rainbow[Traffic Volume Predictor]')
st.write("Utilize our advanced Machine Learning application to predict traffic volume")
st.image('traffic_image.gif', use_column_width = True)

alpha = st.slider('Select alpha value for prediction intervals',
          min_value = 0.01,
          max_value = 0.5,
          value = 0.1,
          step = 0.01)

# Reading the pickle file that we created before 
model_pickle = open('xg_ml.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('Traffic_Volume.csv')
default_df['month'] = pd.DatetimeIndex(default_df['date_time']).month_name()
default_df['weekday'] = pd.DatetimeIndex(default_df['date_time']).day_name()
default_df['hour'] = pd.DatetimeIndex(default_df['date_time']).hour
default_df['holiday'] = default_df['holiday'].fillna('None')

# Sidebar for user inputs with an expander
with st.sidebar:
    st.image('traffic_sidebar.jpg', use_column_width = True,
             caption = "Traffic Volume Predictor")
    st.header("Input Features")
    st.write("You can either upload your data file or manually enter input features")
    with st.expander("Option 1: Upload CSV File"):
        st.header("Upload a CSV file containing traffic details.")
        traffic_file = st.file_uploader("Choose a CSV file")
        st.header("Sample Data Format for Upload")
        st.dataframe(default_df.head())
        st.warning("Ensure your uploaded file has the same column names and data types as shown above.", icon="⚠️")
    with st.expander("Option 2: Fill Out Form"):
        with st.form("user_input_form"):
            st.header("Enter the traffic details manually using the form below.")
            holiday = st.selectbox('Choose whether today is a designated holiday or not', 
                                    options = default_df['holiday'].unique())
            temp = st.number_input('Average temperature in Kelvin',
                                    min_value = default_df['temp'].min(),
                                    max_value = default_df['temp'].max(),
                                    value = default_df['temp'].mean(),
                                    step = .01)
            rain_1h = st.number_input('Amount of mm of rain that occurred in the hour',
                                    min_value = default_df['rain_1h'].min(),
                                    max_value = default_df['rain_1h'].max(),
                                    value = default_df['rain_1h'].mean(),
                                    step = .01)
            snow_1h = st.number_input('Amount of mm of snow that occurred in the hour',
                                    min_value = default_df['snow_1h'].min(),
                                    max_value = default_df['snow_1h'].max(),
                                    value = default_df['snow_1h'].mean(),
                                    step = .01)
            clouds_all = st.number_input('Percentage of cloud cover',
                                    min_value = default_df['clouds_all'].min(),
                                    max_value = default_df['clouds_all'].max(),
                                    value = int(default_df['clouds_all'].mean()),
                                    step = 1)
            weather_main = st.selectbox('Choose the current weather', 
                                    options = default_df['weather_main'].unique())
            month = st.selectbox('Choose month', 
                                    options = default_df['month'].unique())
            weekday = st.selectbox('Choose weekday',
                                   options = default_df['weekday'].unique())
            hour = st.selectbox('Choose hour', 
                                    options = default_df['hour'].unique())
            submit_button = st.form_submit_button("Submit Form Data")

if traffic_file is None and not submit_button:
    st.info('Please choose a data input method to proceed', icon="ℹ️")

elif traffic_file is None and submit_button:
    st.success('Form data submitted successfully', icon="✅")

    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume','date_time'])

    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday,
                                    temp,
                                    rain_1h,
                                    snow_1h,
                                    clouds_all,
                                    weather_main,
                                    month,
                                    weekday,
                                    hour]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]

    # Ensure limits are within [0, 10000]
    lower_limit = max(0, lower_limit[0][0])
    upper_limit = min(10000, upper_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{int(pred_value)}")
    st.write(f"**Confidence Interval** ({int(100-alpha*100)}%): [{int(lower_limit)}, {int(upper_limit)}]")

else:
    st.success('CSV file uploaded successfully', icon="✅")

    # Loading data
    user_df = pd.read_csv(traffic_file) # User provided data

    # Dropping null values
    
    user_df['holiday'] = user_df['holiday'].fillna('None')
    user_df = user_df.dropna() 
    default_df = default_df.dropna() 

    # Remove output and datetime columns from original data
    default_df = default_df.drop(columns = ['traffic_volume','date_time'])
    
    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[default_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([default_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = default_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Align columns explicitly to avoid mismatch
    default_encoded_columns = pd.get_dummies(default_df).columns
    combined_df_encoded = combined_df_encoded.reindex(columns=default_encoded_columns, fill_value=0)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Predictions for user data
    user_pred, user_intervals = reg_model.predict(user_df_encoded, alpha = alpha)

    # Predicted traffic
    user_pred_traffic = user_pred
    user_lower_limit = user_intervals[:, 0]
    user_upper_limit = user_intervals[:, 1]

    # Adding predicted traffic to user dataframe
    user_df['Predicted Traffic'] = user_pred_traffic.astype(int)
    user_df['Lower Limit'] = user_lower_limit.astype(int)
    user_df['Upper Limit'] = user_upper_limit.astype(int)
    
    user_df['Lower Limit'] = user_df['Lower Limit'].apply(lambda x: max(0, x))
    user_df['Upper Limit'] = user_df['Upper Limit'].apply(lambda x: min(10000, x))

    # Show the predicted traffic on the app
    st.subheader(f"Prediction Results with Confidence Interval of {int(100-alpha*100)}%")
    st.dataframe(user_df)

# Additional tabs for model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
