import streamlit as st
import pandas as pd
import pickle as pkl

# Load the trained model
model = pkl.load(open('model.pkl', 'rb'))

# Load the column names used for training the model
column_names = pkl.load(open('column_names.pkl', 'rb'))

# Define the function to preprocess user inputs
def preprocess_input(df):
    # Perform the necessary preprocessing steps on the input data
    df['joining_day'] = pd.to_datetime(df['joining_date'], format="%Y/%m/%d").dt.day
    df['joining_month'] = pd.to_datetime(df['joining_date'], format="%Y/%m/%d").dt.month
    df['joining_year'] = pd.to_datetime(df['joining_date'], format="%Y/%m/%d").dt.year
    df.drop(['joining_date', 'last_visit_time'], axis=1, inplace=True)

    # Drop the 'referral_id' column if it exists
    if 'referral_id' in df.columns:
        df.drop('referral_id', axis=1, inplace=True)

    # Perform one-hot encoding for categorical features
    categorical_cols = ['gender', 'region_category', 'membership_category', 'joined_through_referral',
                        'preferred_offer_types', 'medium_of_operation', 'internet_option',
                        'used_special_discount', 'offer_application_preference', 'past_complaint', 'feedback']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Reorder the columns to match the training data columns
    df_processed = df_encoded.reindex(columns=column_names[:-1], fill_value=0)

    return df_processed

# Define the function to predict churn risk scores
def predict_churn_risk(df):
    # Preprocess the input data
    df_processed = preprocess_input(df)

    # Make predictions using the loaded model
    predictions = model.predict(df_processed)

    return predictions

# Create the Streamlit app
def main():
    st.markdown(
    """
    <style>
    .title-text {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown('<h1 class="title-text">Churn Risk Score Prediction App</h1>', unsafe_allow_html=True)

    # Collect user input
    age = st.number_input("Age:")
    gender = st.selectbox("Gender:", ['Male', 'Female'])
    region_category = st.selectbox("Region Category:", ['Village', 'City', 'Town'])
    membership_category = st.selectbox("Membership Category:", ['Basic', 'Gold', 'Silver', 'Platinum', 'Premium'])
    joined_through_referral = st.selectbox("Joined Through Referral:", ['No', 'Yes'])
    preferred_offer_types = st.selectbox("Preferred Offer Types:", ['Gift Vouchers/Coupons', 'Credit/Debit Card Offers', 'Without Offers'])
    medium_of_operation = st.selectbox("Medium of Operation:", ['Desktop', 'Smartphone', 'Both'])
    internet_option = st.selectbox("Internet Option:", ['Wi-Fi', 'Mobile_Data'])
    avg_frequency_login_days = st.text_input("Average Frequency Login Days:")
    points_in_wallet = st.number_input("Points in Wallet:")
    used_special_discount = st.selectbox("Used Special Discount:", ['No', 'Yes'])
    offer_application_preference = st.selectbox("Offer Application Preference:", ['No', 'Yes'])
    past_complaint = st.selectbox("Past Complaint:", ['No', 'Yes'])
    feedback = st.selectbox("Feedback:", ['Products always in Stock', 'Quality Customer Care', 'No reason specified', 'Poor Website', 'Poor Product Quality'])
    last_visit_time = st.text_input("Last Visit Time (HH-MM-SS):")
    joining_date = st.text_input("Joining Date (YYYY/MM/DD) :")

    # Create a dictionary with the user input
    user_input = {
        'age': age,
        'gender': gender,
        'region_category': region_category,
        'membership_category': membership_category,
        'joined_through_referral': joined_through_referral,
        'preferred_offer_types': preferred_offer_types,
        'medium_of_operation': medium_of_operation,
        'internet_option': internet_option,
        'avg_frequency_login_days': avg_frequency_login_days,
        'points_in_wallet': points_in_wallet,
        'used_special_discount': used_special_discount,
        'offer_application_preference': offer_application_preference,
        'past_complaint': past_complaint,
        'feedback': feedback,
        'joining_date': joining_date,
        'last_visit_time': last_visit_time
    }

    # Convert the dictionary into a DataFrame
    input_df = pd.DataFrame(user_input, index=[0])

    # Predict churn risk score
    col1, col2, col3 = st.columns(3)

    with col1:
        pass

    with col2:
        if st.button("Predict Churn Risk Score"):
            predictions = predict_churn_risk(input_df)
            churn_risk_score = predictions[0]
            st.success(f"The predicted churn risk score is: {churn_risk_score}")

    with col3:
        pass

# Run the app
if __name__ == '__main__':
    main()
