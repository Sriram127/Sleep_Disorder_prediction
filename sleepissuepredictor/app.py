import streamlit as st
import numpy as np
import pandas as pd
import time
from modeling import load_model 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Loading issue model with scaling
issue_model = load_model(r"NoSystolic_ScaledModelSVC.pkl")
# Loading type of issue model with scaling
issue_type_model = load_model(r"SleepIssueType_ModelScaled.pkl")

def proba_predict(model, input_data):
    """
    Extracts the probability of having any sleep issue
    """ 
    prediction = model.predict_proba(input_data)
    return prediction[0][1]

def plot_filled_gender(male, percentage):

    fig, ax = plt.subplots(figsize=(1,2))
    fig.patch.set_alpha(0)

    img_path = (r"male_silhouette.png") if (male==1) else (r"female_silhouette.png")

    # Load image
    img = mpimg.imread(img_path)

    # Convert the entire image to gray while preserving the alpha (transparency) channel
    grayscale_img_with_alpha = np.zeros_like(img)
    grayscale_img_with_alpha[:, :, :3] = [0.05098039, 0.07058824, 0.50980392] # issue color
    grayscale_img_with_alpha[:, :, 3] = img[:, :, 3]

    # Calculate the y limit based on percentage
    ylim = int(img.shape[0] * (1 - (percentage / 100)))

    # Replace the grayscale values with the original values up to the y-limit
    grayscale_img_with_alpha[:ylim, :, :3] = [0,0,0]

    # Display the modified image
    ax.imshow(grayscale_img_with_alpha, interpolation='none')

    # Add the percentage text
    ax.text(0.5, 0.5, f'{percentage}%', ha='center', va='center', color='white', 
            fontweight='bold', fontsize=10, transform=ax.transAxes)

    # Remove axis - 
    ax.axis('off')
    
    return fig

def sleep_issue_image(issue, TITLE):

    fig, ax = plt.subplots(figsize=(2,2))
    fig.patch.set_alpha(0)

    img_path = (r"sleep_apnea.jpeg") if (issue=='Sleep Apnea') else (r"insomnia.jpeg")

    # Load image
    img = mpimg.imread(img_path)

    # Display the modified image
    ax.imshow(img, interpolation='none')

    # Remove axis
    ax.axis('off')

    # Comment on the issue for the user
    ax.set_title(TITLE, fontsize=8, color="#FFFFFF")
    
    return fig

# Set title
st.title("Sleep Issue Predictor💤")


data_tab, result_tab, pre,instr_tab = st.tabs(["Set readings", "Results", "Precautions","Instructions"])

with data_tab:

    # Creating column components for the plot
    _col1, _col2 = st.columns([.5,.5])
    
    with _col1:
        # User inputs
        age = st.number_input("Enter your age", value=31, min_value=18, max_value=60)
        sleep_duration = st.number_input("Enter sleep duration", value=8.0)
        heart_rate = st.number_input("Enter heart rate", value=70, min_value=60)
        daily_steps = st.number_input("Enter daily steps", value=8000)
    
    with _col2:
        is_male = st.selectbox("Select your gender", ["Male", "Female"])
        wf_technical = st.selectbox("Do you work in a technical or numeric field?", ["Yes", "No"])#such as: accounting, sofware, engineering, scientist...
        
        # BMI Calculation
        # elevated_bmi = st.selectbox("Is your BMI elevated?", ["Yes", "No"])
        weight = st.number_input("What's your weight in Kg (kilograms)?", value=70.0)
        height = st.number_input("What's your height in M (meters)?", value=1.60)


    BMI = weight/np.square(height)
     
    elevated_bmi = 1 if BMI >= 25 else 0
    # Bounds Reference: https://www.thecalculatorsite.com/articles/health/bmi-formula-for-bmi-calculations.php

    # Convert categorical data to numeric
    is_male = 1 if is_male == "Male" else 0
    wf_technical = 1 if wf_technical == "Yes" else 0


    # Predict button
    if st.button("Predict"):

        # Getting input data
        input_data = np.array([[int(age), sleep_duration, heart_rate, int(daily_steps), is_male, elevated_bmi, wf_technical]])
        
        # Convert input_data to a DataFrame with appropriate column names
        columns = ['age', 'sleep_duration', 'heart_rate', 'daily_steps', 'is_male', 'elevated_bmi', 'wf_technical']
        input_df = pd.DataFrame(input_data, columns=columns)   

        print(input_df)

        # Predicting ussing the model that already has scaling implemented in the pipeline
        issue_prob = proba_predict(issue_model, input_df)*100

        print(issue_prob)

        with result_tab:
            with st.spinner('Wait for it...'):
                time.sleep(5)
                st.write(f"### Probability of having a sleep issue:")
                # Creating figure
                fig = plot_filled_gender(is_male, np.round(issue_prob,2))

                # -------
                # This value will control the layout and will give the user extra information 
                # on the sleep condition it suffers ussing the logistic model
                # -------
                cut_off = 50

                # Creating column components for the plot
                col1, col2, col3 = st.columns([.2,.8,.1])
                # Show plot in the middle of the app
                with col2:
                    st.pyplot(fig, use_container_width=False)
                    #sleep_apnea=0 vs insomnia=1
                    global issue_prediction
                    issue_prediction = issue_type_model.predict(input_df)[0]
                    print(issue_prediction)
                    print(issue_type_model.predict_proba(input_df))
                    global se
                    issue_prediction = "Sleep Apnea" if (issue_prediction == 1) else "Insomnia"
                    se=issue_prediction
                    if (issue_prob >= cut_off):
                        #st.write(f"It is likely that your sleep issue is {issue_prediction}")

                        title = f"You may have {issue_prediction}"

                        st.pyplot( sleep_issue_image(issue_prediction, title), use_container_width=False )   
        with pre:
            title = f"Precautions to avoid {se}"
            st.pyplot( sleep_issue_image(issue_prediction, title), use_container_width=False ) 
            if (se=="Insomnia"):
                st.write("1.Stick to a Sleep Schedule: Go to bed and wake up at the same time every day, even on weekends.")
                st.write("2.Create a Relaxing Bedtime Routine: Do something calming before bed, like reading or taking a warm bath.")
                st.write("3.Limit Naps: Try not to nap during the day to make sure you're tired at bedtime.")
                st.write("4.Avoid Stimulants: Cut down on caffeine and nicotine, especially in the afternoon and evening.")
                st.write("5.Don't Eat or Drink Too Much Before Bed: Avoid large meals and too many fluids late in the evening.")
                st.write("6.Exercise Regularly: Physical activity can help you sleep better, but try to finish exercising at least a few hours before bedtime.")
                st.write("7.Make Your Sleep Environment Comfortable: Keep your bedroom dark, quiet, and cool. Use a comfortable mattress and pillows.")
                st.write("8.Limit Screen Time: Avoid screens from phones, tablets, and computers at least an hour before bed.")
                st.write("9.Manage Stress: Practice relaxation techniques like deep breathing, meditation, or gentle yoga before bed.")
                st.write("10.Avoid Clock-Watching: Turn your clock around so you don't watch the time if you wake up during the night.")
                st.markdown("here is a youtube video for reference ", unsafe_allow_html=True)
                st.markdown("https://youtu.be/fXUbGooyG2E?si=bBQeLaOXaQIRJM0S")
            else:
                st.write("1.Maintain a Healthy Weight: Extra weight can worsen sleep apnea. Losing weight can help reduce symptoms.")
                st.write("2.Sleep on Your Side: Sleeping on your back can make sleep apnea worse. Try sleeping on your side instead.")
                st.write("3.Avoid Alcohol and Smoking: Alcohol can relax the muscles in your throat, making sleep apnea worse. Smoking can also worsen symptoms.")
                st.write("4.Keep Your Nasal Passages Open: Use nasal sprays or allergy medicines to keep your nasal passages open at night.")
                st.write("5.Use a CPAP Machine: If prescribed by your doctor, a Continuous Positive Airway Pressure (CPAP) machine can help keep your airway open while you sleep.")
                st.write("6.Create a Regular Sleep Schedule: Going to bed and waking up at the same time every day can help improve your sleep quality.")
                st.write("7.Avoid Heavy Meals and Caffeine Before Bed: Eating large meals or consuming caffeine late in the day can affect your sleep.")
                st.write("8.Exercise Regularly: Regular physical activity can help improve your sleep and reduce sleep apnea symptoms.")
                st.write("9.Elevate Your Head: Sleeping with your head slightly elevated can help keep your airway open.")
                st.write("10.Stay Hydrated: Drink plenty of water throughout the day to prevent throat dryness, which can contribute to sleep apnea.")
                st.markdown("here is a youtube video for reference ", unsafe_allow_html=True)
                st.markdown("https://youtu.be/4b9MM6Z_QnE?si=NJx3IG3XFeEYLB1E")  
    else:
        with result_tab:
            st.markdown('Set your readings in the "Set readings" tab and hit predict to get your prediction')
        
with instr_tab:
    st.markdown("""
                Set your readings in the "Set readings" tab, click the predict button and you'll see your result in the "Results" tab.
                
                Your personal data won't be saved or used for any purpose, this is app is purely recreational.

                The predictions are obtained by means of a Machine Learning algorithm trained using publicly available data at [Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset?datasetId=3321433).

                """)
        # Bringing image
        #fig, ax = plt.subplots(figsize=(1,2))
        #ax.imshow(grayscale_img_with_alpha, interpolation='none')