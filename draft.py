import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")

# Contribution data
data = {
    "Name": ["Saloni Jain", "Sneha Pal", "Sooriya Senthilkumar", "Aneesh Sabarad", "Vibha Guru"],
    "Contribution": [
        "Worked on M1 Data Cleaning",
        "Worked on M1 Coding",
        "Worked on M1 Feature Reduction",
        "Worked on Results Evaluation",
        "Worked on M1 Data Visualization"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Centered title using HTML
st.markdown(
    "<h1 style='text-align: center;'>Group 104 - ML Proposal</h1>", 
    unsafe_allow_html=True
)

# Sections with dropdowns
with st.expander("Introduction and Literature Review"):
    st.write("""
    People's physical capabilities change over time, which can be challenging for those with disabilities or chronic conditions. 
    Therefore, a dynamic smart bathroom environment can be beneficial for health monitoring and diagnosing[1]. 
    In addition, monitoring physical activity with sensors and force readings can help create tailored treatments quickly and in real time[2]. 
    A smart bathroom environment can also be used for documenting physical data and identifying behavioral changes[3]. 
    The goal of our research is to develop a machine learning model that classifies movement as a person sits on the toilet into three classes: 
    sitting, offboarding, and onboarding to identify movement patterns in adults with physical impairments.
    """)

with st.expander("Dataset Description"):
    st.write("""
    Our dataset consists of force sensor readings from 40 people with 20 trials each. Some key features include:
    - Timestamps (Unix format): These represent the exact time when the sensor data was captured.
    - Load sensor values: These are readings from several sensors scattered around the toilet seat. 
      These numbers will be useful in determining the Center of Pressure (COP).
    """)
    st.write("Dataset Link: https://tinyurl.com/toiletseatdata")

with st.expander("Problem and Motivation"):
    st.write("""
    The purpose of this research is to detect unexpected behaviors such as abnormal sitting or rising patterns in participants with physical disabilities. 
    This technology can be integrated into alarm systems that alert family members or caretakers in the event of a fall or prolonged sitting. 
    In addition, this technology can help monitor changes in movement patterns of those with disabilities to monitor long term health.
    """)

with st.expander("Data Preprocessing Methods"):
    st.subheader("Data Cleaning and Selection")
    st.write("""
    We will remove all columns except the last 4 toilet seat data and time columns. 
    We will convert the Unix timestamp to the number of seconds passed since data collection began.
    """)

    st.subheader("Noise Reduction")
    st.write("""
    To reduce noise from the raw sensor data, we'll apply a low-pass filter. 
    This will smooth out rapid fluctuations while preserving the overall trends in force measurements.
    """)

    st.subheader("Feature Extraction")
    st.write("""
    We'll calculate the Center of Pressure (COP) using the four load sensor values to calculate a single metric 
    representing the overall pressure distribution.
    """)

with st.expander("ML Models/Algorithms"):
    st.subheader("Random Forest")
    st.write("""
    We'll use scikit-learn's RandomForestClassifier to differentiate between the three states (onboarding, sitting, off-boarding). 
    Random forests can capture non-linear relationships in data.
    """)

    st.subheader("Support Vector Machine (SVM)")
    st.write("""
    We'll implement an SVM using scikit-learn's SVC class which is accurate at finding optimal segments between classes in high-dimensional spaces.
    """)

    st.subheader("Logistic Regression")
    st.write("""
    We will use scikit-learn's LogisticRegression as a baseline model. 
    Logistic regression can serve as a good comparison point for more complex models.
    """)

with st.expander("Quantitative Metrics"):
    st.subheader("Precision")
    st.write("Ratio of correctly predicted positive instances to total predicted positive instances.")
    st.write("(true positives) / (false positives + true positives)")

    st.subheader("Accuracy")
    st.write("Ratio of correctly predicted instances to total number of instances")
    st.write("(true positives + true negatives) / all instances")

    st.subheader("F1 Score")
    st.write("Mean of precision and recall")
    st.write("(F1 Score=2×(precision*recall / precision + recall)")

    st.subheader("Recall")
    st.write("Ratio of correctly predicted positive instances to all actual positive instances")
    st.write("(true positives) / (false negatives + true positives)")

with st.expander("Project Goals and Expected Results"):
    st.write("""
    The project goal would be to have overall high precision and accuracy which therefore leads to a high F1 score. 
    A high recall is also important because of the need to minimize the number of false negatives in a medical setting. 
    One ethical consideration is if the model is wrong and does not find when a participant is struggling with their movement patterns. 
    Conversely, the model could wrongly identify patients with abnormal movement patterns. 
    For expected results, the model should be able to classify a time range in which a participant is getting on/off the toilet seat.
    """)

#midterm report
with st.expander("Methods: Midterm Report"):
    st.subheader("Data Preprocessing")
    st.write("""The force sensor readings from toilet seat sensors in CSV format, which was organized by each participant and timestamps. 
        Unncessary colums were removed from the data, leaving only the timestamps as well as the four sensor values. 
        Outliers that were outside the normal range were removed, as well as any invalid sensor reading.""")
    st.write("""The filtered data was then compiled into a single CSV file. The next step was to calculate the center of pressure 
        for the sensor readings in regards to the X and Y axis. These features will be considered COP_X and COP_Y. This was done by 
        adding up all sensor readings based on the physical layout of the sensors, and dividing it by the sum all 4 sensor readings for 
        that particular datapoint. Timestamps with missing data/data outside the normal range were removed.""")
    st.write("""Data was labeled based on the various categories with not being on the seat was considered the default state.
        The other three categories were onboarding, sitting, and offboard. The dataset was then labeled by assigning each data point a state based on the timestamp ranges. """)
    

    st.subheader("Ml Model: Random Forest")
    st.write("""The first model we decided to implement was the random forest classifier, using scikit-learn's randomforestclassifier method. 
    The three features that were used were Time_Diff_Milliseconds (tracking how much time as passed), COP_X(left to right weight distribution), COP_Y(front and back weight distribution). 
    The hyperparameters specified for the model include estimators at 200, a random_state at 42 (arbitrary randomness metric), and class_weights to have balanced results(higher importance for certain movements). 
    While Training the model, we used 80% 
    of the data for training, while the rest was used as testing data. 
    We also made sure to adjust class weights if needed, 
    as well as cross valdiate to make sure the model performed well.
    We chose the random forest model because it had the capacity to perform well with non-linear patterns. This non-linear relationship applies to our dataset and the features we had including
    center of pressure and time differences. It also allowed us to compare how important each of the features were in producing the results given which is important for analysis.
    The random forest model also allowed us to adjust class weights to better optimize the model to be more accurate and avoid overrepresenting the more frequent classes relative
    to the lesser ones. These purposes all together is why the Random Forest model was appealing to utilize in this case.""")

with st.expander("Results: Midterm Report"):
    st.subheader("Visualizations")
    st.write("""The confusion matrix is one way we can visualize the results of the model depicting the relationship between the predicted label and the true label of the classification.
    The values in the matrix that provide the most value to us are along the diagonal. A strong diagonal pattern in the confusion matrix indicates good classification, which is generally true in this case.
    However, the biggest thing to note is the difference between the values for the stable states and the values for the transition states. The sitting and not sitting states had values of 7509 and 13721,
    respectively on the diagonal signaling high correlation, but the onboarding and offboarding values were just 1333 and 975 which is far worse relatively indicative of poor correlation there.
    Overall, the confusion matrix was effective at showing whether or not a participant was sitting on the toilet seat, with some confusion regarding 
    whether or not the participant is in a transition state.""") 

    st.write("""The feature importance visualization allows us to see that the Time_Diff_Milliseconds had the highest imporance, 
    and was more effective in predicting the state of a particpant than the center of pressure values. As this feature was the most important, although not by a large margin, 
    it allows us to see that changes in sequential movement over time were key in determining different states of the participant. It is likely that this will come into play
    again when the remaining models are implemented and tested.""")
    

    st.image("ConfusionMatrix.jpg", caption="Confusion Matrix")
    st.image("FeatureImportance.jpg", caption="Feature Importance")
    st.subheader("Metrics")
    st.image("Metrics.jpg", caption="Metrics")
    st.subheader("Analysis")
    st.write("""Based on the quantitative metrics and visualizations, the random forest model was accurate in certain areas, but showed that it could improve in others.
     Non-transitional state detection showed greater accuracy compared to transition states. The overall accuracy was nearly 90% but we are relying on the F1 score, the mean of precision and recall values,
     to provide greater detail. The scores for sitting and not sitting were an outstanding 0.93 and 0.94 but the onboarding and offboarding were only a poor 0.7 and 0.62.
     
    It had difficulty in differentiating similar movement during transition, but was able to do it with moderate accuracy. There are a few possible explanations into why the model performed
     not as well in the transition states. One is the lower representation in the dataset. As it is seen in the support values for each label, the onboarding and offboarding had only 3885 as compared
     to 22317 for the sitting and non sitting states. This means that the transition states only comprised of less than 15% of the dataset. This is likely playing a factor in the results,
     allowing the model to be much better trained on some states compared to the others.
     Another possible reason is overlapping features for the transition states between time difference and the center of pressure values which may be better adjusted to with the other models.""")

    st.write("""The next steps that we plan to take are to implement the other models, Support Vector Machine and Logistic Regression, and run them on our dataset.
    In a similar fashion to the first one, we will analyze the metrics and visualizations to determine the results of these models. By doing so, we can compare the outcomes to find
    which model serves the needs for our data best and also understand why this is the case and the implications of any necessary trade-offs.
    A key change that we plan to carry out is adding features to help the model take into account the previous movement and state to improve on detecting the onbaording
    and offboarding transition states along with the current timestamp. We predict that this should considerably improve the performance of classification for these states
    compared to the preliminary results we have seen so far without having features to provide the previous state context information.
    Another next step is with the usage of timestamps, we want to implement and run a Support Vector Machine model because it is designed to extract meaningful relationships
    for time series data and that should work well for our dataset considering that we have the ability to use timestamps as a feature.""")
    
    st.write("""The final verdict in summary is that the model was effective in detecting non-transitional states, which were sitting on the seat, and not on the seat. 
    It was good at classification, but it had lower performance when it came to onboarding and offboarding. """)


# Gantt chart section with link
with st.expander("Gantt Chart: Midterm Report"):
   st.markdown("[View Gantt Chart in Google Sheets](https://docs.google.com/spreadsheets/d/1pWEyieNCmKAQnlG2C3LrY10Mgpy52Tdx/edit?usp=sharing&ouid=108969903742919067214&rtpof=true&sd=true)")

# Contribution table
with st.expander("Contribution Table: Midterm Report"):
   st.table(df)

# Proposal Video
with st.expander("Proposal Video"):
   st.markdown("[View Video on Youtube](https://youtu.be/OhnM3QEBTxs)")

# Citations
with st.expander("Citations"):
    st.write("""
    [1] B. Jones et al., "Smart Bathroom: Developing a Smart Environment to Study Bathroom Transfers," 2017, https://www.resna.org/sites/default/files/conference/2017/pdf_versions/outcomes/Jones.pdf

    [2] J. Pohl et al., "Accuracy of gait and posture classification using movement sensors in individuals with mobility impairment after stroke," Frontiers in Physiology, vol. 13, Sep. 2022, doi: https://doi.org/10.3389/fphys.2022.933987.

    [3] S. Hermsen, V. Verbiest, Marije Buijs, and E. Wentink, "Perceived Use Cases, Barriers, and Requirements for a Smart Health-Tracking Toilet Seat: Qualitative Focus Group Study," JMIR human factors, vol. 10, pp. e44850–e44850, Aug. 2023, doi: https://doi.org/10.2196/44850.
    """)
