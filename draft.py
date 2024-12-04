import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")

# Contribution data
data = {
    "Name": ["Saloni Jain", "Sneha Pal", "Sooriya Senthilkumar", "Aneesh Sabarad", "Vibha Guru"],
    "Contribution": [
        "M1 Data Cleaning, M2 Feature Reduction, M3 Data Visualization",
        "M1 Coding, M2 Data Visualization, M3 Coding",
        "M1 Feature Reduction, M2 Coding, M3 Results",
        "M1 Results Evaluation, M2 Results, M3 Data Cleaning",
        "M1 Data Visualization, M2 Data Cleaning, M3 Feature Reduction"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Centered title using HTML
st.markdown(
    "<h1 style='text-align: center;'>Group 104 - ML Final Report</h1>", 
    unsafe_allow_html=True
)

# Introduction / Background
with st.expander("Introduction / Background"):
    st.write("""

    Smart home technologies, including sensors embedded in toilets, chairs, and beds, are gaining traction for health monitoring, particularly for the elderly or those with chronic conditions. 
             
    As people age or develop disabilities, their physical capabilities change, and a smart bathroom environment with health monitoring features could play a crucial role in addressing these challenges. 
             Our focus is on using sensors and force readings to monitor physical activity, enabling real-time health assessments and tailored treatments.

    These systems can track daily activities, provide feedback, and trigger alerts when abnormal behavior is detected. Existing research shows that sensor-based systems can identify changes in physical activity, offering valuable data for caregivers and healthcare professionals. However, challenges remain in accurately classifying and interpreting movement patterns, especially for people with physical impairments. This study aims to leverage machine learning to enhance sensor data analysis, improve classification accuracy, and reduce false positives.
    """)
    
with st.expander("Motivation"):
    st.write("""
    The purpose of this research is to detect unexpected behaviors such as abnormal sitting or rising patterns in participants with physical disabilities. 
    This offers several key benefits that drive the motivation for our study:
    - **Health Monitoring**: Using sensor data for real-time health monitoring to help detect unexpected behaviors such as abnormal sitting or rising patterns in participants indicating health deterioration.
    - **Behavioral Insights**: Analyzing long-term movement patterns of those with disabilities to identify patterns or changes in movement that can help understand the needs of patients and develop more effective treatments [3].
    - **Preventative Care**: Using alert systems that notify family members or caretakers in the event of a fall or prolonged sitting can help prevent further health deterioration.
    """)

# Problem Definition Section
with st.expander("Problem Definition"):
    st.write("""
    The problem at hand is to develop a machine learning model that can accurately classify different movement states of an individual sitting on a toilet, using sensor data. These movements include:
    - **Sitting**: The person is seated on the toilet.
    - **Offboarding**: The person is leaving the seat.
    - **Onboarding**: The person is approaching and sitting on the seat.
    - **Not on Toilet**: The person is not seated on the toilet.

    The primary challenge is to create a model that can accurately classify these states based on sensor data, while minimizing false positives and ensuring that abnormal behaviors, such as prolonged sitting or difficulties in standing, are detected. This is especially important for individuals with physical impairments, as it can enable timely interventions and improved care.
    """)


with st.expander("Dataset Description"):
    st.write("""
    Our dataset consists of force sensor readings from 40 people with 20 trials each. Some key features include:
    - Timestamps (Unix format): These represent the exact time when the sensor data was captured.
    - Load sensor values: These are readings from several sensors scattered around the toilet seat. 
      These numbers will be useful in determining the Center of Pressure (COP).
    """)
    st.write("Dataset Link: https://tinyurl.com/toiletseatdata")




st.title("Methods")

with st.expander("Data Preprocessing Methods"):
    st.subheader("Data Cleaning and Selection")
    st.write("""
    The raw sensor data often contains noise and irrelevant features. Therefore, in order to optimize model performance:
    - We removed all unnecessary columns, leaving only the timestamps and the four critical sensor values. For instance, columns such as floor tile data and grab bar data, which were irrelevant to our goal of investigating movement patterns specific to the toilet seat, were excluded.
    - The data was filtered to remove outliers that were outside the normal range, as well as any invalid sensor readings, to ensure the integrity of the dataset.
    - Timestamps with missing or invalid data were removed.
    - The filtered data was then compiled into a single CSV file for consistency and ease of analysis.
    - To standardize timestamps across different trials and make them easier to process, Unix timestamps were converted into the number of seconds since data collection began.
    """)

    st.subheader("Feature Engineering")
    st.write("""
    To enhance the model’s ability to identify patterns and reduce dimensionality:

    - **Feature Extraction**: To simplify the analysis while retaining meaningful information, we calculated the **Center of Pressure (COP)** using the four sensor values. 
        The COP provides a single metric that reflects the overall pressure distribution, offering insights into the user's approximate position on the toilet seat. Individual sensor values alone lack this context.
        Additionally, we incorporated data points from 1-2 seconds prior to the current timestamp to capture transitional states, such as changes in weight or center of pressure (COP), over time.


    The COP was calculated using the following formulas:

    - **COP X-Axis**:
        ```
        COP_X = (Σ (F_i * X_i)) / Σ F_i
        ```
    - **COP Y-Axis**:
        ```
        COP_Y = (Σ (F_i * Y_i)) / Σ F_i
        ```

    Where:
    - \( F_i \) is the force value from each sensor.
    - \( X_i, Y_i \) are the positions of the sensors relative to a defined coordinate system.

    To ensure accuracy, data points with COP values outside the range \([-2, 2]\) were removed, as these suggest that the user is not near the toilet seat.

    By calculating COP, we can:
    - Monitor balance during activities.
    - Detect irregularities in pressure distribution due to external factors or physical conditions.
    - Provide meaningful insights for feedback or training systems.
    """)



with st.expander("ML Models/Algorithms"):
    st.subheader("Random Forest")
    st.write("""
        The first model we decided to implement was the Random Forest classifier, using scikit-learn's RandomForestClassifier method. Random Forest was chosen for this study to its ability to handle non-linear relationships inherent to movement data which was necessary for our datset 
    that analyzed complex data like the center of pressure (COP) values and time differences. Additionally, the RF model is able to handle imbalanced datasets when compared to other classification based models; this 
    was necessary for our dataset because we had a lot more data points for sitting and not on toilet states in comparison to off-boarding and onboarding states. Lastly, this model allowed us to adjust class weights to avoid overrepresenting the more frequent classes relative
    to the lesser ones. 
    
    The three features that were used were Time_Diff_Milliseconds (tracking how much time as passed), COP_X (left to right weight distribution), COP_Y( front and back weight distribution). 
    The hyperparameters specified for the model include estimators at 200, a random_state at 42 (arbitrary randomness metric), and class_weights to have balanced results(higher importance for certain movements). 
    
    While Training the model, we used 80% of the data for training, while the rest was used as testing data. We also made sure to adjust class weights if needed, as well as cross valdiate to make sure the model performed well.
        """)

    st.subheader("Support Vector Machine (SVM)")
    st.write("""
    The second machine learning model we implemented was the Support Vector Machine (SVM). The SVM model was chosen for its ability to handle non-linear data and generalize well in high-dimensional spaces. 
    SVM can effectively separate classes, and its performance with limited data makes it suitable for applications where overfitting needs to be controlled. It is particularly useful for datasets where distinguishing between
    classes requires finding a decision boundary that maximizes the margin between them.

    We used a radial basis function (RBF) kernel, a common kernel function for SVM. This kernel can handle non-linearly separable data by projecting it into a higher-dimensional space where it can be separated using a hyperplane. 
    The RBF function computes the dot product and squared distances between features in the dataset, and then performs classification using linear SVM. Similar to Random Forest, class weights can be adjusted to handle class imbalances.

    Another step we implemented was cross-validation, where the dataset was divided into equal splits. For each split, the model was trained on some portions of the data and tested on the remaining split. 
    This helps reduce overfitting and ensures reliable performance. Cross-validation is particularly important for SVM models, as they are sensitive to hyperparameters, and this approach helps evaluate performance in a more robust way.
    """)

    st.subheader("XGBoost")
    st.write("""
    The last model we used was XGBoost, which is known for being effective in both regression and classification tasks. 
    The model works by building trees sequentially, with each tree improving on the previous one. The importance of each feature is understood over time as the model identifies which features contribute the most.

    XGBoost was our third chosen model because it is highly effective with structured datasets and can mitigate overfitting. 
    We believed that the repetitive process of capturing relationships within the data over each tree (and improving each tree sequentially) made XGBoost a better model than the logistic regression model we originally implemented. 
    Logistic regression assumes a linear relationship in the data, which makes it difficult to recognize non-linear patterns. On the other hand, XGBoost can learn complex relationships between features and handle these non-linear patterns more effectively.

    There are several parameters used to build these trees in sequence, including the number of estimators, learning rate, max depth, and weights. The number of estimators determines the number of trees the model will build, 
    which increases the model's ability to learn complex patterns but also opens up the risk of overfitting. The learning rate indicates the size of the update with each sequential tree—lower learning rates result in slower convergence. 
    The max depth parameter limits the depth of each tree, which affects the number of splits in a tree. Deeper trees tend to have more splits but also have a higher chance of overfitting. It is important to note that trees with low depth may 
    lead to underfitting, as they may not be complex enough to capture important patterns.
    """)


with st.expander("Results: Final Report"):
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
with st.expander("Gantt Chart: Final Report"):
   st.markdown("[View Gantt Chart in Google Sheets](https://docs.google.com/spreadsheets/d/1pWEyieNCmKAQnlG2C3LrY10Mgpy52Tdx/edit?usp=sharing&ouid=108969903742919067214&rtpof=true&sd=true)")

# Contribution table
with st.expander("Contribution Table: Final Report"):
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
