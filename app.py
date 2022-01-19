import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Support Vector Machine Model 
svm_model = pickle.load(open('t20_Svm_model.pkl', 'rb'))

# Decision Tree Model 
decision_tree_model = pickle.load(open('t20_decisionTree_model.pkl', 'rb'))

# Random Forest Model
random_forest_model = pickle.load(open('t20_randomForest_model.pkl', 'rb'))


teams = ['Australia',
     'India',
     'Bangladesh',
     'New Zealand',
     'South Africa',
     'England',
     'West Indies',
     'Afghanistan',
     'Pakistan',
     'Sri Lanka']

cities = ['Colombo',
     'Mirpur',
     'Johannesburg',
     'Dubai',
     'Auckland',
     'Cape Town',
     'London',
     'Pallekele',
     'Barbados',
     'Sydney',
     'Melbourne',
     'Durban',
     'St Lucia',
     'Wellington',
     'Lauderhill',
     'Hamilton',
     'Centurion',
     'Manchester',
     'Abu Dhabi',
     'Mumbai',
     'Nottingham',
     'Southampton',
     'Mount Maunganui',
     'Chittagong',
     'Kolkata',
     'Lahore',
     'Delhi',
     'Nagpur',
     'Chandigarh',
     'Adelaide',
     'Bangalore',
     'St Kitts',
     'Cardiff',
     'Christchurch',
     'Trinidad']

st.title('T20 Score Predictor')

batting_team = st.selectbox('Select Batting Team',sorted(teams))
bowling_team = st.selectbox('Select bowling team', sorted(teams))


city = st.selectbox('Select city',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
        current_score = st.number_input('Current Score')
with col4:
        overs = st.number_input('Overs done(works for over > 5)')          
with col5:
        wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
        balls_left = 120 - (overs*6)
        wickets_left = 10 - wickets
        crr = current_score/overs


        input_df = pd.DataFrame(
         {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':[city], 'current_score': [current_score],'balls_left': 
          [balls_left], 'wickets_left': [wickets_left], 'crr': [crr], 'last_five': [last_five]})
        
        if str(input_df['batting_team'][0]) == str(input_df['bowling_team'][0]):
            raise Exception("Batting Team and Bowling Team Must Be Different For Prediction")


        elif int(input_df['current_score']) >= 720:
            raise Exception("Current Score Is Too High & Not Possible: Enter Below 720")
                
        elif int(input_df['current_score']) <= 0:
            raise Exception("Current Score Must Be Greater Than Zero")

        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over") 
               
        elif int(input_df['balls_left']) >= 120:
            raise Exception("Overs Must be Greater Than Zero ")  
                
        elif int(input_df['balls_left']) >= 90:
            raise Exception("Overs Must be Greater than 5 Over")     

        elif int(input_df['wickets_left'])  <= 0:
            raise Exception("Wickets Must be less than 10 Wicket")
                
        elif int(input_df['wickets_left'])  > 10:
             raise Exception("Wickets Must Be Positive Integar > 0")

        elif int(input_df['last_five'])  >= 181:
             raise Exception("Last Five Over Score Is Too High & Not Possible: Enter Below 181")
                
        elif int(input_df['last_five'])  <= 0:        
            raise Exception("Last Five Over Score Must Be Greater Than Zero ( > 0 )")

                
                
        # SVM Model
        
        svm_model_result = svm_model.predict(input_df)

        svm_graph = pd.DataFrame({'Score': [int(input_df['current_score'][0]),int(svm_model_result[0])],
                                                'Over': [int(overs), 20 ], 
                                                 'Team':[batting_team,                                                                                                                              batting_team]})
        
        

        fig, ax = plt.subplots(figsize=(10,4)) 
        ax.bar(svm_graph['Over'], svm_graph['Score'], color='g', edgecolor='b', linewidth=3, alpha=0.7, label='Runs')

        ax.set_xlabel('Overs')
        ax.set_ylabel('Runs')
        ax.set_title('SVM Predict Score: ' + str(svm_graph['Score'][1]))
        st.pyplot(fig) # SVM Model ENDED
        
        # Decision Tree Model 
        decision_tree_model_result = decision_tree_model.predict(input_df)

        decision_tree_graph = pd.DataFrame({'Score': [int(input_df['current_score'][0]),int(decision_tree_model_result[0])],
                                                'Over': [int(overs), 20 ], 
                                                 'Team':[batting_team,                                                                                                                              batting_team]})

        fig, ax = plt.subplots(figsize=(10,4)) 
        ax.bar(decision_tree_graph['Over'], decision_tree_graph['Score'], color='g', edgecolor='b', linewidth=3, alpha=0.7, label='Runs')

        ax.set_xlabel('Overs')
        ax.set_ylabel('Runs')
        ax.set_title('Decision Tree Predict Score: ' + str( decision_tree_graph['Score'][1]))
        st.pyplot(fig) # Decision Tree Model  ENDED
        
        # Random Forest Model 
        
        random_forest_model_result = random_forest_model.predict(input_df)

        random_forest_graph = pd.DataFrame({'Score': [int(input_df['current_score'][0]),int(random_forest_model_result[0])],
                                                'Over': [int(overs), 20 ], 
                                                 'Team':[batting_team,                                                                                                                              batting_team]})

        fig, ax = plt.subplots(figsize=(10,4)) 
        ax.bar(random_forest_graph['Over'], random_forest_graph['Score'], color='g', edgecolor='b', linewidth=3, alpha=0.7, label='Runs')

        ax.set_xlabel('Overs')
        ax.set_ylabel('Runs')
        ax.set_title('Random Forest Predict Score: ' + str( random_forest_graph['Score'][1]))
        st.pyplot(fig) # Random Forest ENDED
        
        
#  ********************************************************************** Footer ****************************************************#


footer="""<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
font-size: 100%;
text-align: center;
}
</style>
<div class="footer">
<p>Developed By FET, Students <br> &copy; 2022 SU Students, All rights reserved</p>
</div>
"""


st.markdown(footer,unsafe_allow_html=True)

# ************************************************** END ****************************************************#