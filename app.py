import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px

model = st.sidebar.selectbox("Select Model", ("Random Forest", "Decision Tree", "SVM"))

if model == "Decision Tree":
    pipe = pickle.load(open('t20_decisionTree_model.pkl', 'rb'))

    score = []

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
        overs = st.number_input('Overs done(works for over>5)')          
    with col5:
        wickets = st.number_input('Wickets out')

    last_five = st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score'):
        balls_left = 120 - (overs*6)
        wickets_left = 10 -wickets
        crr = current_score/overs


        input_df = pd.DataFrame(
         {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':[city], 'current_score': [current_score],'balls_left': 
          [balls_left], 'wickets_left': [wickets_left], 'crr': [crr], 'last_five': [last_five]})


        if int(input_df['current_score']) >= 720:
            raise Exception("Current Score Is Too High & Not Possible: Enter Below 720")


        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over")    

        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over")

        elif int(input_df['wickets_left'])  <= 0:
            raise Exception("Wickets Must be less than 10 Wicket")

        elif int(input_df['last_five'])  >= 181:
            raise Exception("Last Five Over Score Is Too High & Not Possible: Enter Below 181")    


        result = pipe.predict(input_df)

        graph = pd.DataFrame({'Score': [int(input_df['current_score'][0]),int(result[0])], 'Over': [ int(overs), 20 ], 'Team':[batting_team,                                                                                                                              batting_team]})

        fig, ax = plt.subplots() 
        ax.bar(graph['Over'], graph['Score'], color='g', edgecolor='b', linewidth=3, alpha=0.7, label='Runs')

        ax.set_xlabel('Overs')
        ax.set_ylabel('Runs')
        ax.set_title('Predict Score: ' + str(graph['Score'][1]))
        st.pyplot(fig)
        
elif model == "SVM":
    pipe = pickle.load(open('t20_Svm_model.pkl', 'rb'))

    score = []

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
        overs = st.number_input('Overs done(works for over>5)')          
    with col5:
        wickets = st.number_input('Wickets out')

    last_five = st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score'):
        balls_left = 120 - (overs*6)
        wickets_left = 10 -wickets
        crr = current_score/overs

        input_df = pd.DataFrame(
         {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':[city], 'current_score': [current_score],'balls_left': 
                                                 [balls_left], 'wickets_left': [wickets_left], 'crr': [crr], 'last_five': [last_five]})

        if int(input_df['current_score']) >= 720:
            raise Exception("Current Score Is Too High & Not Possible: Enter Below 720")


        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over")    

        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over")

        elif int(input_df['wickets_left'])  <= 0:
            raise Exception("Wickets Must be less than 10 Wicket")

        elif int(input_df['last_five'])  >= 181:
            raise Exception("Last Five Over Score Is Too High & Not Possible: Enter Below 181")    


        result = pipe.predict(input_df)

        graph = pd.DataFrame({'Score': [int(input_df['current_score'][0]),int(result[0])], 'Over': [ int(overs), 20 ], 'Team':[batting_team, 
                                                                                                                             batting_team]})

        fig, ax = plt.subplots() 
        ax.bar(graph['Over'], graph['Score'], color='g', edgecolor='b', linewidth=3, alpha=0.7,     label='Runs')

        ax.set_xlabel('Overs')
        ax.set_ylabel('Runs')
        ax.set_title('Predict Score: ' + str(graph['Score'][1]))
        st.pyplot(fig)




else:
    pipe = pickle.load(open('t20_randomForest_model.pkl', 'rb'))

    score = []

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
        overs = st.number_input('Overs done(works for over>5)')          
    with col5:
        wickets = st.number_input('Wickets out')

    last_five = st.number_input('Runs scored in last 5 overs')

    if st.button('Predict Score'):
        balls_left = 120 - (overs*6)
        wickets_left = 10 -wickets
        crr = current_score/overs


        input_df = pd.DataFrame(
         {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':[city], 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets_left], 'crr': [crr], 'last_five': [last_five]})


        if int(input_df['current_score']) >= 720:
            raise Exception("Current Score Is Too High & Not Possible: Enter Below 720")


        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over")    

        elif int(input_df['balls_left']) <= 0:
            raise Exception("Overs Must be less than 20 Over")

        elif int(input_df['wickets_left'])  <= 0:
            raise Exception("Wickets Must be less than 10 Wicket")

        elif int(input_df['last_five'])  >= 181:
            raise Exception("Last Five Over Score Is Too High & Not Possible: Enter Below 181")    

        # Model Prediction  
        result = pipe.predict(input_df)    
        graph = pd.DataFrame({'Score': [int(input_df['current_score'][0]),int(result[0])], 'Over': [ int(overs), 20 ], 'Team':[batting_team,                                                                                                                            batting_team]})
        # Prediction Graph 
        fig, ax = plt.subplots() 
        ax.bar(graph['Over'], graph['Score'], color='g', edgecolor='b', linewidth=3, alpha=0.7, label='Runs')
        ax.set_xlabel('Overs')
        ax.set_ylabel('Runs')
        ax.set_title('Predict Score: ' + str(graph['Score'][1]))
        st.pyplot(fig)
