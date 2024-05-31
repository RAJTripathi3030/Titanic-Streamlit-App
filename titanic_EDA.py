"""
This script performs exploratory data analysis (EDA) on the Titanic dataset using Streamlit. 
It includes univariate analysis and bivariate analysis to gain insights into the data.

Author: [Raj Tripathi]
Date: [31/05/2024]

Usage:
    - Run the script to launch the Streamlit app.
    - The app will display the Titanic dataset and provide options for analysis.
    - Select columns for univariate analysis and view histograms, bar plots, or pie charts.
    - Select columns for bivariate analysis and view scatter plots, crosstabs, or box plots.
"""
# Rest of the code...
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

df = pd.read_csv('train.csv')
st.title('Titanic Data Analysis:ship:')
st.write(df)

#Shape of data
st.subheader('Shape of data')
st.write("Total passengers :", df.shape[0]) 
st.write("Total columns :", df.shape[1])


col1, col2 = st.columns([1,1])

# WITH Keyword means - with column 1 do this thing that I am specifying in this code

with col1:
    col1.markdown("#### Data Types")
    col1.dataframe(df.dtypes.reset_index().rename(columns={'index':'Column', 0:'Data Type'}))

with col2:
    col2.markdown("#### Null Values")
    col2.dataframe(df.isnull().sum().reset_index().rename(columns={'index':'Column', 0:'Null Values'}))

# UNIVARIATE ANALYSIS -----------------------------------------------------------------------------------------------------------------------------------------
if st.sidebar.checkbox('Show Univariate Analysis'):
    st.markdown("<h2 style='text-decoration: underline;'> Univariate Analysis </h2>", unsafe_allow_html = True)
    
    # Exclude certain columns from the selectbox options as univariate on thesee columns will be of no use
    columns_to_exclude = ['PassengerId', 'Parch', 'Ticket', 'Cabin']
    # Fill all the columns that are suitable for UNIVARIATE ANALYSIS
    columns_for_selectbox = [col for col in df.columns if col not in columns_to_exclude]
    
    # Create a selectbox for the user to select a column, o/p of select box is stored in the variable "column_to_plot"
    column_to_plot = st.selectbox('Select a column for univariate analysis', columns_for_selectbox)

    st.write(f"Selected column: {column_to_plot}")
    
    # Check if the selected column is 'Name', if yes then make a word cloud
    if column_to_plot == 'Name':
        # Create a word cloud
        wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(df['Name']))
        plt.figure(figsize=(15,8))
        plt.imshow(wordcloud)
        plt.axis("off")
        st.pyplot(plt)
    else:
        # Count the values in the selected column
        column_counts = df[column_to_plot].value_counts().reset_index()
        column_counts.columns = [column_to_plot, 'Count']

        # Check if the selected column is one of the columns for which you want to create a pie chart
        if column_to_plot in ['Survived', 'Pclass', 'Sex', 'Embarked']:
            # Create a pie chart
            fig = px.pie(column_counts, values = 'Count', names = column_to_plot, 
                         title = f'{column_to_plot} Distribution', 
                         labels = {column_to_plot: column_to_plot})
            # Display the pie chart
            st.plotly_chart(fig)
            column_counts 
        else:
            # Create a histogram or bar plot
            fig = px.histogram(df, x = column_to_plot)
            # Display the histogram or bar plot
            st.plotly_chart(fig)

# MULTIVARIATE ANALYSIS -----------------------------------------------------------------------------------------------------------------------------------------
if st.sidebar.checkbox('Show Bivariate Analysis'):
    st.title("Titanic Dataset Bivariate Analysis")
    col1, col2 = st.columns(2)

    with col1:
        column1 = st.selectbox("Select first column", df.columns)
    with col2:
        column2 = st.selectbox("Select second column", df.columns)

    import plotly.graph_objects as go

    # Function to perform bivariate analysis
    def bivariate_analysis(df, col1, col2):
        if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
            # Scatter plot
            st.write(f"Scatter plot between {col1} and {col2}")
            fig = px.scatter(df, x=col1, y=col2)
            st.plotly_chart(fig)

            # Line plot
            st.write(f"Line plot between {col1} and {col2}")
            fig = px.line(df, x=col1, y=col2)
            st.plotly_chart(fig)

        elif df[col1].dtype == 'object' and df[col2].dtype == 'object':
            # Heatmap
            st.write(f"Heatmap between {col1} and {col2}")
            crosstab = pd.crosstab(df[col1], df[col2])
            fig = px.imshow(crosstab)
            st.plotly_chart(fig)

        elif df[col1].dtype == 'object' and df[col2].dtype in ['int64', 'float64']:
            # Box plot
            st.write(f"Box plot of {col2} by {col1}")
            fig = px.box(df, x=col1, y=col2)
            st.plotly_chart(fig)

            # Bar plot
            st.write(f"Bar plot of {col2} by {col1}")
            fig = px.bar(df, x=col1, y=col2)
            st.plotly_chart(fig)

        elif df[col1].dtype in ['int64', 'float64'] and df[col2].dtype == 'object':
            # Box plot
            st.write(f"Box plot of {col1} by {col2}")
            fig = px.box(df, x=col2, y=col1)
            st.plotly_chart(fig)

            # Bar plot
            st.write(f"Bar plot of {col1} by {col2}")
            fig = px.bar(df, x=col2, y=col1)
            st.plotly_chart(fig)

        else:
            st.write("Bivariate analysis is not possible for the selected columns.")

    # Perform analysis
    if column1 and column2:
        bivariate_analysis(df, column1, column2)
        
