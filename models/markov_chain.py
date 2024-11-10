# markov_chains.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress warnings

# Load and prepare data
data = pd.read_csv('./data/data.csv')
# Assumes columns: 'student_id', 'time_point', 'cesd_score', etc.

# Step 1: Define the Set of Latent States (Using Gaussian Mixture Model as a Latent Markov Model)
def find_optimal_states(data, max_states=6):
    aic_scores = []
    bic_scores = []
    models = {}
    
    for n_states in range(3, max_states+1):
        gmm = GaussianMixture(n_components=n_states, random_state=42).fit(data[['cesd_score']])
        aic_scores.append(gmm.aic(data[['cesd_score']]))
        bic_scores.append(gmm.bic(data[['cesd_score']]))
        models[n_states] = gmm
    
    optimal_states = min(range(3, max_states+1), key=lambda x: bic_scores[x-3])
    print(f"Optimal number of latent states: {optimal_states}")
    return models[optimal_states], optimal_states

gmm_model, num_states = find_optimal_states(data)

# Add latent state labels to data
data['latent_state'] = gmm_model.predict(data[['cesd_score']])

# Step 2: Calculate Transition Probabilities
def calculate_transition_matrix(data, num_states):
    transition_matrix = np.zeros((num_states, num_states))
    for student_id in data['student_id'].unique():
        student_data = data[data['student_id'] == student_id].sort_values('time_point')
        for i in range(len(student_data) - 1):
            current_state = student_data['latent_state'].iloc[i]
            next_state = student_data['latent_state'].iloc[i + 1]
            transition_matrix[current_state, next_state] += 1

    # Normalize rows to get probabilities
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix

transition_matrix = calculate_transition_matrix(data, num_states)
print("Transition Matrix:")
print(transition_matrix)

# Step 3: Estimate Steady States
def estimate_steady_state(transition_matrix):
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    steady_state = eigvecs[:, np.isclose(eigvals, 1)]
    steady_state = steady_state / steady_state.sum()  # Normalize
    return steady_state.real.flatten()

steady_state_distribution = estimate_steady_state(transition_matrix)
print("Steady State Distribution:", steady_state_distribution)

# Step 4: Regress Steady States Against Step 1 Results (Assuming 'step_1_score' is available)
# Prepare data for regression
data['steady_state_prob'] = data['latent_state'].map(dict(enumerate(steady_state_distribution)))
X = data.groupby('student_id')['steady_state_prob'].mean().values.reshape(-1, 1)
y = data.groupby('student_id')['step_1_score'].mean().values

# Perform Linear Regression
reg_model = LinearRegression()
reg_model.fit(X, y)
y_pred = reg_model.predict(X)

# Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Regression Results:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# import pandas as pd
# import numpy as np
# from collections import defaultdict

# class MarkovChain:
#     def __init__(self, states, transition_matrix=None):
#         self.states = states
#         self.transition_matrix = transition_matrix if transition_matrix is not None else np.ones((len(states), len(states))) / len(states)

#     def fit(self, data):
#         """
#         Fit the Markov Chain model to data.
#         The data should be a list or array of states over time for each student.
#         """
#         state_counts = defaultdict(int)
#         transition_counts = defaultdict(lambda: defaultdict(int))
        
#         for student_data in data:
#             for t in range(len(student_data) - 1):
#                 state_counts[student_data[t]] += 1
#                 transition_counts[student_data[t]][student_data[t + 1]] += 1
        
#         # Normalize the counts to get probabilities
#         self.transition_matrix = np.zeros((len(self.states), len(self.states)))
#         for i, state in enumerate(self.states):
#             total_count = sum(transition_counts[state].values())
#             if total_count > 0:
#                 for j, next_state in enumerate(self.states):
#                     self.transition_matrix[i, j] = transition_counts[state].get(next_state, 0) / total_count

#     def predict(self, start_state, steps=1):
#         """
#         Predict the next state given a start state using the transition matrix.
#         """
#         state_index = self.states.index(start_state)
#         predicted_state = np.random.choice(self.states, p=self.transition_matrix[state_index])
#         return predicted_state


# def load_data(file_path):
#     """
#     Load the dataset from the given file path.
#     """
#     return pd.read_csv(file_path)


# # Example usage
# if __name__ == "__main__":
#     states = ['low', 'moderate', 'severe']

#     # Update with your correct file path
#     file_path = r'C:\Users\shefa\Documents\GitHub\modeling-mental-health-using-markov-chains\data\data.csv'
    
#     # Load the data
#     data = load_data(file_path)

#     # For example, assuming 'cesd' column represents the depression state
#     # You might need to preprocess this column if it's not already categorical
#     student_data = []
#     for _, row in data.iterrows():
#         # Assuming `cesd` is a column that has depression states for each time point for a student
#         student_data.append(row['cesd'].split())  # This assumes `cesd` contains space-separated states (adjust accordingly)

#     # Initialize and fit the Markov Chain
#     markov = MarkovChain(states)
#     markov.fit(student_data)

#     # Predict the next state after 'low'
#     print("Predicted next state from 'low':", markov.predict('low'))
