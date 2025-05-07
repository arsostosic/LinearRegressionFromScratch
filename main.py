import pandas as pd
import matplotlib.pyplot as plt
import math


data = pd.read_csv("study_score_dataset.csv")

# print(data)

# Zelimo da predstavimo nase podatke kao tacke na grafu

sampled_data = data.sample(n=100, random_state=42)  # Uzimamo 100 slučajnih primera
# plt.scatter(sampled_data.studytime, sampled_data.score)
# plt.show()

def loss_function(m, b, points): # MSE  - Mean Squared Error points = Xi
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y-(m*x+b))**2 # for ptelja je zapravo suma u ovom slucaju, a ovo je iteracija u sumi
    mse = total_error/float(len(points))
    return mse

def gradient_descent(m_now, b_now, points, L): # L-learning rate
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n)*x*(y-(m_now*x + b_now))
        b_gradient += -(2/n)*(y-(m_now*x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return  m, b

m = 0
b = 0
L = 0.01
epochs = 1000
for i in range(epochs):
    if i%100==0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m,b,data,L)

print(m,b)

plt.scatter(sampled_data.studytime, sampled_data.score, color="black")
plt.plot(list(range(1,12)), [m*x+b for x in range(1,12)], color="red")
plt.show()



print(f"MSE = {loss_function(4.69,41.77,data):.2f}")
print(f"RMSE = {math.sqrt(loss_function(4.69,41.77,data)):.2f}")
# RMSE - Shows MSE error in its natural scale as the original data are: for example here in score points
# so we can see what is the mean error of this model in score/points



# Training of LR1 model for future use, best m and b values
