# %%
# import dependencies

import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

# %%
# break up image into test and train datasets

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# %%
# test images

index = 200
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

# %%
# flatten dataset

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# %%
# force data to be between 0 and 1

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# %%
# implement activation function on neuron output


def sigmoid(z):
    return 1/(1+np.exp(-z))

# %%
# initialize the parameters for the neuron's inputs


def initialize_parameters(dims):
    w = np.random.randn(dims, 1) * 0.01
    b = 0

    return w, b

# %%
# runs data through neuron & calculates error


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X)+b)
    cost = -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    dw = (1/m)*np.dot(X, ((A-Y).T))
    db = (1/m)*np.sum(A-Y)

    grads = {"dw": dw,
             "db": db}

    return grads, cost
# %%
# trains neuron developing weights and bias


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate*dw
        b -= learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    return params, costs

# %%
# determines neurons final perdiction after training


def predict(w, b, X):

    Y_prediction = np.zeros((1, X.shape[1]))
    w = w.reshape(X.shape[0], 1)

    Y_prediction = sigmoid(np.dot(w.T, X)+b)
    Y_prediction = np.where(Y_prediction < 0.5, 0, 1)

    return Y_prediction
# %%
# put all the pieces together


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_parameters(X_train.shape[0])

    parameters, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# %%
# run model
d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=2000, learning_rate=0.005, print_cost=True)

# %%
# illustrate cost over time
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# %%
# illustrate error
index = 20
plt.imshow(test_set_x[:, index].reshape((64, 64, 3)))
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
      classes[d["Y_prediction_test"][0, index]].decode("utf-8") + "\" picture.")
# %%
