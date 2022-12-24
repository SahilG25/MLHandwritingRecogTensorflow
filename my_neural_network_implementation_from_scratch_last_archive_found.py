# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:19:14 2022

@author: Sahil
"""
import numpy as np
from scipy.special import expit, softmax
##Miscellaneous functions##


def dotProduct(arr1, arr2): #taking two arrays (arr1 and arr2) as parameters
    if np.shape(arr1) != np.shape(arr2): # since vector dot product requires both vectors to be of the same dimensions, I have used a selection to check the dimensions of both arrays (parameters).
        return False # if the dimensions of arr1 and arr2 are not equal, then return the Boolean value False, as dot product cannot be carried out
    else:
        if len(np.shape(arr1)) == 1: # if there are no columns then I have to remove one of the iterative loops
            result = [] #assigned result to an empty array, this will be used to store the result of dot product
            for x in range(0, len(arr1)):
                temp = arr1[x] * arr2[x] # the result of multiplication between each element of arr1 and element of arr2 at corresponding position is added to temp
                result.append(temp) 
            return result
        result = [] #assigned result to an empty array, this will be used to store the result of dot product
        for x in range(0, len(arr1)):
            temp = [] # stores dot product results for each arr1[x] and arr2[x]
            for y in range(0, len(arr1[x])):
                temp.append(arr1[x][y] * arr2[x][y]) # the result of multiplication between each element of arr1[x] and element of arr2[x] at corresponding position is added to temp
            result.append(temp) #at the end of each iteration of x, array temp is appended to result
        return result # at the end of dot product between arr1 and arr2, the result of the dot product is returned.
def sigmoid(x): #function sigmoid() takes x as a parameter where x is a floating point number
    return expit(x) #since sigmoid is 1/(1+e^(-x)), I have used np.exp() for euler's number, where np.exp() takes x as a parameter.
def sigmoid_prime(x): #function sigmoid_prime() takes x as a parameter where x is a floating point number
    return expit(x)*(1-expit(x)) #since sigmoid is 1/(1+e^(-x)), I have used np.exp() for euler's number, where np.exp() takes x as a parameter.

def softmax1(x): # the parameter x will be an array with dimensions (10, 784) whic represents layer 2 nodes
    exp_total = 0 # this will contain the sum of every element of every array of x, which will be used as a denominator for the softmax formula
    for index in range(0, len(x)): # iteration in every node of layer 2 (i.e. x)
        exp_total += round(np.exp(x[index]), 6) # I will increment (x[index]) to exp_total, raised as a power of euler's number
    exp_final_prob = [] # this will store the final probabilities of each node in relation with every node in layer 2
    for index in range(0, len(x)): # iterates for every index between x
        element = round(np.exp(x[index])/exp_total, 6) # this is where the softmax formula is applied
        exp_final_prob.append(element)
    return exp_final_prob # final list of probabilities
def matrixMultiplication(mat1, mat2):
    counter_mat1 = False # if mat1 has no column, the counter is set true so the dimensions can be temporarily changed before being restored
    counter_mat2 = False# if mat2 has no column, the counter is set true so the dimensions can be temporarily changed before being restored
    row_mat1 = len(mat1)
    row_mat2 = len(mat2)
    #print(len(np.shape(mat1)))
    #print(len(np.shape(mat2)))
    if len(np.shape(mat1)) == 1: # case for if mat has no columns, then len(mat[0]) is an error, so I am validating that
        a = []
        for x in range(0, len(mat1)): # I am changing the dimensions of mat1 so the dimensions have a column in it, this way my matrix multiplication will work
            a.append([mat1[x]]) # adding [mat[x]] will add an extra dimension to the array
        mat1 = a # assigning new array to mat1
        counter_mat1 = True
    column_mat1 = len(mat1[0])
    if len(np.shape(mat2)) == 1:
        a = []
        for x in range(0, len(mat2)): # I am changing the dimensions of mat1 so the dimensions have a column in it, this way my matrix multiplication will work
            a.append([mat2[x]]) # adding [mat[x]] will add an extra dimension to the array
        mat2 = a # assigning new array to mat2
        counter_mat2 = True
    column_mat2 = len(mat2[0])
    if column_mat1 != row_mat2:
        return False
    else:
        final_dim = (row_mat1, column_mat2)
        final_list = []
        for x in range(0, row_mat1):
            row = []
            for y in range(0, column_mat2):
                row.append(0)
            final_list.append(row)
        for x in range(0, row_mat1):
            for y in range(0, column_mat2):
                total = 0
                for z in range(0, column_mat1):
                    total += mat1[x][z] * mat2[z][y]
                final_list[x][y] = total
        if counter_mat1 == True or counter_mat2 == True: # if initially mat1 or mat2 had no columns, then the resulting matrix won't have columns, we need to restore the final matrix
            final_list = np.resize(final_list, np.shape(final_list)[0]) # resized so dimensions only have (np.shape(final_list)[0]) and not columns
        return final_list
def transpose(arr1): # for backpropagation, certain matrix multiplication calculations wouldn= not work unless you transpose them, so this is my implementation
    counter_mat1 = False # if mat1 has no column, the counter is set true so the dimensions can be temporarily changed before being restored
    if len(np.shape(arr1)) == 1: # case for if mat has no columns, then len(mat[0]) is an error, so I am validating that
        return arr1
    final_list = [] # currently empty but will contain results of transposing    
    for x in range(0, len(arr1[0])): # I will iterate in the number of column first so I can locate element at every arr1 row at a given column and put them in a new list
        temp = [] # will contain elements from every row for every given column(x)
        for y in range(0, len(arr1)):
            temp.append(arr1[y][x]) # elements from every row for every given row(x) is appended to temp
        final_list.append(temp) # the new row is added to final_list()
    return final_list # the results are outputted
class Network:
    def __init__(self):
        #there will be 3 layers:
            #layer 1(input layer) which will have weights to be saved as an array of dimensions (1, 784)
            #layer 2(hidden layer) which will have weights to be saved as an array of dimensions (10, 784)
            #layer 3(output layer) which will have weights to be saved as an array of dimensions (10, 1)
        self.weights_layer1 = []
        for x in range(0, 10):
            self.weights_layer1.append(np.random.randn(784))
        self.biases_layer1 = []
        for x in range(0, 10):
            self.biases_layer1.append(np.random.randn(1))
        self.weights_layer2 = []
        for x in range(0, 10):
            self.weights_layer2.append(np.random.rand(10))
        self.biases_layer2 = []
        for x in range(0, 10):
            self.biases_layer2.append(np.random.rand(1))
    def layer1_to_layer2(self, inputArr):
        dotProduct_l1w_inpArr = [] # currently empty but will contain all the dot product arrays sums between self.weights_layer1 and inputArr
        #print("before dotProduct, the dimensions of self.weights_layer1: " + str(np.shape(self.weights_layer1)))
        for x in range(0, len(self.weights_layer1)): # since the dimension of self.weights_layer1 is (10, 784) and the size of inputArr is (1, 784), I wil need to dot product all the 10 arrays of self.weights_layer1 with inputArr and add it to dotProduct_l1w_inpArr
            dotProduct_temp = dotProduct(self.weights_layer1[x], inputArr) # every array (of length 784) of self.weights_layer1 is dot product with inputArr, and the result is assigned to dotProduct_temp
            dotProduct_l1w_inpArr.append(sum(dotProduct_temp)) # each sum of all 784 elements of dotProduct_temp is added to dotProduct_l1w_inpArr
        #print(np.shape(dotProduct_l1w_inpArr))
        # adding bias to dotProduct_l1w_inpArr
        n_values_with_added_biases = [] # currently empty but will contain all the dotProduct_l1w_inpArr elements with added self.biases_layer1 elements.
        for x in range(0, len(dotProduct_l1w_inpArr)):
            added_bias = dotProduct_l1w_inpArr[x] + self.biases_layer1[x][0] # I need to add the bias of each each node to all elements of dotProduct_l1w_inpArr
            n_values_with_added_biases.append(added_bias)
        self.dotP_inputArr_weights1_with_bias = n_values_with_added_biases
        # performing sigmoid on n_values_with_added_biases
        new_weights = [] # currently empty but will contail all the arrays of n_values_with_added_biases once sigmoid is applied
        for x in range(0, len(n_values_with_added_biases)):
            temp = [] # empty array which will contain the values of sigmoid activation done on array of n_values_with_added_biases[x]
            sigmoid_val = sigmoid(n_values_with_added_biases[x]) # sigmoid is applied to every floating point element of n_values_with_added_biases[x]
            temp.append(sigmoid_val)
            new_weights.append(temp)
        self.layer1_activation = new_weights # I created a new attribute layer1_activation that stores the activation of layer 1 when inputArr, and weights_layer1 are passed through sigmoid activation
        
    def layer2_to_layer3(self):
        matrixMult_l2w_l1 = matrixMultiplication(self.weights_layer2, self.layer1_activation) # matrix multiplication of weights of weights_layer2 (layer 3) and weights of weights_layer1 (layer 2)
        #return matrixMult_l2w_l1
        matrixMult_l2w_l1_sum = []
        for x in range(0, len(matrixMult_l2w_l1)):
            matrixMult_l2w_l1_sum.append(sum(matrixMult_l2w_l1[x])) # because I want a sum of all 784 elements in each array of matrixMult_l2w_l1, I need to use the sum() for each matrixMult_l2w_l1[x]          
        n_values_with_added_biases = [] # currently empty but will contain all the matrixMult_l2w_l1_sum elements with added self.biases_layer2 elements.
        for x in range(0, len(matrixMult_l2w_l1_sum)):
            added_bias = matrixMult_l2w_l1_sum[x] + self.biases_layer2[x][0] # I need to add the bias of each each node to all elements of matrixMult_l2w_l1_sum[x]
            n_values_with_added_biases.append(added_bias)
        #print(n_values_with_added_biases)
        final_probabilities = softmax(n_values_with_added_biases - max(n_values_with_added_biases)) # Softmax activation applied here, the output will be a list of probabilities (dimension (10))
        #print("final prob " + str(sum(final_probabilities)))
        self.layer2_activation = final_probabilities # I created a new attribute layer2_activation that stores the activation of layer 2 when weights_layer1, and weights_layer2 are passed through softmax activation
    def cost_and_cost_deriv(self, inputLabel): # this method will be used inside backpropagation for chain rule but also for testing if the error is minimised or not
        final_cost_function_arr = [] # this will contain result of (prediction - target)^2, I intend to make its dimensions (10, 1)
        final_cost_deriv_function_arr = [] # this will contain result of derivitive of cost 2*(prediction - target), I intend to make its dimensions (10, 1)
        for x in range(0, 10): # since there are 10 output nodes, so I have a made a loop that iterates 10 times
            final_cost_function_arr.append([(self.layer2_activation[x] - inputLabel[x])**2]) # cost of node appended to final_cost_function_arr
            final_cost_deriv_function_arr.append([(self.layer2_activation[x] - inputLabel[x])]) # cost derivative of node appended to final_cost_deriv_function_arr
        return final_cost_function_arr, final_cost_deriv_function_arr # I have returned both things, so the output of cost_and_cost_deriv() will be a list of size 2 containing 2 (10, 1) arrays
    def backpropagation(self, inputArr, inputLabel): # the only parameters I need are inputArr (dimensions (1,784)) and inputLabel (contains desired output) of dimensions (10)
        dCost_dx = self.cost_and_cost_deriv(inputLabel)[1] # this represents my partial derivative d(cost)/d(x)
        #print("cost: " + str(np.shape(dCost_dx)))
        self.layer1_activation = np.resize(self.layer1_activation, (10, 1))
        dWeight2 = matrixMultiplication(dCost_dx, transpose(self.layer1_activation)) # matrix multiplication will result in dimensions (784, 1)
        dBias2 = dCost_dx # bias error margin uses the sum of dCost_dx
        sigmoid_prime_layer1 = [] # currently empty but will contail all the arrays of weights_layer1 once sigmoid is applied, dimension will be (10, 784)
        for x in range(0, len(self.weights_layer1)):
            temp = [] # empty array which will contain the values of sigmoid activation done on array of self.weights_layer1[x]
            sigmoid_val = sigmoid_prime(self.dotP_inputArr_weights1_with_bias[x]) # sigmoid is applied to every floating point element of self.weights_layer1[x]
            temp.append(sigmoid_val)
            sigmoid_prime_layer1.append(temp)
        #dotProduct(self.weights_layer1, dWeight2)
        transposed_weights_layer2 = np.array(transpose(self.weights_layer2)) # self.weights_layer2 need to be transposed so matrix multiplication with dWeight2 of dimension (784, 1) can be done.
        matrixMult_W2_dCost_dx = transposed_weights_layer2.dot(dCost_dx) # this is a part of dActivation_layer1, we had to transpose weights_layer2 so that matrix multiplication is possible
        matrixMult_W2_dCost_dx = np.array(matrixMult_W2_dCost_dx)
        sigmoid_prime_layer1 = np.array(sigmoid_prime_layer1)
        #print("sigmoid_prime_layer1 shape: " + str(np.shape(sigmoid_prime_layer1)))
        dActivation_layer1 = matrixMult_W2_dCost_dx * sigmoid_prime_layer1 # we use the error interval from layer 3 so that we can work out the error interval at layer 1. The dimension of the array is (10,1)
        #print("dActivation_layer1 shape: " + str(np.shape(dActivation_layer1)))
        inputArr = np.resize(inputArr, (784,1)) # for Kaggle reference
        dActivation_layer1 = np.resize(dActivation_layer1, (10, 1)) # for Kaggle reference
        dWeight1 = matrixMultiplication(dActivation_layer1, transpose(inputArr)) # this is the error interval that the initial weights of layer 2 should be nudged by.
        dBias1 =  np.sum(dActivation_layer1, axis = 1) # bias error margin uses the sum of dActivation_layer1
        return dWeight1, dWeight2, dBias1, dBias2
    def update_weights_biases(self,learningRate, inputArr, inputLabel):
        dWeights1, dWeights2, dBiases1, dBiases2 = self.backpropagation(inputArr, inputLabel) # backpropagation outputs values of increment/decrement of weights of layers 2 and 3 and biases of layers 2 and 3 so the error margin can be minimised
        for x in range(0, len(self.weights_layer1)): # weights_layer1 has dimensions (10, 784) and each of the 784 elements of all 10 nodes must be updated so two loops are required, one to visit each node and one to visit all 784 elements of each node.
            for y in range(0, len(self.weights_layer1[x])):
                self.weights_layer1[x][y] = self.weights_layer1[x][y] + learningRate*dWeights1[x][y] # each element of weights_layer1 is nudged by a dWeights1 (which is multiplied by a learning rate)    
        for x in range(0, len(self.biases_layer1)):
            self.biases_layer1[x][0] = self.biases_layer1[x][0] + learningRate*dBiases1[x] # each bias in biases_layer1 is nudged to minimize error margin
        for x in range(0, len(self.weights_layer2)):
            for y in range(0, len(self.weights_layer2[x])):
                self.weights_layer2[x][y] += self.weights_layer2[x][y] + learningRate*dWeights2[x][y]  # each element of weights_layer2 is nudged by a dWeights1 (which is multiplied by a learning rate)
        #self.biases_layer2 = self.biases_layer2 - alpha * dBiasess2   
        for x in range(0, len(self.biases_layer2)):
            self.biases_layer2[x][0] = self.biases_layer2[x][0] + learningRate*dBiases2[x][0] # each bias in biases_layer1 is nudged to minimize error margin
    def gradient_descent(self, learningRate, epoch, inputArr, inputLabels): #
        recording_accuracy = []
        for x in range(0, epoch): # the neural network training will run until x = epoch-1
            prediction_count = 0 # this has a global sccope, it stores the number of correct predictions and will be used for determining accuracy
            for y in range(0, len(inputArr)): # within each epoch, the neural network is feeded with MNIST image data
                self.layer1_to_layer2(inputArr[y]) # feedforward that determines self.layer1_activation
                self.layer2_to_layer3() # feedforward that determines self.layer2_activation
                label = inputLabels[y] 
                label_list = [0,0,0,0,0,0,0,0,0,0]
                label_list[label] = 1 # by default, labels are stored as numbers in MNIST, however for my neural network, I need to convert it to a list where the index of label number is equal to 1
                self.update_weights_biases(learningRate, inputArr[y], label_list)  # weights and biases are updated based on MNIST image array
                if np.argmax(nn.layer2_activation, 0 ) == label: # numpy.argmax just outputs the index of the highest probability index in layer2_activation
                    prediction_count += 1 # if the neural network correctly recognises the image, increment the prediction count
            print("Iteration {} ".format(x) + "accuracy {}".format(prediction_count/len(inputArr))) # at the end of each epoch, the accuracy of the neural network is output
            recording_accuracy.append(prediction_count/len(inputArr))
        import matplotlib.pyplot as plt
        plt.plot(recording_accuracy)
        
nn = Network()
#openFile = open("MNIST_resized_ready_for_nn")
#randomInp = np.random.rand(1, 784)
#nn.layer1_to_layer2(randomInp)
#nn.layer2_to_layer3()
#a = nn.backpropagation(randomInp, [1,0,0,0,0,0,0,0,0,0])
#nn.update_weights_biases(0.09, randomInp, [1,0,0,0,0,0,0,0,0,0])
#nn.gradient_descent(0.009, 3, randomInp, [2])

#print(np.shape(a[0]), np.shape(a[1]), np.shape(a[2]), np.shape(a[3]))
#import json
#input1 = json.load(openFile)
'''while True:
    input_index = int(input("Enter index: "))
    input_array = input1["training_resized"][input_index]
    print("input_array shape: " + str(np.shape(input_array)))
    label = input1["training_label"][input_index]
    label_list = [0,0,0,0,0,0,0,0,0,0]
    label_list[label] = 1
    nn.layer1_to_layer2(input_array)
    nn.layer2_to_layer3()
    print("label: " + str(label_list) + " " + str(label))
    #print("cost before: " + str(nn.cost_and_cost_deriv(label_list)[0]))
    print("cost at label interval before: " + str(nn.cost_and_cost_deriv(label_list)[0][label]))
    a = nn.update_params(1, input_array, label_list)
    #print("cost before: " + str(nn.cost_and_cost_deriv(label_list)[0]))
    print("cost at label interval after: " + str(nn.cost_and_cost_deriv(label_list)[0][label]))
    print("prediction softmax: " + str(np.argmax(nn.layer2_activation, 0)))'''
openFile = open("undressed_data_normalised")
import json
input1 = json.load(openFile)
print("program started")
input_array_total = input1["training_resized"][:300]
label_total = input1["training_label"]
nn = Network()
nn.gradient_descent(0.0001, 6, input_array_total, label_total)
