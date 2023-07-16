extends Node

class_name NeuralNetwork

var num_inputs = 0
var num_hidden_layers = 0
var neurons_per_hidden_layer = []
var num_outputs = 0
var weights = []  # Declare weights as an instance variable
var biases = []

# Constructor
func _init(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs):
	self.num_inputs = num_inputs
	self.num_hidden_layers = num_hidden_layers - 1
	self.neurons_per_hidden_layer = neurons_per_hidden_layer
	self.num_outputs = num_outputs
	# Initialize weights
	initialize_weights()

func train(inputs, expected_ouputs, learning_rate, num_epochs):
	print("Training Neural Network...")
	for epoch in range(num_epochs):
		for i in range(len(inputs)):
			var input_data = inputs[i]
			var expected_output = expected_ouputs[i]
			forward_propagation(input_data)
			back_propogation(input_data, expected_output, learning_rate)
		
		var last_input_data = inputs[len(inputs) - 1]
		var network_output = forward_propagation(last_input_data)
		if (epoch + 1) % 500 == 0:
			print("Iteration: " + str(epoch + 1) + "\nNetwork Output: " + str(network_output))

func initialize_weights():
	# Calculate the total number of layers in the network
	var num_layers = num_hidden_layers + 1
	
	# Initialize weights for each layer
	for layer_idx in range(num_layers):
		var input_size
		var output_size
		
		# Determine the input and output size for the current layer
		if layer_idx == 0:
			# First hidden layer
			input_size = num_inputs
			output_size = neurons_per_hidden_layer[layer_idx]
		elif layer_idx == num_layers - 1:
			# Output layer
			input_size = neurons_per_hidden_layer[layer_idx - 1]
			output_size = num_outputs
		else:
			# Hidden layer
			input_size = neurons_per_hidden_layer[layer_idx - 1]
			output_size = neurons_per_hidden_layer[layer_idx]
			
		# Calculate the range for random initialization
		var weight_range = sqrt(6.0 / (input_size + output_size))
		
		# Initialize the weights with random values
		var layer_weights = []
		for i in range(output_size):
			var neuron_weights = []
			for j in range(input_size):
				var weight = randf_range(-weight_range, weight_range)
				neuron_weights.append(weight)
				
			layer_weights.append(neuron_weights)
			
		weights.append(layer_weights)  # Store the layer weights in the instance variable

func sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

func forward_propagation(inputs):
	var outputs = inputs
	
	for layer_weights in weights:
		var layer_outputs = []
		for neuron_weights in layer_weights:
			var weighted_sum = 0.0
			for i in range(len(outputs)):
				weighted_sum += neuron_weights[i] * outputs[i]
				
			# Apply activation function
			var activation_result = sigmoid(weighted_sum)
			
			layer_outputs.append(activation_result)
			
		outputs = layer_outputs
		
	return outputs

func back_propogation(inputs,expected_outputs, learning_rate):
	#perform forward propogation
	var outputs = forward_propagation(inputs)
	
	#init empty lists to store gradients of weights and biases
	var weight_gradients = []
	var bias_gradients = []
	
	#calculate the error at the output layer
	var output_errors = []
	for i in range(num_outputs):
		var output_error = outputs[i] - expected_outputs[i]
		output_errors.append(output_error)
	
	#calculate gradients for the output layer
	var output_gradients = []
	for i in range(num_outputs):
		var gradient = output_errors[i] * outputs[i] * (1.0 - outputs[i])
		output_gradients.append(gradient)
		
	#store the gradients for the output layer
	weight_gradients.push_back(output_gradients)
	
	#iterate over the hidden layers in reverse order
	for layer_idx in range(num_hidden_layers, 0, -1):
		var hidden_gradients = []
		
		#calculate the gradients for the neurons in the current hidden layer
		for neuron_idx in range(neurons_per_hidden_layer[layer_idx - 1]):
			var gradient = 0.0
			for next_neuron_idx in range(neurons_per_hidden_layer[layer_idx - 1]):
				gradient += weights[layer_idx][next_neuron_idx][neuron_idx] * weight_gradients[0][next_neuron_idx]
			
			var output = outputs[layer_idx][neuron_idx]
			gradient *= output * (1.0 - output)
			
			hidden_gradients.append(gradient)
			
		#store the gradients for the current hidden layer
		weight_gradients.insert(0, hidden_gradients)
		
	#update the weights and biases using the gradients
	for layer_idx in range(num_hidden_layers + 1):
		for neuron_idx in range(neurons_per_hidden_layer[layer_idx]):
			for weight_idx in range(len(weights[layer_idx][neuron_idx])):
				weights[layer_idx][neuron_idx][weight_idx] -= learning_rate * weight_gradients[0][neuron_idx - 1]
