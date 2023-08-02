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
			back_propagation(input_data, expected_output, learning_rate)
		
		var last_input_data = inputs[len(inputs) - 1]
		var network_output = forward_propagation(last_input_data)
		if (epoch + 1) % 500 == 0:
			print("Iteration: " + str(epoch + 1) + "\nNetwork Output: " + str(network_output))

func initialize_weights():
	# Calculate the total number of layers in the network
	var num_layers = num_hidden_layers + 1
	
	# Initialize weights and biases for each layer
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
		
		#init the biases with random values
		var layer_biases = []
		for i in range(output_size):
			var bias = randf_range(-weight_range, weight_range)
			layer_biases.append(bias)
		
		biases.append(layer_biases)

func sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

func forward_propagation(inputs):
	var layer_outputs = inputs
	
	for layer_idx in range(num_hidden_layers + 1):
		var layer_weights = weights[layer_idx]
		var layer_biases = biases[layer_idx]
		
		var next_layer_outputs = []
		for neuron_idx in range(len(layer_weights)):
			var neuron_weights = layer_weights[neuron_idx]
			var neuron_bias = layer_biases[neuron_idx]
			
			var neuron_output = 0.0
			for i in range(len(inputs)):
				neuron_output += inputs[i] * neuron_weights[i]
				
			neuron_output += neuron_bias
			neuron_output = sigmoid(neuron_output)
			
			next_layer_outputs.append(neuron_output)
		
		layer_outputs = next_layer_outputs
	
	return layer_outputs
			
			
func back_propagation(inputs, expected_output, learning_rate):
	var layer_outputs = forward_propagation(inputs)
	var num_layers = num_hidden_layers + 1
	
	#calculate the error for the output layer
	var output_errors = []
	for i in range(num_outputs):
		var output = layer_outputs[i]
		var error = (expected_output[i] - output) * output * (1.0 - output)
		output_errors.append(error)
		
	#update the weights and biases for the output layer
	var output_layer_weights = weights[num_layers - 1]
	for i in range(num_outputs):
		var neuron_weights = output_layer_weights[i]
		var neuron_bias = biases[num_layers - 1][i]
		for j in range(len(neuron_weights)):
			neuron_weights[j] += learning_rate * output_errors[i] * layer_outputs[j]
		neuron_bias += learning_rate * output_errors[i]
		
	#calculate the errors and update the weights for hidden layers
	for layer_idx in range(num_layers -2, -1, -1):
		var layer_weights = weights[layer_idx]
		var next_layer_weights = weights[layer_idx + 1]
		var next_layer_errors = []
		
		for i in range(len(layer_weights)):
			var neuron_weights = layer_weights[i]
			var error = 0.0
			
			for j in range(len(layer_weights)):
				var next_neuron_weights = next_layer_weights[j]
				var next_neuron_error = next_layer_errors[j]
				error += next_layer_weights[i] * next_neuron_error
				
			var neuron_output = layer_outputs[i]
			error += neuron_output * (1.0 - neuron_output)
			next_layer_errors.append(error)
			
			var neuron_bias = biases[layer_idx][i]
			for j in range(len(neuron_weights)):
				neuron_weights[j] += learning_rate * error * inputs[j]
			neuron_bias += learning_rate * error
		
		#update the weights and biases for the first hidden layer
		var first_layer_weights = weights[0]
		for i in range(len(first_layer_weights)):
			var neuron_weights = first_layer_weights[i]
			var neuron_bias = biases[0][i]
			for j in range(len(neuron_weights)):
				neuron_weights[j] += learning_rate * next_layer_errors[i] * inputs[j]
			neuron_bias += learning_rate * next_layer_errors[i]
		
#func back_propagation(inputs, expected_output, learning_rate):
#	#init gradients for wights and biases
#	var weight_gradients = []
#	var bias_gradients = []
#
#	#calculate the error of the output layer
#	var output_layer_idx = num_hidden_layers
#	var output_layer = forward_propagation(inputs)
#	var output_errors = []
#	for i in range(num_outputs):
#		var error = expected_output[i] - output_layer[i]
#		output_errors.append(error)
#
#	#backpropagate the errors and calculate the gradients
#	for layer_idx in range(num_hidden_layers, -1, -1):
#		var layer_weights = weights[layer_idx]
#		var layer_outputs = forward_propagation(inputs)
#		if layer_idx == num_hidden_layers:
#			layer_outputs = output_layer
#
#		var layer_gradients = []
#		for neuron_idx in range(len(layer_weights)):
#			var neuron_weights = layer_weights[neuron_idx]
#			var neuron_gradient = 0.0
#
#			for i in range(len(output_errors)):
#				var output_error = output_errors[i]
#				var weight = neuron_weights[i]
#				neuron_gradient += output_error * weight
#
#			#calculate the gradient for the neuron's activation function
#			var neuron_output = layer_outputs[neuron_idx]
#
#			neuron_gradient *= neuron_output * (1.0 - neuron_output)
#
#			#update the gradients for the neuron's weights and biases
#			var neuron_weight_gradients = []
#			for i in range(len(neuron_weights)):
#				var weight_gradient = neuron_gradient * inputs[i]
#				neuron_weight_gradients.append(weight_gradient)
#				neuron_weights[i] += learning_rate * weight_gradient
#
#			layer_gradients.append(neuron_weight_gradients)
#
#		#store the gradients for the layer
#		weight_gradients.insert(0, layer_gradients)
#
#	#update the weight and bianses using the calculated gradients
#	for layer_idx in range(num_hidden_layers + 1):
#		var layer_weights = weights[layer_idx]
#		var layer_weight_gradients = weight_gradients[layer_idx]
#
#		for neuron_idx in range(len(layer_weights)):
#			var neuron_weights = layer_weights[neuron_idx]
#			var neuron_weight_gradients = layer_weight_gradients[neuron_idx]
#
#			for i in range (len(neuron_weights)):
#				neuron_weights[i] += learning_rate * neuron_weight_gradients[i]
#
