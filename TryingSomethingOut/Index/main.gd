extends Node2D

@export var num_inputs = 2
@export var num_hidden_layers = 1
@export var neurons_per_hidden_layer = [4]
@export var num_outputs = 1
@export var learning_rate = 0.3
@export var num_epochs = 10000

var training_inputs = [
	[0,0],
	[0,1],
	[1,0],
	[1,1]
]

var expected_outputs = [
	[0],
	[1],
	[1],
	[0]
]

func _ready():
	randomize()
	#print training data
	print("Training Inputs:")
	for i in range(len(training_inputs)):
		print(training_inputs[i])
		
	print("\n")
	
	#print expected outputs
	print("Expected Outputs:")
	for i in range(len(expected_outputs)):
		print(expected_outputs[i])
	
	
	var neural_network = NeuralNetwork.new(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs)
	neural_network.train(training_inputs, expected_outputs, learning_rate, num_epochs)
