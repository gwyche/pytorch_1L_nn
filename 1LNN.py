import torch

#initialize
targets = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,.9]
inputs = torch.rand(1,10)

target_tensor = torch.tensor(targets,dtype=torch.float32)
input_tensor = torch.tensor(inputs,dtype=torch.float32)
weight_tensor = torch.rand(10,10)-.5
ones_square_tensor = torch.ones(10,10)

learningRate = .3
runs = 1000
count = 0

while count < runs:
    #feedforward
    output_tensor = torch.matmul(input_tensor,weight_tensor)
    activated_tensor = torch.tanh(output_tensor)

    #calculate error
    total_error = torch.sum(torch.pow(torch.subtract(activated_tensor,target_tensor),2))/2
    print(total_error)

    #calculate activation derivatives
    a = torch.exp(-output_tensor)
    b = torch.exp(output_tensor)
    c = sum(a,b)
    d = torch.pow(c,2)
    delta_activation_tensor = 4/d

    #calculate output error derivatives
    delta_error_tensor = torch.subtract(activated_tensor,target_tensor,alpha=1)

    #calculate weight gradient tensor
    activation_and_error_deltas = torch.multiply(delta_error_tensor,delta_activation_tensor)
    input_tensor_transpose = torch.transpose(input_tensor,0,1)
    weight_gradient_tensor_from_inputs = torch.multiply(input_tensor_transpose,ones_square_tensor)
    complete_weight_gradient_tensor = torch.multiply(torch.multiply(activation_and_error_deltas,weight_gradient_tensor_from_inputs),learningRate)

    #update weight tensor
    weight_tensor = torch.subtract(weight_tensor,complete_weight_gradient_tensor,alpha=1)

    count = count + 1
















