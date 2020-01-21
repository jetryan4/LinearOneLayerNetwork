#Jet Ryan Project

#Linear classifier

using Plots, Images, Distributions, LinearAlgebra
using Random
using HDF5
using Polynomials
using StatsPlots, SpecialFunctions

# softmax activation
softmax(x) = exp.(x) ./ sum(exp.(x))
# and its derivative
softpmax(x) = softmax(x) .* (Diagonal(ones(10,10)) .- softmax.(x))'

function change_one_hot(value)

    #value will be an number 0-9
    tempvec = zeros(10)
    tempvec[value + 1] = 1
    return tempvec

end

# activation function for output layer
hout = softmax#hlin
hpout = softpmax#hplin

# activation function for hidden layer
hhid = hsig
hphid = hpsig

function test(weight1, input, target, idx)

    N = length(idx)
    D = length(input[1])

    error = 0.0
    # error_value = 0
    for n = 1:N

        x = input[idx[n]]
        t = target[idx[n]]

        # forward propagate

        a = weight1 * x

        z = hout(a)

        vector_holder = sum(((z .- t).^2))

        error += vector_holder

    end

    return error

end


function train(input, target)

    ep = 10^(-8)
    beta1 = 0.9
    beta2 =.999
    learning_rate = .000018

    # number of samples
    N = length(target)

    # dimension of input
    D = length(input[1])
    #println(D)

    O = 10 # number of classes in output

    # number to hold out
    Nhold = round(Int64, N/10)

    # number in training set
    Ntrain = N - Nhold

    # create indices
    idx = shuffle(1:N)

    trainidx = idx[1:Ntrain]
    testidx = idx[(Ntrain+1):N]

    println("$(length(trainidx)) training samples")
    println("$(length(testidx)) validation samples")

    # batch size
    B = 1000

    # input layer activation
    inputnode = zeros(D)

    # output node activation
    outputnode = zeros(10)

    # layer 1 weights
    weight1 = .00083*randn(O, D)
    bestweight1 = weight1

    numweights = prod(size(weight1))
    println("$(numweights) weights")

    error = test(weight1, input, target, trainidx)
    println("Initial Training Error = $(error)")

    error = test(weight1, input, target, testidx)
    println("Initial Validation Error = $(error)")

    pdf = Uniform(1,Ntrain)

    error = []

    stop = false

    index = 1
    m1 = zeros(O, D)
    v1 = zeros(O, D)

    #runtil = 0
    temp = 5000

    while temp > 1300

        grad1 = zeros(O, D)



        for n = 1:B

            sample = trainidx[round(Int64, rand(pdf, 1)[1])]
            x = input[sample]
            t = target[sample]
            #print(size(t))

            # forward propagate
            inputnode = x

            outputnode = weight1 * inputnode

            z = hout(outputnode)
            # end forward propagate

            # output error
            delta = z.-t

            # compute layer 1 gradients by backpropagation of delta

            grad1 = (hpout(outputnode) * delta * x')
            #print(size(grad1))

        end

        grad1 = grad1 / B

        #adam backpropagation

        # update layer 1 weights
        m1 = beta1 .* m1 .+ (1 - beta1) .* grad1
        mt1 = m1 ./ (1 - (beta1 ^ index))
        v1 = beta2 .* v1 .+ (1 - beta2) .* (grad1 .^ 2)
        vt1 = v1 ./ (1 - (beta2 ^ index))
        weight1 += -learning_rate .* mt1 ./ (sqrt.(vt1) .+ ep)

        temp = test(weight1, input, target, testidx)
        push!(error, temp)

        println("Batch Training Error = $(temp)")

        index = index + 1
    end

    error = test(weight1, input, target, trainidx)
    println("Final Training Error = $(error)")

    error = test(weight1, input, target, testidx)
    println("Final Validation Error = $(error)")

    return weight1, error
end

# the error rate for

function error_rate(weight1, input, target)

    N = length(input)
    D = length(input[1])

    #error = 0.0
    error_value = 0
    for n = 1:N
        x = input[n]
        t = target[n]

        # forward propagate

        a = weight1 * x

        z = hout(a)

        val = argmax(z)
        class = argmax(t)

        if(val != class)
            error_value = error_value + 1
        end

    end

    return error_value
end




function demo()

    #input, target = draw(300)
    h5open("mnist.h5", "r") do file
        labels = read(file, "train/labels")
        images = read(file, "train/images")

        input = []
        target = []

        N_size = size(labels)[1]

        for i = 1:N_size
            datanew = reshape(images[:,:,i], 784)
            prepend!(datanew, 1)

            targetnew = change_one_hot(labels[i])
            #preprocess the data with a mean shift of 785
            push!(target, targetnew)
            push!(input, datanew)
        end
        w1, err = train(input, target)

        #test_and_train_error_noh5(w1, w2)

        h5open("twolayer_10class.h5", "w") do file
            write(file, "weight1", w1)  # alternatively, say "@write file A"
            #write(file, "weight2", w2)
            # write(file, "error", err)
        end
    end

end


function test_and_train_error()

    h5open("twolayer_10class.h5", "r") do file
        w1 = read(file, "weight1")  # alternatively, say "@write file A"
        #w2 = read(file, "weight2")
        # err = read(file, "error")

        h5open("mnist.h5", "r") do file
            labels_train = read(file, "train/labels")
            images_train = read(file, "train/images")
            labels_test = read(file, "test/labels")
            images_test = read(file, "test/images")

            input_train = []
            target_train = []
            input_test = []
            target_test = []

            N_sizetrain = size(labels_train)[1]
            n_train = 0
            for i = 1:N_sizetrain

                #print
                n_train = n_train + 1
                datanew = reshape(images_train[:,:,i], 784)
                prepend!(datanew, 1)
                targetnew = change_one_hot(labels_train[i])
                push!(target_train, targetnew)
                push!(input_train, datanew)
            end
            error_train = error_rate(w1, input_train, target_train)

            N_sizetest = size(labels_test)[1]
            n_test = 0
            for i = 1:N_sizetest

                #print
                n_test = n_test + 1
                datanew = reshape(images_test[:,:,i], 784)
                prepend!(datanew, 1)
                targetnew = change_one_hot(labels_test[i])
                push!(target_test, targetnew)
                push!(input_test, datanew)
            end
            error_test = error_rate(w1, input_test, target_test)

            print("The number of misclassifications for train is: ")
            println(error_train)
            print("The result for the error rate for train is: ")
            println(error_train/n_train)

            print("The number of misclassifications for test is: ")
            println(error_test)
            print("The result for the error rate for test is: ")
            println(error_test/n_test)

        end


    end

end

@time demo()
test_and_train_error()
