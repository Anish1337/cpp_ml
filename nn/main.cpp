
#include "Tensor.hpp"

#include <iostream>

int main()
{
    std::cout << "cpp_ml builds\n";
    return 0;
}

/*#include "Tensor.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "MSELoss.hpp"
#include "NeuralNetwork.hpp"

#include <iostream>

int main(){
    // Build NN
    NeuralNetwork nn;
    net.add<Linear>(2, 4);
    net.add<ReLU>();
    net.add<Linear(4,1);

    MSELoss loss_fn;
    // learning rate and # epochs
    constexpr double lr = 0.01;
    constexpr int epochs = 10000

    for(int epoch=0; epoch<epochs; ++epochs){
        //forward pass
        Tensor pred = net.forward(X);
        // calc loss
        double loss = loss_fn.forward(prediction, y)
        // backpropagation
        Tensor grad = loss_fn.backward(prediction, y)
        net.update(learning_rate);
        // log every 1000
        if(epoch & 1000 == 0){
            std::cout<<"Epoch" << epoch
                << " Loss: " << loss << '\n';
        }
    }

    Tensor pred = net.forward(X);

    std::cout<<"\nPredictions:\n" << pred << '\n';

}*/
