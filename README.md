# pea :melon:

A brain the size of a pea :melon:.  
There is no pea emoji, but I WANT an emoji here so there is a melon.  

This is an implementation of [dlidstrom/NeuralNetworkInAllLangs](https://github.com/dlidstrom/NeuralNetworkInAllLangs) in crystal lang.  

> [!NOTE]
> This repository exists only for my own entertainment.  

I understand the basics of the theory of simple neural networks, but I don't have the math for it so I moslty translated the GO implementation.  
I did it because I wanted to like two hours ago but I don't remind why. I think it seemed fun. Neural networks look fun.  
But things like 'gradient descent of a loss function of matrix products' looks fun only when in my tensorflow book.  

This is under MIT. I dont like headers into sourcefiles so I wont add any.  

## Status

It does seems to handle the logic function test. YAY. I have not yet checked the weights.  
It kind of handles semeion but not very well. Get to 85% accuracy with a 50/50 learn/test spread of the dataset.  
For some reasons the random number generator seems to yield number that quickly drifts apparts from the one given as an example.  
It's late so I'm gonna blame floating point arithmetic implementation differences and do something else more interseting with the remaining of my evening.  

## Usage

Test it with `crystal spec`.
It serves no purposes in itself. So maybe just look at the code, idk.  

## Contributors

- [globoplox](https://github.com/globoplox)
