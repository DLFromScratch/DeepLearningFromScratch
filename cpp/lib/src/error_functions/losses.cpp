#include <stdlib.h> 

class Losses{

public:

	float mean_squared_error (float , float);
	float mean_absolute_error (float, float);
	float binary_crossentropy (float, float);
	float categorical_crossentropy (std::vector<int>, std::vector<float>) 
}


Losses::mean_squared_error(float Ytrue, float Ypred)
/*
 *Function implimenting the mean squared error 
 *function. Currently implimented for a binary 
 *input. 
 *@author Luke Jenkinson 
 */ 
{
	//TODO
}


Losses::mean_absolute_error(float Ytrue, float Ypred)
{
	//TODO
}


Losses::binary_crossentropy(float Ytrue, float Ypred)
{
	//TODO
}


Losses::categorical_crossentropy(std::vector<int>, std::vector<float>)
{
	//TODO
}