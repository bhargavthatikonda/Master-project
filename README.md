# Master-project
master project on stok exchange
There are 7 set of Java programs:
--------------------------------

1) ArrayWriteable.java & TextArrayWriteable.java have been written since apache classes were giving issues when initializing with String arrays.

2) Backpropogation.java contains implementation logic of neural network. This defines network layer and calculation of weights.

3) BackpropogationMapper.java maps inputs and output records to key "1". All records in each node will be consumed by combiner.

4) BackpropogationCombiner.java takes all records in each node and trains neural network using all records available in particular node. Once it completely 
calculates weights using BPN it outputs weights of each layer and also number of records that was used for training. The number of records is necessary to calculated weighted average
of all weights.
Note: We have specified learning rate as 0.5. In case you want to change learning rate of network change it in this code.

5) BackpropogationReducer.java gather all weights from each node and calculates weighted average of weights. This is finally written in output location.

6) RunTraining.java: This is main program which controls most of things. This assumes 4 parameters have been specified to program:

1st parameter: Directory where input files are located.
2nd parameter: Directory where final set of weights will be written.
3rd parameter: Directory where input file on which you want to run the network to predict outcomes
4th parameter: Directory where output file of predicted outcomes will be written.



The program has set of configurations which makes this software generic to be used by any problem which wants to utilize BPN. Here is list of parameters:
---------------------------------------------------------------------------------------------------------------------------------------------------------
epoch: Specify number of times you want to train BPN with data that has been provided. For example XOR convergence was reached when epoch was set to 400.

inputColumns: Column number of input columns in input csv file.

outputColumns: Column number of output columns in input csv file. The columns whose value you want BPN to predict.

nodesInHiddenLayers: Specify number of nodes in each hidden layer and number of output columns. Ex: if there are 2 hidden nodes with number of nodes 4 and 3, we have 1output column.
Then values of this parameter will be 4,3,1


Notes: Program assumes following
1) Output file created by BackpropogationReducer.java is part-r-00000. In case file naming conventions is different in your system please change file name in RunTraining.java.
2) It assumes a file with name input.csv located in directory specified in 3rd parameter. If the file name is different please change the same.
3) It creates file with name output.txt in path specified by parameter number 4. In case you want final output file name to be different please change the same.
