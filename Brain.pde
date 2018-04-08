class Brain {
  
    private int[]           BRAIN_LAYER_SIZES; //array to store size of each layer of neurons - remove this variable
  
    private double[][]      neurons; //2D array of neurons - change to its own data type
    private double[][][]    axons; //3D array of axons
    
    private double          alpha; //0.1 would be a good default
    private int             topOutput; //index with highest confidence in prediction


    Brain(int[] brainLayerSizes, float startingAxonVariability, float alpha) {
      
        this.BRAIN_LAYER_SIZES   = brainLayerSizes;
        
        this.neurons             = new double[ brainLayerSizes.length ][]; //set the number of layers
        this.axons               = new double[ brainLayerSizes.length-1 ][][]; //set the number of layers
        
        this.alpha               = alpha;

        for(int i = 0; i < this.BRAIN_LAYER_SIZES.length; i++) {

            this.neurons[i] = new double[ BRAIN_LAYER_SIZES[i] ];
            
            this.neurons[i][ BRAIN_LAYER_SIZES[i]-1 ] = 1; //Set the last neuron of each layer to 1

            if (i < BRAIN_LAYER_SIZES.length - 1) {
                this.axons[i] = new double[ BRAIN_LAYER_SIZES[i] ][];

                for (int j = 0; j < brainLayerSizes[i]; j++) {
                    this.axons[i][j] = new double[ BRAIN_LAYER_SIZES[i+1]-1 ];

                    for(int k = 0; k < BRAIN_LAYER_SIZES[i+1] - 1; k++)
                        this.axons[i][j][k] = (Math.random() * 2 - 1) * startingAxonVariability; //set the starting value
                        
                }
            }
        }
    }
    
    
    /*
      this is for using the neuronet once it is trained
    */
    public double useBrain(double[] inputs, double[] desiredOutputs) {
        return useBrainGetError(inputs, desiredOutputs, false);
    }


    public double useBrainGetError(double[] inputs, double[] desiredOutputs, boolean evolve){
        int[] nonzero = {this.BRAIN_LAYER_SIZES[0] - 1};

        for (int i = 0; i < this.BRAIN_LAYER_SIZES[0]; i++) {
            this.neurons[0][i] = inputs[i];
            if (inputs[i] != 0)
                nonzero = append(nonzero, i);
        }

        for (int i = 0; i < this.BRAIN_LAYER_SIZES.length; i++)
            this.neurons[i][this.BRAIN_LAYER_SIZES[i] - 1] = 1.0;

        for (int i = 1; i < this.BRAIN_LAYER_SIZES.length; i++) {
          
            for (int j = 0; j < this.BRAIN_LAYER_SIZES[i] - 1; j++) {
                float total = 0;
                if (i == 1) {
                    for(int k = 0; k < nonzero.length; k++)
                        total += this.neurons[i-1][nonzero[k]] * this.axons[i - 1][nonzero[k]][j];
                } else {
                    for(int k = 0; k < this.BRAIN_LAYER_SIZES[i - 1] - 1; k++)
                        total += this.neurons[i-1][k] * this.axons[i - 1][k][j];
                }
                
                this.neurons[i][j] = sigmoid(total);
            }
        }

        if(evolve) {
            for (int i = 0; i < nonzero.length; i++) {
                for (int j = 0; j < this.BRAIN_LAYER_SIZES[1] - 1; j++) {
                    double delta = 0;
                    for (int k = 0; k < this.BRAIN_LAYER_SIZES[2] - 1; k++)
                        delta += 2 * (this.neurons[2][k] - desiredOutputs[k]) * this.neurons[2][k] * (1 - this.neurons[2][k]) * this.axons[1][j][k] * this.neurons[1][k] * (1 - this.neurons[1][k]) * this.neurons[0][nonzero[i]] * this.alpha;
                    this.axons[0][ nonzero[i] ][j] -= delta;
                }
            }

            for (int i = 0; i < this.BRAIN_LAYER_SIZES[1]; i++) {
                for (int j = 0; j < this.BRAIN_LAYER_SIZES[2]-1; j++)
                    this.axons[1][i][j] -= 2 * (this.neurons[2][j] - desiredOutputs[j]) * this.neurons[2][j] * (1 - neurons[2][j]) * this.neurons[1][i] * this.alpha; //delta
            }
        }

        this.topOutput = getTopOutput();
        double totalError = 0;
        int end = this.BRAIN_LAYER_SIZES.length - 1;

        for(int i = 0; i < this.BRAIN_LAYER_SIZES[end] - 1; i++)
            totalError += Math.pow(neurons[end][i] - desiredOutputs[i], 2);
        
        return totalError / (this.BRAIN_LAYER_SIZES[end] - 1); //average the error
        
    }
    
    
    
    /*
      sigmoid function
    */
    public double sigmoid(double input) {
        return 1.0 / (1.0 + Math.exp( -input ));
    }


    public int getTopOutput() {
        double record = 0;
        int recordHolder = 0;
        int end = this.BRAIN_LAYER_SIZES.length-1;
        for (int i = 0; i < this.BRAIN_LAYER_SIZES[end] - 1; i++){
            if (this.neurons[end][i] > record) {
                record = this.neurons[end][i];
                recordHolder = i;
            }
        }
        return recordHolder;
    }
    
    
}

