package neuralNetworkLanguageDetection;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Scanner;

class Brain {

    private double[][]      neurons; //2D array of neurons - change to its own data type
    private double[][][]    axons; //3D array of axons
    
    private double          alpha; //0.1 would be a good default
    private int             prediction; //index with highest confidence in prediction


    /**
     * 
     * @param brainLayerSizes
     * @param startingAxonVariability
     * @param alpha
     */
    Brain(int[] brainLayerSizes, double startingAxonVariability, double alpha) {

        this.neurons             = new double[ brainLayerSizes.length ][]; //set the number of layers
        this.axons               = new double[ this.neurons.length-1 ][][]; //set the number of layers
        
        this.alpha               = alpha;

        for (int i = 0; i < this.neurons.length; i++)
            this.neurons[i] = new double[ brainLayerSizes[i] ];

        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i][ this.neurons[i].length-1 ] = 1.; //Set the last neuron of each layer to 1

            if (i < this.neurons.length - 1) {
                this.axons[i] = new double[ this.neurons[i].length ][];//

                for (int j = 0; j < this.axons[i].length; j++) {
                    this.axons[i][j] = new double[ this.neurons[i+1].length-1 ]; //this causes the exception

                    for (int k = 0; k < this.axons[i][j].length; k++)
                        this.axons[i][j][k] = (Math.random() * 2 - 1) * startingAxonVariability; //set the starting value
                        
                }
            }
        }
    }
    
    
    /**
     * 
     * @param brainLayerSizes
     */
    Brain(int[] brainLayerSizes) {
    		this(brainLayerSizes, 1.0, 0.1);
    }
    

    Brain(double[][] neurons, double[][][] axons, double alpha) {
        this.neurons = neurons.clone();
        this.axons = axons.clone();
        this.alpha = alpha;
    }


    /**
     * 
     * @param inputs
     * @param desiredOutputs
     * @param evolve
     */
    //This method should be redesigned to support more layers
    void useBrain(double[] inputs, double[] desiredOutputs, boolean evolve) {
        int[] nonzero = {this.neurons[0].length - 1}; //array to hold index values of nonzero inputs
        //Note: nonzero GREATLY increases performance by eliminating unnecessary operations

        for (int i = 0; i < this.neurons[0].length; i++) {
            this.neurons[0][i] = inputs[i];
            if (inputs[i] != 0)
                nonzero = Processing.append(nonzero, i); //only append the value of i if the input of i has a non-zero value
        }


        for (int i = 0; i < this.neurons.length; i++)
            this.neurons[i][this.neurons[i].length - 1] = 1.0;

        for (int i = 1; i < this.neurons.length; i++) {
          
            for (int j = 0; j < this.neurons[i].length - 1; j++) {
                double total = 0.;
                if (i == 1) {
                    for (int nonzeroIndex : nonzero)
                        total += this.neurons[i-1][nonzeroIndex] * this.axons[i - 1][nonzeroIndex][j];
                } else {
                    for(int k = 0; k < this.neurons[i - 1].length - 1; k++)
                        total += this.neurons[i-1][k] * this.axons[i - 1][k][j];
                }
                
                this.neurons[i][j] = sigmoid(total);
            }
        }


        //If the brain is set to "learn" and not just give back a result
        if(evolve) {

            final int lastLayer = this.neurons.length-1;

            //These nested loops are for the first layer of axons
            for (int i = 0; i < nonzero.length; i++) {
                for (int j = 0; j < this.neurons[1].length - 1; j++) {
                    double delta = 0.;
                    for (int k = 0; k < this.neurons[lastLayer].length - 1; k++)
                        //delta += 2 * (difference between output and desired output for neurons[output layer][k]) * value of neurons[output layer][k] * (1 - value of neuron k) * axon value * neuron value in previous layer * (1 - neuron
                        delta += 2 * (this.neurons[lastLayer][k] - desiredOutputs[k]) * this.neurons[lastLayer][k] * (1 - this.neurons[lastLayer][k]) * this.axons[1][j][k] * this.neurons[1][k] * (1 - this.neurons[1][k]) * this.neurons[0][nonzero[i]] * this.alpha;
                    this.axons[0][ nonzero[i] ][j] -= delta;
                }
            }

            for (int a = 1; a < lastLayer; a++) {
                for (int i = 0; i < this.neurons[a].length; i++) {
                    for (int j = 0; j < this.neurons[lastLayer].length - 1; j++)
                        this.axons[a][i][j] -= 2 * (this.neurons[lastLayer][j] - desiredOutputs[j]) * this.neurons[lastLayer][j] * (1 - this.neurons[lastLayer][j]) * this.neurons[a][i] * this.alpha; //delta
                }
            }
        }

        this.prediction = determinePrediction();
        double totalError = 0.;
        int end = this.neurons.length - 1; //index of the last layer

        for(int i = 0; i < this.neurons[end].length - 1; i++)
            totalError += Math.pow(neurons[end][i] - desiredOutputs[i], 2); //square the error and add it to the total
        
    }
    
    
    /**
     * 
     * @param input
     * @return
     */
    static double sigmoid(double input) {
        return 1.0 / (1.0 + Math.exp( -input ));
    }


    /**
     * 
     * @return
     */
    int determinePrediction() {
        double record = 0;
        int recordHolder = 0;
        int end = this.neurons.length-1;
        for (int i = 0; i < this.neurons[end].length - 1; i++) {
            if (this.neurons[end][i] > record) {
                record = this.neurons[end][i];
                recordHolder = i;
            }
        }
        return recordHolder;
    }
    
    
    /**
     * 
     * @return
     */
    int getPrediction() {
        return this.prediction;
    }
    
    
    /**
     * 
     * @return
     */
    private double[][] getNeurons() {
        return this.neurons;
    }
    
    
    /**
     * 
     * @return
     */
    private double[][][] getAxons() {
        return this.axons;
    }
    

    static Brain loadBrain(String neuronsFileName, String axonsFileName) throws FileNotFoundException {
    	
        Scanner scanner;
        ArrayList<String> neuronText = new ArrayList<>();
        ArrayList<String> axonText = new ArrayList<>();
    		
        scanner = new Scanner( new File(neuronsFileName) );
        while(scanner.hasNextLine())
            neuronText.add( scanner.nextLine() );
        scanner.close();
    		
        scanner = new Scanner( new File(axonsFileName) );
        while(scanner.hasNextLine())
            axonText.add( scanner.nextLine() );
        scanner.close();
    		
        double[][] neurons = new double[neuronText.size()][];
        double[][][] axons = new double[axonText.size()][][];
    		
        for(int i = 0; i < neurons.length; i++) {
            String[] temp = neuronText.get(i).split("\t");
            neurons[i] = new double[temp.length-1];
            for(int j = 0; j < neurons[i].length; j++) {
                //System.out.println(temp[j]);
                neurons[i][j] = Double.parseDouble(temp[j]);
            }
        }
    		
        for(int i = 0; i < axons.length; i++) {
            String[] temp1 = axonText.get(i).split("\t");
            axons[i] = new double[temp1.length-1][];
            for(int j = 0; j < axons[i].length; j++) {
                String[] temp2 = temp1[j].split(",");
                axons[i][j] = new double[temp2.length-1];
                for(int k = 0; k < axons[i][j].length; k++) {
                    //System.out.println(temp2[k]);
                    axons[i][j][k] = Double.parseDouble(temp2[k]);
                }
            }
        }
    		
        return new Brain(neurons, axons, 0.1);

    }


    static void exportBrain(Brain brain, String neuronsFileName, String axonsFileName) throws FileNotFoundException{

        PrintWriter pw = null;

        pw = new PrintWriter(neuronsFileName);

        //System.out.println("NEURONS:");
        for (double[] i : brain.getNeurons()) {
            String line = "";
            for (double j : i)
                //System.out.printf("%5.30f\t", brain.getNeurons()[i][j]);
                line += String.format("%5.30f\t", j);
            pw.println(line);
        }
        pw.close();

        pw = null;
        pw = new PrintWriter(axonsFileName);

        //System.out.println("AXONS:");
        for (double[][] i : brain.getAxons()) {
            String line = "";
            for (double[] j : i) {
                for (double k : j)
                    line += String.format("%5.30f,", k);
                line += "\t";
            }
            pw.println(line);
        }
        pw.close();

    }

}


