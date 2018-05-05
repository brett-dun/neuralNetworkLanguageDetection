package neuralNetworkLanguageDetection;

import java.io.FileNotFoundException;
import java.util.Scanner;

public class Main {
	

	private static final int TRAINS_PER_STEP = 20000; //increase this to improve training speed slightly
	private static final double MIN_ACCURACY = 0.6;
	
	private static final int MINIMUM_WORD_LENGTH = 4;

	private static final String[] LANGUAGES = {"Random","Key Mash","English","Spanish","French","German","Japanese","Swahili","Mandarin","Esperanto","Dutch","Polish","Lojban"};

	private static final String[][] trainingData = new String[LANGUAGES.length][];
	
	private static final int MAX_INPUT_LENGTH = 15;
	private static final int INPUTS_PER_CHAR = 27; //number of letters + 1 extra
	
	private static final int INPUT_LAYER_HEIGHT = INPUTS_PER_CHAR * MAX_INPUT_LENGTH + 1;
	private static final int MIDDLE_LAYER_NEURON_COUNT = 25; //This can be any number, around 20 is a good default
	private static final int OUTPUT_LAYER_HEIGHT = LANGUAGES.length + 1;
	

	private static Brain brain;
	private static int iteration = 0;
	private static boolean[] recentGuesses = new boolean[20000];
	private static int recentRightCount = 0;
	private static int desiredOutput = 0;
	private static int[] countedLanguages = {2,3,4,5}; //indexes matching the languages that will be learned
	
	private static int[][] longTermResults = new int[LANGUAGES.length][LANGUAGES.length];
	
	private static int streak = 0;
	private static int longStreak = 0;
	

	//This method is long and pretty inefficient, speed it up
	private static void train() {
	    int lang = countedLanguages[ (int)(Math.random()*countedLanguages.length) ]; //randomly select the language to use for this iteration
	    String word = ""; //String to store the chosen word
	    while(word.length() < MINIMUM_WORD_LENGTH)
	        word = trainingData[lang][(int)(Math.random()*trainingData[lang].length)];

	    desiredOutput = lang;
	    iteration++;
	    brain.useBrain( setInputs(word), setDesiredOutputs(desiredOutput), true );

	    //the remainder of this method only works to give the streak length which could be excluded
	    if(brain.getPrediction() == desiredOutput) {
	        if(!recentGuesses[iteration%recentGuesses.length])
	            recentRightCount++;
	        
	        recentGuesses[iteration%recentGuesses.length] = true;
	        streak++;
	        
	    } else {
	        if(recentGuesses[iteration%recentGuesses.length])
	            recentRightCount--;
	            
	        if(recentRightCount <= 0)
	            recentRightCount = 0;
	            
	        recentGuesses[iteration%recentGuesses.length] = false;
	        
	        if(streak > longStreak)
	            longStreak = streak;
	        
	        streak = 0;
	    }

	    longTermResults[brain.getPrediction()][desiredOutput]++;
	
	}
	
	
	private static double[] setInputs(String word) {
	    
	    double inputs[] = new double[INPUT_LAYER_HEIGHT]; //values for input layers
	        
	    for (int i = 0; i < MAX_INPUT_LENGTH; i++){
	        int c = 0;
	        if(i < word.length())
	            c = (int) word.toUpperCase().charAt(i)-64;
	        c = c > 0 ? c : 0; //check the maximum upper bound
	        inputs[ i*INPUTS_PER_CHAR+c ] = 1; //set the character 1 when it is the character in the word
	    }
	    
	    return inputs;
	    
	}
	
	
	private static double[] setDesiredOutputs(int desiredOutput) {
	  
	    double desiredOutputs[] = new double[OUTPUT_LAYER_HEIGHT];
	        
	    desiredOutputs[desiredOutput] = 1; //set the desired output to the designated language, 1 = 100% confidence in the result

	    return desiredOutputs;
	  
	}


	//this should probably handle non English characters better
	private static String sanitizeUserInput(String input) {
		return input.replaceAll("[^ a-zA-Z]", "");
	}


	public static void main(String[] args) {
		
	    //Load training data
	    for (int i = 0; i < LANGUAGES.length; i++)// {
	        trainingData[i] = Processing.loadStrings("data/output"+i+".txt");

	    //Parse training data
	    for (int i = 0; i < trainingData.length; i++)
	    	for (int j = 0; j < trainingData[i].length; j++)
	    		trainingData[i][j] = trainingData[i][j].substring(0, trainingData[i][j].indexOf(','));
	  
	    //setup the neuronet
	    int[] bls = { INPUT_LAYER_HEIGHT, MIDDLE_LAYER_NEURON_COUNT, OUTPUT_LAYER_HEIGHT};
	    brain = new Brain(bls, 1.0, 0.1);

	    System.out.println("TRAINING IN PROGRESS...");

		while ( ((double)recentRightCount)/recentGuesses.length < MIN_ACCURACY) {
	    	for (int j = 0; j < TRAINS_PER_STEP; j++)
	    		train();
	    	System.out.println(iteration+"\t"+((double)recentRightCount)/recentGuesses.length+"\t"+longStreak);
	    }

	    System.out.println("TRAINING COMPLETE.");

		//Save the network for use at a later time
		try {
			Brain.exportBrain(brain, "savedNetwork/neurons.txt", "savedNetwork/axons.txt");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	    Scanner scanner = new Scanner(System.in);
	    while (true) {

	    	System.out.print("Type a word, phrase, or sentence: ");

	    	String line = sanitizeUserInput(scanner.nextLine());
	    	String[] words = line.split(" ");

			int[] results = new int[LANGUAGES.length];
	    	for (String i : words) {
	    		if (i.length() < MINIMUM_WORD_LENGTH)
	    			continue;

				brain.useBrain(setInputs(i), setDesiredOutputs(0), false);
				results[brain.getPrediction()]++;
			}
			int max = 0;
	    	int max2 = 0;
	    	int max_index = 0;
	    	int max2_index = 0;
	    	for (int i = 0; i < results.length; i++) {
	    		if (results[i] > max) {
	    			max = results[i];
	    			max_index = i;
				} else if (results[i] > max2) {
	    			max2 = results[i];
	    			max2_index = i;
				}
			}
			System.out.println("The text you provided is most likely in " + LANGUAGES[max_index] + ", but it might be in " + LANGUAGES[max2_index] + ".");
		}

	}

}