package neuralNetworkLanguageDetection;

public class Main {
	

	static final int TRAINS_PER_STEP = 5000;
	
	static final int MINIMUM_WORD_LENGTH = 5;
	static final int LANGUAGE_COUNT = 13;
	
	static final String[][] trainingData = new String[LANGUAGE_COUNT][];
	
	static final int MAX_INPUT_LENGTH = 15;
	static final int INPUTS_PER_CHAR = 27; //number of letters + 1 extra
	
	static final int INPUT_LAYER_HEIGHT = INPUTS_PER_CHAR * MAX_INPUT_LENGTH + 1;
	static final int MIDDLE_LAYER_NEURON_COUNT = 30; //This can be any number, around 20 is a good default
	static final int OUTPUT_LAYER_HEIGHT = LANGUAGE_COUNT + 1;
	
	static final String[] LANGUAGES = {"Random","Key Mash","English","Spanish","French","German","Japanese","Swahili","Mandarin","Esperanto","Dutch","Polish","Lojban"};
	
	
	static Brain brain;
	static int lineAt = 0;
	static int iteration = 0;
	static boolean[] recentGuesses = new boolean[20000];
	static int recentRightCount = 0;
	static int desiredOutput = 0;
	static int[] countedLanguages = {2,4,5,8};
	
	static int[] langSizes = new int[LANGUAGE_COUNT];
	
	static int[][] longTermResults = new int[LANGUAGE_COUNT][LANGUAGE_COUNT];
	static int logNumber = 0;
	
	static int streak = 0;
	static int longStreak = 0;
	
	
	static void train() {
	    int lang = countedLanguages[ (int)(Math.random()*countedLanguages.length) ]; //randomly select the language to use for this iteration
	    String word = ""; //String to store the chosen word
	    while(word.length() < MINIMUM_WORD_LENGTH) {
	        int wordIndex = (int)( Math.random()*langSizes[lang] );
	        lineAt = binarySearch(lang, wordIndex);
	        String[] parts = trainingData[lang][lineAt].split(","); //split the words at the commas
	        word = parts[0];
	    }
	    desiredOutput = lang;
	    iteration++;
	    brain.useBrain( setInputs(word), setDesiredOutputs(desiredOutput), true );
	    
	    //Fix this code
	    //System.out.println(brain.getTopOutput() + "," + desiredOutput);
	    if(brain.getTopOutput() == desiredOutput) {
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
	    } //
	    
	    
	    longTermResults[brain.getTopOutput()][desiredOutput]++;
	
	}
	
	
	static int binarySearch(int lang, int n) {
	    return binarySearch(lang,n,0,trainingData[lang].length-1);
	}
	static int binarySearch(int lang, int n, int beg, int end){
	    
	    if(beg > end)
	        return beg;
	    
	    int mid = (beg+end)/2;
	  
	    String s = trainingData[lang][mid];
	    int diff = n-Integer.parseInt(s.substring(s.lastIndexOf(",")+1,s.length()));
	    
	    if(diff == 0) //if this is the value
	        return mid+1;
	    else if(diff > 0)
	        return binarySearch(lang,n,mid+1,end);
	    else if(diff < 0)
	        return binarySearch(lang,n,beg,mid-1);
	    
	    return -1; //cannot be found
	  
	}
	
	
	static double[] setInputs(String word) {
	    
	    double inputs[] = new double[INPUT_LAYER_HEIGHT]; //values for input layers
	        
	    for(int i = 0; i < MAX_INPUT_LENGTH; i++){
	        int c = 0;
	        if(i < word.length())
	            c = (int) word.toUpperCase().charAt(i)-64;
	        c = c > 0 ? c : 0; //check the maximum upper bound
	        inputs[ i*INPUTS_PER_CHAR+c ] = 1; //set the character 1 when it is the character in the word
	    }
	    
	    return inputs;
	    
	}
	
	
	static double[] setDesiredOutputs(int desiredOutput) {
	  
	    double desiredOutputs[] = new double[OUTPUT_LAYER_HEIGHT];
	        
	    desiredOutputs[desiredOutput] = 1; //set the desired output to the designated language, 1 = 100% confidence in the result
	    
	    //System.out.println(desiredOutputs);
	    
	    return desiredOutputs;
	  
	}
	
	
	/*private void prepareExitHandler () {
	    Runtime.getRuntime().addShutdownHook(new Thread(new Runnable(){
	        public void run () {            
	               
	        }
	    }));
	}*/



	public static void main(String[] args) {
		
	    //Load training data
	    for(int i = 0; i < LANGUAGE_COUNT; i++) {
	        trainingData[i] = Processing.loadStrings("data/output"+i+".txt");
	        String s = trainingData[i][trainingData[i].length-1];
	        langSizes[i] = Integer.parseInt(s.substring(s.indexOf(",")+1,s.length()));
	    }
	  
	    //setup neuronet
	    int[] bls = { INPUT_LAYER_HEIGHT, MIDDLE_LAYER_NEURON_COUNT, OUTPUT_LAYER_HEIGHT};
	    brain = new Brain(bls, 1.0, 0.1);
	    
	    
	    for(int i = 0; i < 2000; i++) {
	    		for(int j = 0; j < TRAINS_PER_STEP; j++)
	    			train();
	    			
	    		//System.out.println("Iteration: "+iteration+" \tLast 20K: "+percentify(((float)recentRightCount)/GUESS_WINDOW)+" \tLongest Streak: "+longStreak);
	    		System.out.println(iteration+"\t"+((double)recentRightCount)/recentGuesses.length+"\t"+longStreak);
	    	    lineAt++;
	    }

	}
	
	

}
