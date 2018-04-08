package neuralNetworkLanguageDetection;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

//This class is for processing-compatible methods that we need
//Methods written by Brett Duncan
public class Processing {
	
	
	//https://processing.org/reference/append_.html
	public static int[] append(int[] array, int value) {
		
		int[] temp = new int[array.length+1];
		
		for(int i = 0; i < array.length; i++)
			temp[i] = array[i];
		
		temp[array.length] = value;
		
		return temp;
		
	}
	
	
	//https://processing.org/reference/loadStrings_.html
	public static String[] loadStrings(String fileName) {
		
		ArrayList<String> text = new ArrayList<String>();
		Scanner input = null;
		
		try {
			
			 input = new Scanner( new File(fileName) );
			 
			 while( input.hasNextLine() )
				 text.add(input.nextLine());
			 
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return (text.toArray( new String[text.size()] ));
		
	}

	
}
