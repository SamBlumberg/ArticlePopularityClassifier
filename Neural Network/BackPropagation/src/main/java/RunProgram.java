
import com.zavtech.morpheus.frame.DataFrame;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.Scanner;

public class RunProgram {
    public static void main(String[] args){

        int input;
        Scanner s = new Scanner(System.in);

        input = 1;
        while(input != 0){
            int choice ;

            do{
                System.out.println("Please press '2' to run Train the network OR press '3' to test the network using already trained network");
                choice = s.nextInt();
            }
            while(choice != 2 && choice != 3);

            if(choice == 2){
                Train();
            }
            else if(choice == 3){
                Test();
            }

            System.out.println("Please press '1' to continue or '0' to terminate program");
            input = s.nextInt();
        }


    }

    public static void Train(){

        // Read dataset
        DataFrame<Integer,String> frame1 = DataFrame.read().csv(options -> {
            options.setResource("APP_NEW_TRAIN.csv");
            options.setParallel(true);
            options.setColumnType("^[a-zA-Z0-9!@#$%^&*()_+\\-=\\[\\]{};':\"\\\\|,.<>\\/?]*$", Double.class);
        });

        NeuralNetwork unTrainedNetwork =  new NeuralNetwork("PP_Modified_100_50_NEW.json");

        try {
            // Build network topology
            unTrainedNetwork.buildNetworkTopology();

            BackpropagateAlgo backProp = new BackpropagateAlgo(unTrainedNetwork, frame1,2,0.5,0.9);
            // Train network
            NeuralNetwork trainedNetwork = backProp.trainNetwork();

            FileOutputStream f = new FileOutputStream(new File("TrainedNetwork.txt"));
            ObjectOutputStream o = new ObjectOutputStream(f);

            //Write trained network to file
            o.writeObject(trainedNetwork);

            o.close();
            f.close();


            System.out.println("Training is Done!.. Trained Network was saved as 'TrainedNetwork.txt'");



        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }

    public static void Test(){

        // Read Test data
        DataFrame<Integer,String> frame1 = DataFrame.read().csv(options -> {
            options.setResource("APP_NEW_TEST.csv");
            options.setParallel(true);
            options.setColumnType("^[a-zA-Z0-9!@#$%^&*()_+\\-=\\[\\]{};':\"\\\\|,.<>\\/?]*$", Double.class);
        });

        try {
            FileInputStream fi = new FileInputStream(new File("TrainedNetwork.txt"));
            ObjectInputStream oi = new ObjectInputStream(fi);
            // Read trained network from file
            NeuralNetwork trainedNetwork = (NeuralNetwork) oi.readObject();
            BackpropagateAlgo backProp = new BackpropagateAlgo(trainedNetwork, frame1,2,0.5,0.9);
            //Classify data
            backProp.predictData(trainedNetwork, frame1);


        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

    }
}
