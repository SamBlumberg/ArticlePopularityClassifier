import com.zavtech.morpheus.frame.DataFrame;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;
import java.util.concurrent.Callable;

public class BackpropagateAlgo {
    private NeuralNetwork netWorkTopology;
    private DataFrame<Integer,String> dataset;
    private int numberOfHiddenLayers;
    private double learningRate;
    private double momentumTerm;
    int[][] confusionMatrix;

    public BackpropagateAlgo(NeuralNetwork netWorkTopology, DataFrame<Integer, String> dataset,
                             int numberOfHiddenLayers, double learningRate, double momentumTerm) {
        this.netWorkTopology = netWorkTopology;
        this.dataset = dataset;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.learningRate = learningRate;
        this.momentumTerm = momentumTerm;
        this.confusionMatrix= new int[2][3];
    }

    public NeuralNetwork trainNetwork(){
        int numberOfEpoch = 0;
        // Training stopping condition (if recognitionRate > 80)
        double recognitionRate = 0.0;
        double dataLength = dataset.rowCount();



       while(recognitionRate < 0.8){
           // Program start time
            long start = System.currentTimeMillis();

            final int[] numberOfCorrectPrediction = {0};
            this.confusionMatrix= new int[2][3];

            //Feed input in parallel to the network
            this.dataset.rows().forEach(row -> {

                // True if the class was predicted correctly
                final boolean[] isCorrect = {true};

                //This is not necessarily required to be here
                // I just used it to pass the expected output of a given output unit
                for (Neuron unit : netWorkTopology.getOutPutLayer().values()) {
                    unit.setExpectedOutput(row.getDouble(unit.getName()));
                }

                // propagate the inputs forward
                for (Neuron unit : netWorkTopology.getInputLayer().values()) {
                    unit.setOutput(row.getDouble(unit.getName()));
                }

                // compute the net input and output of hidden units with respect to the previous layer
                // (previous layer is the unit's back edge)
                for (Neuron unit : netWorkTopology.getHiddenLayer().values()) {
                    computeInputOutput(unit, "hidden");
                }
                /*this.netWorkTopology.getHiddenLayer().values().parallelStream().forEach(unit -> computeInputOutput(unit, "hidden"));*/

                // compute the net input, output and error of output units with respect to the previous layer
                for (Neuron unit : netWorkTopology.getOutPutLayer().values()) {
                    computeInputOutput(unit, "output");

                    // Compute error in output unit
                    // Error = output(1-output)(ExpectedValue-Output)
                    unit.setError(unit.getOutput() * (1 - unit.getOutput()) * (unit.getExpectedOutput() - unit.getOutput()));

                    //Add an entry into confusion matrix (this is used to compute error and recognition rate)
                    insertConfusionMatrix(unit.getExpectedOutput(), unit.getOutput());

                }

                // Backpropagate Errors

                // Compute Error in hidden unit starting from the back
                // Pick a unit from the output layer and use it to retrieve last hidden layer edges
                List<Edge> currentHiddenLayerEdges = this.netWorkTopology.getOutPutLayer().values().iterator().next().getBackEdge();
                for(int i = 0; i<this.numberOfHiddenLayers; i++){
                    for(Edge edge: currentHiddenLayerEdges){
                        this.netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron()).setError(computerHiddenUnitError(this.netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron())));
                    }

                    String nextRandUnit = currentHiddenLayerEdges.get(0).getLeftNeuron();
                    currentHiddenLayerEdges = this.netWorkTopology.getHiddenLayer().get((nextRandUnit)).getBackEdge();
                }

                // Update all weights in the network
                // changeInWeightIJ = learnRate*Errj*OutputI
                // weightIJ =  weightIJ + changeInWeightIJ
                for(Edge edge: this.netWorkTopology.getEdges().values()){
                    double Errj;
                    double outputI;
                    // if unit J is an output unit
                    if(this.netWorkTopology.getOutPutLayer().containsKey(edge.getRightNeuron())){
                        Errj = this.netWorkTopology.getOutPutLayer().get(edge.getRightNeuron()).getError();
                    }
                    // else unit J is a hidden unit
                    else{
                        Errj = this.netWorkTopology.getHiddenLayer().get(edge.getRightNeuron()).getError();
                    }
                    // if unit I is an input unit
                    if(this.netWorkTopology.getInputLayer().containsKey(edge.getLeftNeuron())){
                        outputI = this.netWorkTopology.getInputLayer().get(edge.getLeftNeuron()).getOutput();
                    }
                    // else unit I is a hidden unit
                    else{
                        outputI = this.netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron()).getOutput();
                    }

                    double changeInWeight = learningRate * Errj * outputI;
                    // Update weightIJ with respect to previous change in weight multiply by momentum term
                    edge.setWeight(momentumTerm * edge.getChnageInWeight() + (changeInWeight + edge.getWeight()));
                    edge.setChnageInWeight(changeInWeight);
                }

                // Update all Bias
                // change in Bias = learnRate * Error in unit
                // new Bias = old Bias + change in Bias
                for(Neuron unit: this.netWorkTopology.getHiddenLayer().values()){
                    unit.setBias(unit.getBias() + (learningRate * unit.getError()));
                }
                for(Neuron unit: this.netWorkTopology.getOutPutLayer().values()){
                    unit.setBias(unit.getBias() + (learningRate * unit.getError()));
                }

            });

            recognitionRate = computeRecognitionRate();
           /*for (Neuron unit : netWorkTopology.getOutPutLayer().values()) {
               System.out.println("Expected: "+unit.getExpectedOutput() +" Output: "+unit.getOutput());
           }*/
           System.out.println("Recognition Rate: "+recognitionRate*100 +"%");
            numberOfEpoch += 1;


           // Program end time
           long end = System.currentTimeMillis();
           NumberFormat formatter = new DecimalFormat("#0.00000");
           System.out.print("Epoch " +numberOfEpoch + " Execution time is " + formatter.format((end - start) / 1000d) + " seconds");
           System.out.println();
           System.out.println();
        }

        return this.netWorkTopology;
    }

    public Neuron expectedOutput(Neuron unit, double expectedOutput){
        unit.setExpectedOutput(expectedOutput);
        return unit;
    }

    public void insertConfusionMatrix(double expectedOutput, double output){
        if(expectedOutput == 1){
            //True positive (TP)
            if((expectedOutput - output) <= 0.5){
                this.confusionMatrix[0][0]++;
            }
            //False Negative (FN)
            else{
                this.confusionMatrix[0][1]++;
            }
        }
        else{
            //False Positive (FP)
            if((expectedOutput - output) > 0.5){
                this.confusionMatrix[1][0]++;
            }
            //True Negative (TN)
            else{
                this.confusionMatrix[1][1]++;
            }
        }

    }

    public double computeRecognitionRate(){
        // Number of positive class label (P)
        this.confusionMatrix[0][2] = this.confusionMatrix[0][0] + this.confusionMatrix[0][1];
        //Number of Negative class lable (N)
        this.confusionMatrix[1][2] = this.confusionMatrix[1][0] + this.confusionMatrix[1][1];

        //Recognition rate = TP+TN/P+N
        double recognitionRate = (double)(this.confusionMatrix[0][0] + this.confusionMatrix[1][1])/
                (this.confusionMatrix[0][2] + this.confusionMatrix[1][2]);
        //Error rate = FP+FN/P+N
        double errorRate = (double)( this.confusionMatrix[1][0] + this.confusionMatrix[0][1])/
                (this.confusionMatrix[0][2] +  this.confusionMatrix[1][2]);

        System.out.println("Error Rate: " +errorRate*100 +"%");
        return recognitionRate;
    }

    // Thread to Compute net input and output of a given hidden or output unit
    class SoldierInput implements Callable<Double>{
        Neuron unit;
        String unitType;

        public SoldierInput(Neuron unit, String unitType) {
            this.unit = unit;
            this.unitType = unitType;
        }

        @Override
        public Double call() throws Exception {
            final double[] prod = {0};
            // To compute the net input in a given hidden or output units, we use their back edges weight
            this.unit.getBackEdge().parallelStream().forEach((edge) ->{
                double Oi;
                if(this.unitType.equals("hidden")){
                    // if edge is in hidden layer
                    if(netWorkTopology.getHiddenLayer().containsKey(edge.getLeftNeuron())){
                        // output of previous layer i
                        // (Note a hidden layer can be a previous layer of another hidden layer if the network has several hidden layers
                        Oi = netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron()).getOutput();
                    }
                    // else edge is in input layer
                    else{
                        // output of previous layer i
                        Oi = netWorkTopology.getInputLayer().get(edge.getLeftNeuron()).getOutput();
                    }
                }
                else{
                    // output of previous layer i
                    Oi = netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron()).getOutput();
                }
                // weight of edge ij = edge.getWeight()
                prod[0] += (edge.getWeight() * Oi);
            });

            unit.setInput(prod[0] + unit.getBias());
            // Activate unit by applying activation function
            // I used a logistic function (1/1+e^-input)
            unit.setOutput(1/(1+Math.exp(-unit.getInput())));

            return prod[0] + unit.getBias();
        }
    }

    public void computeInputOutput(Neuron unit, String unitType){
        final double[] prod = {0};
        // To compute the net input in a given hidden or output units, we use their back edges weight
        for (Edge edge : unit.getBackEdge()) {
            double Oi;
            //System.out.println("Unit: "+ unit.getName());
            //System.out.println("Edge: "+ edge.getName());
            if(unitType.equals("hidden")){
                // if edge is in hidden layer
                if(netWorkTopology.getHiddenLayer().containsKey(edge.getLeftNeuron())){
                    // output of previous layer i
                    // (Note a hidden layer can be a previous layer of another hidden layer if the network has several hidden layers
                    Oi = netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron()).getOutput();
                }
                // else edge is in input layer
                else{
                    // output of previous layer i
                    Oi = netWorkTopology.getInputLayer().get(edge.getLeftNeuron()).getOutput();
                }
            }
            else{
                // output of previous layer i
                Oi = netWorkTopology.getHiddenLayer().get(edge.getLeftNeuron()).getOutput();
            }
            // weight of edge ij = edge.getWeight()
            prod[0] += (edge.getWeight() * Oi);
        }

        unit.setInput(prod[0] + unit.getBias());

        // Activate unit by applying activation function
        // I used a logistic function (1/1+e^-input)
        if(unitType.equals("hidden")){
            unit.setOutput(1/(1+Math.exp(-unit.getInput())));

        }
        else{
            unit.setOutput(1/(1+Math.exp(-unit.getInput())));
        }

    }

    public double computerHiddenUnitError(Neuron unit){
        final double[] sumK = {0.0};
        for(Edge edge: unit.getFrontEdges()){
            double errK ;
            // if higher unit K is in output layer
            if(netWorkTopology.getOutPutLayer().containsKey(edge.getRightNeuron())){
                // Retrieve Error of higher unit K
                errK = netWorkTopology.getOutPutLayer().get(edge.rightNeuron).getError();
                sumK[0] += errK * edge.getWeight();
            }
            // if higher unit K is in hidden layer
            else{
                errK = netWorkTopology.getHiddenLayer().get(edge.getRightNeuron()).getError();
                sumK[0] += errK * edge.getWeight();
            }
        }

        return unit.getOutput() * (1 - unit.getOutput()) * sumK[0];
    }

    public void predictData(NeuralNetwork trainedNetwork, DataFrame<Integer, String> testData){

        final int[] numberOfCorrectPrediction = {0};
        double recognitionRate;
        this.netWorkTopology = trainedNetwork;

        final int[] count = {1};

        this.confusionMatrix= new int[2][3];

        testData.rows().forEach(row ->{

            // True if the class was predicted correctly
            final boolean[] isCorrect = {true};

            //This is not necessarily required to be here
            // I just used it to pass the expected output of a given output unit
            for (Neuron unit : trainedNetwork.getOutPutLayer().values()) {
                unit.setExpectedOutput(row.getDouble(unit.getName()));
            }

            // propagate the inputs forward
            for (Neuron unit : trainedNetwork.getInputLayer().values()) {
                unit.setOutput(row.getDouble(unit.getName()));
            }

            // compute the net input and output of hidden units with respect to the previous layer
            // (previous layer is the unit's back edge)
            for (Neuron unit : trainedNetwork.getHiddenLayer().values()) {
                computeInputOutput(unit, "hidden");
            }

            // compute the net input, output and error of output units with respect to the previous layer

            for (Neuron unit : trainedNetwork.getOutPutLayer().values()) {
                computeInputOutput(unit, "output");

                this.insertConfusionMatrix(unit.getExpectedOutput(), unit.getOutput());

                String predictedClass;
                if( unit.getOutput() >= 0.5){
                    predictedClass = "Popular";
                }
                else{
                    predictedClass = "Not Popuplar";
                }

                System.out.println("Tuple " + count[0] +" Expected Output: "+unit.getExpectedOutput() +" Output: "+unit.getOutput() +" Predicted Class: "+predictedClass);
            }


            count[0]++;

        });

        recognitionRate = this.computeRecognitionRate();
        System.out.println("Recognition Rate: " +(recognitionRate * 100) +"%");
    }


}
