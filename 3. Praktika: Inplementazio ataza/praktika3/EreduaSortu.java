package praktika3;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class EreduaSortu {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuak egiaztatu
		if (args.length!=3) {
			System.err.println("Erabilera: java -jar EreduaSortu.jar <data.arff>  <NB.model>  <KalitatearenEstimazioa.txt>");
			return;
		}
		
		String dataPath = args[0];
		String modelPath = args[1];
		String evalPath = args[2];
		
		try {
			// 2. Datuak kargatu
			DataSource fitxategi = new DataSource(dataPath);
			Instances data = fitxategi.getDataSet();
			
			// 3. Klasea ezarri
			if (data.classIndex()==-1)
				data.setClassIndex(data.numAttributes()-1);
			// 4. Eredua sortu
			NaiveBayes nB = new NaiveBayes();
			nB.buildClassifier(data);
			SerializationHelper.write(modelPath, nB);
			
			// 5. Txostena egin
			PrintWriter writer = new PrintWriter(new FileWriter(evalPath));
			writer.println("=== EBALUAZIO TXOSTENA ===");
            writer.println("Data: " + LocalDateTime.now());
            writer.println("Argumentuak:");
            writer.println(" - Input Data: " + dataPath);
            writer.println(" - Model Output: " + modelPath);
            writer.println(" - Eval Output: " + evalPath);
            
            // k-Fold Cross Validation k=10
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(nB, data, 10, new Random(1));
            writer.println("=== 1. ESKEMA: 10-Fold Cross Validation ===");
            writer.println(eval.toMatrixString());
            writer.printf("Asmatze tasa: %.4f%n", eval.pctCorrect());
            writer.println();
            
            // Hold-out (70%train- 30%test)
            Instances dataRandom = new Instances(data);
            dataRandom.randomize(new Random(1));
                      
            int trainTamaina = (int) Math.round(dataRandom.numInstances()*0.7);
            int testTamaina = dataRandom.numInstances()-trainTamaina;
            
            Instances trainSplit = new Instances(dataRandom, 0, trainTamaina);
            Instances testSplit = new Instances(dataRandom, trainTamaina, testTamaina);
            
            NaiveBayes nBHO = new NaiveBayes();
            nBHO.buildClassifier(trainSplit);
            Evaluation evalHO = new Evaluation(trainSplit);
            evalHO.evaluateModel(nBHO, testSplit);
            writer.println("=== 2. ESKEMA: Hold-Out (%70 Train / %30 Test) ===");
            writer.println(evalHO.toMatrixString());
            writer.printf("Asmatze tasa: %.4f%n", evalHO.pctCorrect());
            writer.close();
            
            
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
