package praktika2;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;

public class StratifiedHoldOut {
	public static void main(String[] args) throws Exception{
		// 1. argumentuaren egiaztapena
		if (args.length<3) {
			System.err.println("Erabilera: java -jar StratifiedHoldOut.jar <data.arff> <train.arff> <eval.txt>");
			return;
		}
		// 2. Datuak kargatu
		String inputPath = args[0];
		String trainPath = args[1];
		String outputPath = args[2];
		
		try {
			DataSource dataSource = new DataSource(inputPath);
			Instances data = dataSource.getDataSet();
			
			//3. klase atributua definitu
			if(data.classIndex()==-1)
				data.setClassIndex(data.numAttributes()-1);
			
			// 4. Datuak estratifikatu eta randomizatu
			data.randomize(new Random(1));
			
			data.stratify(5); // %20-ka banatu nahi dugunez 5 aldiz estratifikatu
			
			// 5. Multzoak banatu
			Instances testData = data.testCV(5, 0);
			Instances trainData = data.trainCV(5, 0);
					
			// 6. Fitxategiak gorde
			saveArff(trainData,trainPath);
			System.out.println("Train multzoa: "+ trainData.numInstances()+" instantzia");
		
			// 7. Sailkatzailea eta ebaluazio objektua sortu
			Classifier cls = new NaiveBayes();
			cls.buildClassifier(trainData);
			Evaluation eval = new Evaluation(trainData);
			eval.evaluateModel(cls, trainData);
			
			// 8. eval.txt fitxategia sortu eta idatzi
			PrintWriter pw = new PrintWriter(outputPath);
			pw.println("Ebaluazioaren emaitzak (Hold-Out 80/20)");
			pw.println();
			
			pw.println(eval.toSummaryString());
			pw.println();
			pw.println(eval.toClassDetailsString());
			pw.println();
			pw.println(eval.toMatrixString());
			
			pw.close();
			
		}catch (Exception e) {
			e.getStackTrace();
		}
	}
	public static void saveArff(Instances data, String path) throws Exception {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(path));
		saver.writeBatch();
		
	}
}
