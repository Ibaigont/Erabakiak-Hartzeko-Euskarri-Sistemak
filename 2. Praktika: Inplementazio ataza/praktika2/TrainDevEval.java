package praktika2;


import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;


import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TrainDevEval {
	public static void main(String[] args) throws Exception{
		// 1. Argumentuen egiaztapena
		if (args.length!=3) {
			System.err.println("Erabilera: java -jar TrainDevEval.jar <train.arff> <dev.arff> <eval.txt>");
			return;
		}
	
		String trainPath = args[0];
		String devPath = args[1];
		String outPath = args[2];
		
		try {
			// 2. Datuak kargatu	
			DataSource trainSource = new DataSource(trainPath);
			Instances trainData = trainSource.getDataSet();
			
			DataSource devSource = new DataSource(devPath);
			Instances devData = devSource.getDataSet();
			
			// 3. klase indizea ezarri
			if (trainData.classIndex()==-1)
				trainData.setClassIndex(trainData.numAttributes()-1);
			if(devData.classIndex()==-1)
				devData.setClassIndex(devData.numAttributes()-1);
			// 4. Ereduak entrenatu
			NaiveBayes nB = new NaiveBayes();
			nB.buildClassifier(trainData);
			
			// 5. Ebaluazioa egin
			Evaluation eval = new Evaluation(trainData);
			eval.evaluateModel(nB, devData);
			
			// 6. Fitxategian idatzi emaitzak
			PrintWriter pw = new PrintWriter(new FileWriter(outPath));
			pw.println("=== EBALUAZIO TXOSTENA ===");
			pw.println("Data: " + LocalDateTime.now());
			pw.println("Argumentuak:");
			pw.println("Train: "+ trainPath);
			pw.println("Dev: "+ devPath);
			pw.println(eval.toMatrixString());
			pw.printf("Asmatze tasa: %.4f%n",eval.pctCorrect());
			pw.close();
			
			
			
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}
}
