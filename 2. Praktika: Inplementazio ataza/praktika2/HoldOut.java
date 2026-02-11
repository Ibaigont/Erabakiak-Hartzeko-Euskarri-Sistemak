package praktika2;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class HoldOut {
	public static void main(String[] args) throws Exception {
		// 1. argumentuaren egiaztapena
		if (args.length<2) {
			System.err.println("Erabilera: java -jar HoldOut.jar <data.arff> <eval.txt>");
			return;
		}
		String inputPath = args[0];
		String outputPath = args[1];
		
		try {
			// 2. Datuak kargatu
			DataSource fitxategi = new DataSource(inputPath);
			Instances data = fitxategi.getDataSet();
			// Klasea ezarri
			if (data.classIndex()==-1) {
				data.setClassIndex(data.numAttributes()-1);
			}
			//3. HoldOut prestatu: Randomizatu eta zatitu (%66-an)
			data.randomize(new Random(1));
			
			int trainTamaina= (int) (Math.round(data.numInstances()*0.66));
			int testTamaina= data.numInstances()-trainTamaina;
			
			Instances train = new Instances(data, 0, trainTamaina);
			Instances test = new Instances(data, trainTamaina, testTamaina);
			
			// 4. Eredua entrenatu Naive Bayes-ekin
			NaiveBayes nB = new NaiveBayes();
			nB.buildClassifier(train);
			
			// 5. Eredua burutu Test-arekin
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(nB, test);
			
			// 6. Klase minoritarioa identifikatu
			int minClassIndex = -1;
			double minCount = Double.MAX_VALUE;
			int[] classCount = data.attributeStats(data.classIndex()).nominalCounts;
			
			for (int i=0; i< classCount.length;i++) {
				if (classCount[i]<minCount) {
					minCount= classCount[i];
					minClassIndex=i;
				}
			}
			String minClassName = data.classAttribute().value(minClassIndex);
			
			// 7. Emaitzak fitxategian idatzi
			
			PrintWriter pw = new PrintWriter(new FileWriter(outputPath));
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");

            // a) Data eta Argumentuak
            pw.println("Exekuzio data: " + dtf.format(LocalDateTime.now()));
            pw.println("Argumentuak: Input=" + inputPath + ", Output=" + outputPath);
            pw.println("--------------------------------------------------");
            
            // b) klase minoritarioaren metrikak
            pw.println("Klase Minoriotarioa: " + minClassName);
            pw.printf("Precision: %.4f%n", eval.precision(minClassIndex));
            pw.printf("Recall: %.4f%n", eval.recall(minClassIndex));
            pw.printf("F-score: %.4f%n", eval.fMeasure(minClassIndex));
            
            // c) Weighted Avg metrikak
            pw.println("Weighted Avg Metrikak: ");
            pw.printf("Precision: %.4f%n", eval.weightedPrecision());
            pw.printf("Recall: %.4f%n", eval.weightedRecall());
            pw.printf("F-score: %.4f%n", eval.weightedFMeasure());
            
            // d) Nahasmen Matrizea
            pw.println("Nahasmen Matrizea");
            pw.println(eval.toMatrixString());
            
            pw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
